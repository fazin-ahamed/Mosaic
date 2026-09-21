"""Source L41/L42 expert matrix on CPU; reference evidence only, never D2 timing.

Run: .venv-cu/bin/python -m mosaicvm.bench.d2_cpu_expert
Uses synthetic BF16 inputs and a declared route weight, not captured routes.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mosaicvm.bench.source_corpus_gen import build_kernel_module
from mosaicvm.bench.vnni_v4_race import LIB
from mosaicvm.bench.vnni_source import prepare_weights, quantized_gemm, self_check
from mosaicvm.kernels.fp4_source import act_quant_ref, reference_fp4_linear_source_semantics
from mosaicvm.runtime.loader import ShardIndex


def signed_activation_check():
    # Existing v4-B has no signed-byte correction: -1 becomes unsigned 255.
    a = np.zeros((1, 128), np.int8)
    a[0, :32] = -1
    weights = np.full((8, 128), 2, np.int8)  # qW=2 means source W=1.
    tiled_w = np.empty((1, 32, 32), np.int8)
    tiled_a = np.empty((1, 32, 32), np.uint8)
    y = np.empty((1, 8), np.float32)
    LIB.tile_weights(weights, tiled_w, 8, 128)
    LIB.tile_acts(a, tiled_a, 1, 128)
    LIB.vnni_gemm_tiled(tiled_a, np.ones((1, 1), np.float32), tiled_w,
                       np.ones((8, 4), np.float32), y, 1, 8, 128)
    assert np.all(y == 8160), y  # regression describing the current interface
    return {'expected_signed_dot': -32, 'observed_dot': float(y[0, 0]),
            'signed_activation_supported': False}


def run_expert(snapshot, layer, expert_id, rows, signed, adapter_checks):
    source = (snapshot / 'inference/model.py').read_text()
    expert_ast = next(n for n in ast.parse(source).body
                      if isinstance(n, ast.ClassDef) and n.name == 'Expert')
    index = ShardIndex(snapshot)
    tensors = {}
    hashes = {}
    for linear in ('w1', 'w3', 'w2'):
        pair = []
        for suffix in ('weight', 'scale'):
            name = f'layers.{layer}.ffn.experts.{expert_id}.{linear}.{suffix}'
            t = index.load_tensor(name)
            hashes[name] = hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
            pair.append(t)
        packed, scales = pair
        assert packed.view(torch.uint8).ndim == scales.ndim == 2
        assert packed.shape[0] == scales.shape[0]
        assert packed.shape[1] == scales.shape[1] * 16
        # Do not silently inherit the reference helper's 0/255 scale semantics.
        assert ((scales.view(torch.uint8) > 0) & (scales.view(torch.uint8) < 255)).all()
        tensors[linear] = pair
    prepared = {name: prepare_weights(*pair) for name, pair in tensors.items()}
    stages = {}
    shim = build_kernel_module()

    class Linear(nn.Module):
        def __init__(self, in_features, out_features, dtype=None):
            super().__init__()
            self.name = ('w1', 'w2', 'w3')[len(created)]
            created.append(self)

        def forward(self, x):
            packed, scales = tensors[self.name]
            q, sa = act_quant_ref(x)
            sq, ss = shim.act_quant(x, scale_fmt='ue8m0')
            assert torch.equal(q.view(torch.uint8), sq.view(torch.uint8))
            assert torch.equal(sa, ss)
            qf = q.float()
            if mode[0] == 'ordered':
                stages[self.name] = {
                    'negative_codes': int((qf < 0).sum()),
                    'fractional_codes': int((qf != qf.trunc()).sum()),
                    'codes_above_u8': int((qf > 255).sum()),
                    'max_abs_code': float(qf.abs().max()),
                }
                y = reference_fp4_linear_source_semantics(x, packed.view(torch.uint8), scales)
                ordered[self.name] = y
            else:
                y = (shim.fp4_gemm(q, sa, packed, scales) if mode[0] == 'shim' else
                     quantized_gemm(q, sa, prepared[self.name]).to(torch.bfloat16))
                delta = (y.float() - ordered[self.name].float()).abs()
                stages[self.name][mode[0] + '_vs_ordered'] = {
                    'bit_equal': bool(torch.equal(y.view(torch.int16), ordered[self.name].view(torch.int16))),
                    'different_elements': int((delta != 0).sum()),
                    'max_abs_error': float(delta.max()),
                }
            assert torch.isfinite(y).all()
            return y

    created, ordered, mode = [], {}, ['ordered']
    namespace = {'nn': nn, 'torch': torch, 'F': F, 'Linear': Linear, 'Optional': __import__('typing').Optional}
    exec(compile(ast.Module(body=[expert_ast], type_ignores=[]), str(snapshot / 'inference/model.py'), 'exec'), namespace)
    config = json.loads((snapshot / 'config.json').read_text())
    expert = namespace['Expert'](4096, 2048, swiglu_limit=config['swiglu_limit'])
    x = torch.randn((rows, 4096), generator=torch.Generator().manual_seed(20260907)).to(torch.bfloat16)
    route = torch.linspace(0.05, 1.5, rows, dtype=torch.float32).reshape(rows, 1)
    with torch.no_grad():
        y = expert(x, route)
        mode[0] = 'shim'
        ys = expert(x, route)
        mode[0] = 'vnni_source'
        yv = expert(x, route)
    result = {
        'classification': 'LOCAL_CPU_REAL_EXPERT_SYNTHETIC_INPUT_REFERENCE_ONLY',
        'production_correct': False,
        'source': str(snapshot.resolve()),
        'expert': f'layers.{layer}.ffn.experts.{expert_id}', 'rows': rows,
        'input': 'seeded synthetic BF16, seed=20260907', 'route_weights': route.flatten().tolist(),
        'expert_forward': 'checkpoint AST executed unchanged; source clamp/SwiGLU/route placement/casts',
        'source_model_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'tensor_sha256': hashes,
        'source_tensor_bytes': sum(t.numel() * t.element_size() for pair in tensors.values() for t in pair),
        'stages': stages,
        'final_shim_vs_ordered_bit_equal': bool(torch.equal(y.view(torch.int16), ys.view(torch.int16))),
        'final_vnni_vs_ordered_bit_equal': bool(torch.equal(y.view(torch.int16), yv.view(torch.int16))),
        'vnni_source_adapter_checks': adapter_checks,
        'vnni_prepared_weight_bytes': sum(a.nbytes for group in prepared.values() for a in group),
        'vnni_signed_regression': signed,
        'blockers': ['no source GPU kernel comparison or full routed/shared MoE reduction',
                     'synthetic input and route; no held-out acceptance or production latency measured'],
    }
    result['gate_pass'] = (result['final_vnni_vs_ordered_bit_equal'] and
                           all(stage['vnni_source_vs_ordered']['bit_equal'] for stage in stages.values()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, default=Path.home() / (
        '.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/'
        'snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062'))
    p.add_argument('--output', type=Path)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--matrix', action='store_true', help='L41/L42 experts 0/1; M1/2/3/4/8/16')
    args = p.parse_args()
    if args.output is None:
        args.output = Path('artifacts/d2e_matrix_v3.json' if args.matrix else
                           'artifacts/d2_cpu_expert_v3.json')
    torch.set_num_threads(2)
    signed = signed_activation_check()
    adapter_checks = self_check()
    if args.self_test:
        print(json.dumps({'historical_signed_mapping': signed, 'adapter': adapter_checks}, indent=2))
        return
    cases = ([(layer, expert_id, rows) for layer in (41, 42)
              for expert_id in (0, 1) for rows in (1, 2, 3, 4, 8, 16)]
             if args.matrix else [(41, 0, 1)])
    results = []
    for layer, expert_id, rows in cases:
        result = run_expert(args.snapshot, layer, expert_id, rows, signed, adapter_checks)
        results.append(result)
        gc.collect()  # Release nested Expert/Linear closure cycles before the next expert.
        output = ({'classification': 'LOCAL_CPU_SYNTHETIC_REAL_EXPERT_MATRIX_REFERENCE_ONLY',
                   'complete': len(results) == len(cases), 'cases': results,
                   'gate_pass': all(case['gate_pass'] for case in results)}
                  if args.matrix else result)
        args.output.write_text(json.dumps(output, indent=2) + '\n')
        print(f"L{layer} E{expert_id} M{rows}: gate_pass={result['gate_pass']}", flush=True)
    assert all(case['gate_pass'] for case in results), 'VNNI mismatch; see artifact'


if __name__ == '__main__':
    main()
