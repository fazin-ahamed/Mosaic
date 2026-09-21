"""Pinned source L41/L42 replay; correctness harness, not production timing."""
from __future__ import annotations

from collections import OrderedDict
import dataclasses
import importlib
import json
from pathlib import Path
import sys
import types

import torch
from safetensors import safe_open

from mosaicvm.bench.source_corpus_gen import build_kernel_module
from mosaicvm.bench.d1b_tail_robustness import reset_decode_state
from mosaicvm.model.dsa import hadamard

SNAPSHOT = Path.home() / ('.cache/huggingface/hub/models--deepseek-ai--'
    'DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062')


class SourceTail:
    def __init__(self, snapshot=SNAPSHOT, device='cpu', backend='reference', max_seq=256, cache_cap=24):
        if backend not in ('reference', 'vnni') or (backend == 'vnni' and device != 'cpu'):
            raise ValueError('VNNI requires CPU; backend must be reference or vnni')
        if max_seq < 1 or cache_cap < 1:
            raise ValueError('Positive max_seq and cache_cap required')
        self.snapshot, self.device, self.backend = Path(snapshot), device, backend
        self.max_seq, self.cache_cap, self.cache = max_seq, cache_cap, OrderedDict()
        sys.path.insert(0, str(self.snapshot / 'inference'))
        sys.modules['kernel'] = build_kernel_module()
        fht = types.ModuleType('fast_hadamard_transform')
        fht.hadamard_transform = lambda x, scale=None: hadamard(x)
        sys.modules['fast_hadamard_transform'] = fht
        self.km = importlib.import_module('model')
        cfg = json.loads((self.snapshot / 'inference/config.json').read_text())
        fields = {f.name for f in dataclasses.fields(self.km.ModelArgs)}
        args = self.km.ModelArgs(**{k: v for k, v in cfg.items() if k in fields})
        args.max_batch_size, args.max_seq_len = 1, max_seq
        self.km.world_size, self.km.rank = 1, 0
        self.km.default_dtype = torch.float8_e4m3fn if args.dtype == 'fp8' else torch.bfloat16
        self.km.scale_fmt = 'ue8m0' if args.scale_dtype == 'fp8' else args.scale_fmt
        self.km.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == 'fp8' else torch.float32
        # Match source benchmark creation and forward defaults, including RoPE buffers.
        torch.set_default_dtype(torch.bfloat16)
        original_empty = torch.empty
        def empty(*a, **kw):
            if kw.get('dtype') == torch.float4_e2m1fn_x2:
                kw['device'] = 'meta'
            return original_empty(*a, **kw)
        torch.empty = empty
        try:
            with torch.device('cpu'):
                self.blocks = torch.nn.ModuleList([self.km.Block(l, args) for l in (41, 42)])
        finally:
            torch.empty = original_empty
        tail = self
        class RoutedLinear(torch.nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name
            def forward(self, x):
                value = tail.expert_weight(self.name)
                if tail.backend == 'vnni':
                    from mosaicvm.bench.vnni_source import quantized_gemm
                    q, sa = tail.km.act_quant(x, 128, tail.km.scale_fmt, tail.km.scale_dtype)
                    shape = x.shape[:-1]
                    return quantized_gemm(q.reshape(-1, q.shape[-1]),
                                          sa.reshape(-1, sa.shape[-1]), value).to(x.dtype).reshape(*shape, -1)
                return tail.km.linear(x, value)
        for l, block in zip((41, 42), self.blocks):
            for e, expert in enumerate(block.ffn.experts):
                for name in ('w1', 'w2', 'w3'):
                    setattr(expert, name, RoutedLinear(f'layers.{l}.ffn.experts.{e}.{name}'))
        self.index = json.loads((self.snapshot / 'model.safetensors.index.json').read_text())['weight_map']
        for l, block in zip((41, 42), self.blocks):
            for name, param in block.named_parameters():
                key = f'layers.{l}.{name}'
                t = self.tensor(key)
                if name.endswith('wo_a.weight'):
                    scale = self.tensor(key.replace('.weight', '.scale')).float()
                    t = (t.float() * scale.repeat_interleave(t.shape[0] // scale.shape[0], 0)
                         .repeat_interleave(t.shape[1] // scale.shape[1], 1)).to(torch.bfloat16)
                param.data = t.to(device=device, dtype=param.dtype)
            for buf in block.buffers():
                buf.data = buf.data.to(device)
        self.norm = self.km.RMSNorm(args.dim, args.norm_eps)
        self.norm.weight.data = self.tensor('norm.weight').to(device=device, dtype=torch.float32)
        self.head = self.tensor('head.weight').float()
        self.hc = [self.tensor(n).to(device=device, dtype=torch.float32)
                   for n in ('hc_head_fn', 'hc_head_scale', 'hc_head_base')]
        self.blocks.eval()
        self.route_rows = {41: [], 42: []}
        for l, block in zip((41, 42), self.blocks):
            block.ffn.gate.register_forward_hook(
                lambda mod, args_, out, layer=l: self.route_rows[layer].append(out[1].detach().cpu()))

    def tensor(self, name):
        with safe_open(str(self.snapshot / self.index[name]), framework='pt') as f:
            return f.get_tensor(name)

    def expert_weight(self, name):
        key = (self.backend, name)
        if key not in self.cache:
            w, s = self.tensor(name + '.weight'), self.tensor(name + '.scale')
            if self.backend == 'vnni':
                from mosaicvm.bench.vnni_source import prepare_weights
                value = prepare_weights(w, s)
            else:
                value = w.view(torch.float4_e2m1fn_x2).to(self.device)
                value.scale = s.to(self.device)
            self.cache[key] = value
            if len(self.cache) > self.cache_cap:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    @torch.inference_mode()
    def replay(self, h40, tokens, S):
        if h40.ndim != 3 or h40.shape[1:] != (4, 4096) or not 0 < S <= len(h40) <= self.max_seq:
            raise ValueError('Expected full [positions,4,4096] trajectory with valid committed prefix')
        if len(tokens) < len(h40) or not torch.isfinite(h40).all():
            raise ValueError('Need finite states and an input token for every state')
        reset_decode_state(self.blocks)
        for rows in self.route_rows.values():
            rows.clear()
        x = h40.to(device=self.device, dtype=torch.bfloat16)
        ids = torch.tensor(tokens[:len(h40)], device=self.device, dtype=torch.long).unsqueeze(0)
        outputs = []
        for start, end in [(0, S)] + [(p, p + 1) for p in range(S, len(h40))]:
            h = x[start:end].unsqueeze(0)
            with torch.device(self.device):
                for block in self.blocks:
                    h = block(h, start, ids[:, start:end])
            outputs.append(h.squeeze(0).cpu())
        return torch.cat(outputs)

    @torch.inference_mode()
    def logits(self, h42):
        h = self.blocks[-1].hc_head(h42.to(device=self.device, dtype=torch.bfloat16).unsqueeze(0), *self.hc)
        h = self.norm(h).squeeze(0).float().cpu()
        return torch.nn.functional.linear(h, self.head)
