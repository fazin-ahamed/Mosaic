"""CPU source-activation adapter for VNNI; historical integer kernels unchanged.

Run bounded checks: OMP_NUM_THREADS=2 .venv-cu/bin/python -m mosaicvm.bench.vnni_source
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from mosaicvm.kernels.fp4_source import act_quant_ref, reference_fp4_linear_source_semantics
from mosaicvm.kernels.mxint import qint_encode_w

HERE = Path(__file__).parent
_LIB = None


def library():
    global _LIB
    if _LIB is None:
        source = HERE / 'vnni_source_kernel.c'
        binary = HERE / 'libvnni_source.so'
        if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(['gcc', '-O3', '-shared', '-fPIC', '-mavxvnni', '-fopenmp',
                            '-ffp-contract=off', str(source), '-o', str(binary)], check=True)
        _LIB = ctypes.CDLL(str(binary))
        ptr = lambda dtype: np.ctypeslib.ndpointer(dtype, flags='C_CONTIGUOUS')
        _LIB.vnni_source_gemm.argtypes = [ptr(np.uint8), ptr(np.float32), ptr(np.int8),
                                        ptr(np.float32), ptr(np.int32), ptr(np.float32)] + [ctypes.c_int] * 3
        _LIB.vnni_source_gemm.restype = None
    return _LIB


def prepare_weights(packed, scales):
    if packed.device.type != 'cpu' or scales.device.type != 'cpu':
        raise ValueError('CPU tensors required')
    packed = packed.contiguous().view(torch.uint8)
    if packed.ndim != 2 or packed.shape[0] % 8 or packed.shape[1] % 64:
        raise ValueError('Expected packed [N,K/2] with N divisible by 8 and K by 128')
    n, half_k = packed.shape
    if scales.shape != (n, half_k // 16) or scales.dtype != torch.float8_e8m0fnu:
        raise ValueError('Expected source E8M0 [N,K/32] scales')
    sb = scales.view(torch.uint8)
    if not ((sb > 0) & (sb < 255)).all():
        raise ValueError('Scale 0/255 semantics require separate certification')
    qw = qint_encode_w(packed).numpy()
    # Existing v4-B layout: output groups of 8, K groups of 4, then lane and K.
    tiled = qw.reshape(n // 8, 8, half_k // 2, 4).transpose(0, 2, 1, 3).copy()
    sums = qw.reshape(n, half_k // 16, 32).sum(-1, dtype=np.int32)
    return tiled, scales.float().contiguous().numpy(), sums


def quantized_gemm(q, sa, weights):
    if q.device.type != 'cpu' or q.dtype != torch.float8_e4m3fn or q.ndim != 2:
        raise ValueError('Expected CPU E4M3 [M,K] activation codes')
    m, k = q.shape
    if not m or k % 128 or sa.shape != (m, k // 128) or sa.device.type != 'cpu':
        raise ValueError('Expected positive rows, K divisible by 128, CPU scales [M,K/128]')
    qf = q.float()
    if not torch.isfinite(qf).all() or not torch.isfinite(sa).all() or not (sa > 0).all():
        raise ValueError('Finite activations and positive finite scales required')
    # Every finite E4M3 code times 512 is an exact integer; no requantization.
    qa = (qf * 512).to(torch.int32).numpy()
    unsigned = qa + 262144
    planes = np.stack([(unsigned >> shift).astype(np.uint8) for shift in (0, 8, 16)])
    tiled_a = np.broadcast_to(planes.reshape(3, m, k // 4, 1, 4),
                             (3, m, k // 4, 8, 4)).copy()
    tiled_w, sw, sums = weights
    n = sw.shape[0]
    if tiled_w.shape != (n // 8, k // 4, 8, 4) or sw.shape != (n, k // 32) or sums.shape != sw.shape:
        raise ValueError('Prepared weight dimensions do not match activations')
    y = np.empty((m, n), dtype=np.float32)
    library().vnni_source_gemm(tiled_a, sa.float().contiguous().numpy(), tiled_w,
                               sw, sums, y, m, n, k)
    return torch.from_numpy(y)


def linear(x, weights):
    q, sa = act_quant_ref(x)
    return quantized_gemm(q, sa, weights).to(torch.bfloat16)


def self_check():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(7)
    all_codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
    finite = all_codes.float()[torch.isfinite(all_codes.float())]
    cases = [finite.repeat(3)[:256].reshape(2, 128),
             torch.full((1, 128), -448.), torch.full((1, 128), 448.),
             torch.full((1, 128), -1. / 512)]
    results = []
    for values in cases:
        q = values.to(torch.float8_e4m3fn)
        m, k = q.shape
        packed = torch.randint(0, 256, (16, k // 2), generator=generator, dtype=torch.uint8)
        scales = torch.randint(120, 132, (16, k // 32), generator=generator, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        sa = torch.full((m, k // 128), 1. / 128)
        weights = prepare_weights(packed, scales)
        actual = quantized_gemm(q, sa, weights)
        qa = (q.float().numpy() * 512).astype(np.int64)
        qw = qint_encode_w(packed).numpy().astype(np.int64)
        expected = np.zeros((m, 16), np.float32)
        for wb in range(k // 32):
            sl = slice(wb * 32, (wb + 1) * 32)
            partial = (qa[:, sl] @ qw[:, sl].T).astype(np.float32) / np.float32(1024)
            partial = partial * sa.numpy()[:, wb // 4, None] * scales.float().numpy()[None, :, wb]
            expected = expected + partial
        assert np.array_equal(actual.numpy().view(np.uint32), expected.view(np.uint32))
        results.append({'rows': m, 'integer_oracle_fp32_bit_equal': True})
    x = torch.randn((2, 128), generator=generator).to(torch.bfloat16)
    reference = reference_fp4_linear_source_semantics(x, packed, scales)
    actual = linear(x, weights)
    assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
    # Sweep every admitted scale byte with a small exactly representable dot.
    packed_edge = torch.full((256, 64), 0x22, dtype=torch.uint8)
    scale_bytes = torch.arange(1, 255, dtype=torch.uint8).repeat(2)[:256]
    scale_edge = scale_bytes[:, None].expand(256, 4).contiguous().view(torch.float8_e8m0fnu)
    q_edge = torch.zeros((1, 128), dtype=torch.float32)
    q_edge[0, 0] = 1. / 512
    edge = quantized_gemm(q_edge.to(torch.float8_e4m3fn), torch.ones((1, 1)),
                          prepare_weights(packed_edge, scale_edge))
    expected_edge = scale_edge.float()[:, 0] * (1. / 512)
    assert torch.equal(edge.view(torch.int32)[0], expected_edge.view(torch.int32))
    for bad in (0, 255):
        invalid = scale_edge.clone()
        invalid.view(torch.uint8)[0, 0] = bad
        try:
            prepare_weights(packed_edge, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Scale byte {bad} was accepted')
    for bad in (127, 255):  # E4M3 NaN encodings.
        invalid = q_edge.to(torch.float8_e4m3fn)
        invalid.view(torch.uint8)[0, 0] = bad
        try:
            quantized_gemm(invalid, torch.ones((1, 1)), prepare_weights(packed_edge, scale_edge))
        except ValueError:
            pass
        else:
            raise AssertionError(f'E4M3 NaN {bad} was accepted')
    return {'full_finite_e4m3_code_coverage': 254, 'integer_cases': results,
            'source_bf16_reference_bit_equal': True,
            'admitted_scale_bytes_tested': [1, 254], 'scale_0_255_rejected': True,
            'e4m3_nan_codes_rejected': True}


if __name__ == '__main__':
    print(json.dumps(self_check(), indent=2))
