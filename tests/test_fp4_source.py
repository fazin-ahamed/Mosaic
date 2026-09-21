from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.fp4_source import (
    ACT_FP8_MAX,
    act_quant_legacy_phase27,
    act_quant_ref,
    pack_mxfp4_nk,
    reference_fp4_linear_source_semantics,
)
from mosaicvm.kernels.qpa_mma import qpa_mma_gemm
from mosaicvm.kernels.xsimt import xsimt_gemm


def _official_act_quant(x: torch.Tensor, block: int = 128):
    """DeepSeek V4-0731: E8M0 pow2 scale via IEEE bits, ceil(log2(amax/448))."""
    m, k = x.shape
    xf = x.to(torch.float32).view(m, k // block, block)
    amax = xf.abs().amax(dim=-1).clamp_min(1e-4)
    bits = (amax * (1.0 / ACT_FP8_MAX)).view(torch.int32)
    pow_exp = ((bits >> 23) & 0xFF) - 127 + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    s = ((pow_exp + 127) << 23).view(torch.float32)
    y = (xf / s.unsqueeze(-1)).clamp(-ACT_FP8_MAX, ACT_FP8_MAX).reshape(m, k).to(torch.float8_e4m3fn)
    return y, s


def test_act_quant_matches_official_math():
    torch.manual_seed(0)
    x = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
    y, s = act_quant_ref(x, 128)
    y0, s0 = _official_act_quant(x, 128)
    assert torch.equal(s, s0), (s - s0).abs().max().item()
    finite = torch.isfinite(y.float()) & torch.isfinite(y0.float())
    assert torch.equal(y.view(torch.uint8)[finite], y0.view(torch.uint8)[finite])


def test_act_quant_pow2_scale_boundaries():
    vals = [0.0, 1e-5, -1e-5, 1e-4, 448.0, -448.0, 1.0, 2.0 ** -60]
    # exact pow2 and straddling-pow2 amax cases
    vals += [2.0 ** e * 448.0 for e in (-10, -1, 0, 4)]
    vals += [2.0 ** e * 448.0 * 1.0000001 for e in (-10, -1, 0, 4)]
    x = torch.zeros(1, 128 * len(vals), dtype=torch.float32, device="cuda")
    for i, v in enumerate(vals):
        if v != 0.0:
            x[0, 128 * i + 7] = v
    y, s = act_quant_ref(x.to(torch.bfloat16), 128)
    # scale must be an exact power of two and round-trip through E8M0
    bits = s.view(torch.int32)
    assert torch.equal(bits & ((1 << 23) - 1), torch.zeros_like(bits))
    assert torch.equal(s, s.to(torch.float8_e8m0fnu).float())
    y0, s0 = _official_act_quant(x.to(torch.bfloat16), 128)
    assert torch.equal(s, s0)
    # legacy phase27 differs on straddling cases (documents the fix)
    _, sl = act_quant_legacy_phase27(x.to(torch.bfloat16), 128)
    assert not torch.equal(sl, s0)


def test_pack_mxfp4_nk_shapes():
    torch.manual_seed(0)
    n, k = 64, 128
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    assert p.dtype == torch.uint8
    assert p.shape == (n, k // 2)
    assert s.shape == (n, k // 32)


@pytest.mark.parametrize("m,k,n", [(1, 128, 16), (2, 256, 32), (8, 128, 16)])
def test_source_linear_vs_reference(m, k, n):
    torch.manual_seed(0)
    w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    ref = reference_fp4_linear_source_semantics(x, p, s)
    ox = xsimt_gemm(x, p, s, n)
    oq = qpa_mma_gemm(x, p, s, n)
    err_x = (ox.float() - ref.float()).abs().max().item()
    err_q = (oq.float() - ref.float()).abs().max().item()
    assert err_x < 0.05, f"xsimt max abs {err_x}"
    assert err_q < 0.05, f"qpa max abs {err_q}"
