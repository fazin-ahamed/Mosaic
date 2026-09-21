"""Ordered K-split vs single-warp qpa_gemm_pre (Phase 2.8 Step 6)."""
from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.fp4_source import pack_mxfp4_nk
from mosaicvm.kernels.qpa_mma import qpa_mma_gemm
from mosaicvm.kernels.xsimt import decode_activation_exact
from mosaicvm.kernels.fp4_source import act_quant_ref


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.float()
    bf = b.float()
    # bf16 bitwise
    au = a.view(torch.uint16)
    bu = b.view(torch.uint16)
    eq = (au == bu)
    frac = float(eq.float().mean())
    # ULP in bf16 space: abs of uint16 difference on finite values
    ulp = (au.to(torch.int32) - bu.to(torch.int32)).abs()
    finite = torch.isfinite(af) & torch.isfinite(bf)
    max_ulp = int(ulp[finite].max().item()) if finite.any() else 0
    max_abs = float((af - bf).abs().max())
    denom = bf.abs().clamp_min(1e-8)
    max_rel = float(((af - bf).abs() / denom).max())
    nan_inf = int((~torch.isfinite(af) ^ ~torch.isfinite(bf)).sum().item())
    return {
        "bf16_bitwise_frac": frac,
        "max_bf16_ulp": max_ulp,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nan_inf_mismatches": nan_inf,
        "torch_equal": bool(torch.equal(a, b)),
    }


@pytest.mark.parametrize("m,k,n,warps", [
    (1, 128, 16, 2),
    (2, 128, 32, 4),
    (8, 256, 64, 8),
    (4, 128, 32, 8),
])
def test_ksplit_vs_single_warp(m, k, n, warps):
    torch.manual_seed(0)
    w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    xf8, sA = act_quant_ref(x)
    x_norm = decode_activation_exact((xf8, sA))
    ref = qpa_mma_gemm(x, p, s, n, qpa_k=1, x_norm=x_norm, sA=sA)
    got = qpa_mma_gemm(x, p, s, n, qpa_k=warps, x_norm=x_norm, sA=sA, ordered=True)
    met = _metrics(got, ref)
    print(f"W{warps} M={m} K={k} N={n} {met}")
    # K=32 (one group) and W8 must be bitwise; others record numeric_parity
    if k == 32 or warps == 8:
        assert met["torch_equal"], met
    else:
        assert met["nan_inf_mismatches"] == 0
        assert met["max_abs"] < 1e-2, met
