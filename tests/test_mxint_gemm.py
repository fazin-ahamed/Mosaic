from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.fp4_source import pack_mxfp4_nk, reference_fp4_linear_source_semantics
from mosaicvm.kernels.mxint import fused_moe_mxint, mxint_dp4a_gemm, mxint_imma_gemm, prepack_expert_bank_mxint


def _metrics(got, ref):
    au, bu = got.view(torch.uint16), ref.view(torch.uint16)
    af, bf = got.float(), ref.float()
    fin = torch.isfinite(af) & torch.isfinite(bf)
    nan_inf = int((~torch.isfinite(af) ^ ~torch.isfinite(bf)).sum().item())
    ulp = int((au.to(torch.int32) - bu.to(torch.int32)).abs()[fin].max().item()) if fin.any() else 0
    return {
        "bf16_bitwise_frac": float((au == bu).float().mean()),
        "max_bf16_ulp": ulp,
        "max_abs": float((af - bf).abs().max()),
        "max_rel": float(((af - bf).abs() / bf.abs().clamp_min(1e-8)).max()),
        "nan_inf_mismatches": nan_inf,
        "kernel_numeric": "bitwise_mosaic" if torch.equal(got, ref) else "numeric_parity",
    }


@pytest.mark.parametrize("m,k,n", [(1, 128, 16), (2, 256, 32), (3, 128, 16)])
def test_mxint_dp4a_vs_reference(m, k, n):
    torch.manual_seed(0)
    w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    ref = reference_fp4_linear_source_semantics(x, p, s)
    got = mxint_dp4a_gemm(x, p, s, n)
    met = _metrics(got, ref)
    assert met["nan_inf_mismatches"] == 0, met
    assert met["max_abs"] < 0.05, met


@pytest.mark.parametrize("m,k,n", [(4, 128, 16), (8, 128, 16)])
def test_mxint_imma_w8_vs_reference(m, k, n):
    torch.manual_seed(0)
    w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    ref = reference_fp4_linear_source_semantics(x, p, s)
    got = mxint_imma_gemm(x, p, s, n, qpa_k=8, ordered=True)
    met = _metrics(got, ref)
    assert met["nan_inf_mismatches"] == 0, met
    assert met["max_abs"] < 0.05, met


def test_mxint_imma_w8_vs_w1_kernel_numeric():
    torch.manual_seed(0)
    n, k, m = 16, 128, 8
    w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    w1 = mxint_imma_gemm(x, p, s, n, qpa_k=1, ordered=True)
    w8 = mxint_imma_gemm(x, p, s, n, qpa_k=8, ordered=True)
    met = _metrics(w8, w1)
    assert met["nan_inf_mismatches"] == 0, met
    # integer reduction is exact; last-bit diffs vs QPA are expected, not vs W1
    assert torch.equal(w8, w1) or met["max_abs"] < 1e-5, met


def test_fused_moe_mxint_two_run_equal():
    torch.manual_seed(2)
    k, top_k, e, h, inter = 4, 2, 8, 256, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    pgu, sgu, pdu, sdu = [], [], [], []
    for _ in range(e):
        pg, sg = pack_mxfp4_nk((torch.randn(2 * inter, h, device="cuda") / math.sqrt(h)).to(torch.bfloat16))
        pd, sd = pack_mxfp4_nk((torch.randn(h, inter, device="cuda") / math.sqrt(inter)).to(torch.bfloat16))
        pgu.append(pg); sgu.append(sg); pdu.append(pd); sdu.append(sd)
    packed_gu = torch.stack(pgu)
    scale_gu = torch.stack(sgu)
    packed_d = torch.stack(pdu)
    scale_d = torch.stack(sdu)
    y1 = fused_moe_mxint(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d)
    y2 = fused_moe_mxint(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d)
    assert torch.equal(y1, y2)
    assert y1.shape == (k, h)


def test_fused_moe_mxint_no_item_in_dispatch():
    torch.manual_seed(3)
    k, top_k, e, h, inter = 4, 2, 4, 128, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    pgu, sgu, pdu, sdu = [], [], [], []
    for _ in range(e):
        pg, sg = pack_mxfp4_nk((torch.randn(2 * inter, h, device="cuda") / math.sqrt(h)).to(torch.bfloat16))
        pd, sd = pack_mxfp4_nk((torch.randn(h, inter, device="cuda") / math.sqrt(inter)).to(torch.bfloat16))
        pgu.append(pg); sgu.append(sg); pdu.append(pd); sdu.append(sd)
    packed_gu = torch.stack(pgu); scale_gu = torch.stack(sgu)
    packed_d = torch.stack(pdu); scale_d = torch.stack(sdu)
    bank = prepack_expert_bank_mxint(packed_gu, scale_gu, packed_d=packed_d, scale_d=scale_d)
    n = {"n": 0}
    orig = torch.Tensor.item
    def counted(self):
        n["n"] += 1
        return orig(self)
    torch.Tensor.item = counted
    try:
        fused_moe_mxint(x, ids, weights, bank=bank)
    finally:
        torch.Tensor.item = orig
    assert n["n"] == 0, n["n"]
