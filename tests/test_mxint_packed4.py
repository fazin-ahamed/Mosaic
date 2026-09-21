from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.fp4_source import pack_mxfp4_nk, reference_fp4_linear_source_semantics
from mosaicvm.kernels.mxint import (
    MXINTHybridBank,
    fused_moe_mxint,
    mxint_dp4a_gemm,
    mxint_imma_gemm,
    prepack_expert_bank_mxint,
    prepack_expert_bank_mxint_packed4,
)


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
def test_mxint_dp4a_packed4_vs_reference(m, k, n):
    torch.manual_seed(0)
    w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    ref = reference_fp4_linear_source_semantics(x, p, s)
    got = mxint_dp4a_gemm(x, p, s, n, weight_repr="packed4")
    met = _metrics(got, ref)
    assert met["nan_inf_mismatches"] == 0, met
    assert met["max_abs"] < 0.05, met
    hot = mxint_dp4a_gemm(x, p, s, n, weight_repr="hot8")
    assert torch.equal(got, hot) or _metrics(got, hot)["max_abs"] < 0.05


@pytest.mark.parametrize("m,k,n", [(4, 128, 16), (8, 128, 16)])
def test_mxint_imma_packed4_vs_reference(m, k, n):
    torch.manual_seed(0)
    w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    ref = reference_fp4_linear_source_semantics(x, p, s)
    got = mxint_imma_gemm(x, p, s, n, qpa_k=8, ordered=True, weight_repr="packed4")
    met = _metrics(got, ref)
    assert met["nan_inf_mismatches"] == 0, met
    assert met["max_abs"] < 0.05, met
    hot = mxint_imma_gemm(x, p, s, n, qpa_k=8, ordered=True, weight_repr="hot8")
    assert torch.equal(got, hot) or _metrics(got, hot)["max_abs"] < 0.05


def _make_banks(e, h, inter):
    pgu, sgu, pdu, sdu = [], [], [], []
    for _ in range(e):
        pg, sg = pack_mxfp4_nk((torch.randn(2 * inter, h, device="cuda") / math.sqrt(h)).to(torch.bfloat16))
        pd, sd = pack_mxfp4_nk((torch.randn(h, inter, device="cuda") / math.sqrt(inter)).to(torch.bfloat16))
        pgu.append(pg); sgu.append(sg); pdu.append(pd); sdu.append(sd)
    return torch.stack(pgu), torch.stack(sgu), torch.stack(pdu), torch.stack(sdu)


def test_fused_moe_mxint_packed4_two_run_equal():
    torch.manual_seed(2)
    k, top_k, e, h, inter = 4, 2, 8, 256, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    packed_gu, scale_gu, packed_d, scale_d = _make_banks(e, h, inter)
    y1 = fused_moe_mxint(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d,
                         weight_repr="packed4")
    y2 = fused_moe_mxint(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d,
                         weight_repr="packed4")
    assert torch.equal(y1, y2)
    assert y1.shape == (k, h)


def test_fused_moe_mxint_reprs_agree():
    torch.manual_seed(4)
    k, top_k, e, h, inter = 4, 2, 8, 128, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    packed_gu, scale_gu, packed_d, scale_d = _make_banks(e, h, inter)
    y_hot = fused_moe_mxint(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d,
                            weight_repr="hot8")
    y_p4 = fused_moe_mxint(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d,
                           weight_repr="packed4")
    # same integer math; final bf16 may differ last-bit only
    d = (y_hot.float() - y_p4.float()).abs().max().item()
    assert d < 0.05, d


def test_fused_moe_mxint_hybrid_runs():
    torch.manual_seed(5)
    k, top_k, e, h, inter = 4, 2, 8, 128, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    packed_gu, scale_gu, packed_d, scale_d = _make_banks(e, h, inter)
    hot8 = prepack_expert_bank_mxint(packed_gu, scale_gu, packed_d=packed_d, scale_d=scale_d)
    p4 = prepack_expert_bank_mxint_packed4(packed_gu, scale_gu, packed_d=packed_d, scale_d=scale_d)
    bank = MXINTHybridBank(hot8=hot8, packed4=p4, hot_mask=None)
    y1 = fused_moe_mxint(x, ids, weights, bank=bank, weight_repr="hybrid")
    y2 = fused_moe_mxint(x, ids, weights, bank=bank, weight_repr="hybrid")
    assert torch.equal(y1, y2)
    assert bank.hot_mask is not None and bank.hot_mask.dtype == torch.uint8


def test_fused_moe_mxint_packed4_no_item_in_dispatch():
    torch.manual_seed(3)
    k, top_k, e, h, inter = 4, 2, 4, 128, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    packed_gu, scale_gu, packed_d, scale_d = _make_banks(e, h, inter)
    bank = prepack_expert_bank_mxint_packed4(packed_gu, scale_gu, packed_d=packed_d, scale_d=scale_d)
    n = {"n": 0}
    orig = torch.Tensor.item
    def counted(self):
        n["n"] += 1
        return orig(self)
    torch.Tensor.item = counted
    try:
        fused_moe_mxint(x, ids, weights, bank=bank, weight_repr="packed4")
    finally:
        torch.Tensor.item = orig
    assert n["n"] == 0, n["n"]


def test_fused_moe_mxint_stage_times():
    torch.manual_seed(6)
    k, top_k, e, h, inter = 4, 2, 8, 256, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    packed_gu, scale_gu, packed_d, scale_d = _make_banks(e, h, inter)
    st = {}
    y = fused_moe_mxint(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d,
                        weight_repr="hot8", stage_times=st)
    keys = {"layout_sec", "act_quant_sec", "decode_int_sec", "gu_sec",
            "swiglu_sec", "down_quant_sec", "down_sec", "reduction_sec", "total_sec"}
    assert keys == set(st.keys()), set(st.keys()) ^ keys
    assert all(isinstance(v, float) and v >= 0.0 for v in st.values())
    assert st["total_sec"] >= 0.0
    assert y.shape == (k, h)
