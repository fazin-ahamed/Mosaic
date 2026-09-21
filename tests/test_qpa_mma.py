from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.fp4_source import pack_mxfp4_nk, reference_fp4_linear_source_semantics
from mosaicvm.kernels.gu_pair import pair_gate_up, swiglu_ref, unpair_gate_up
from mosaicvm.kernels.qpa_mma import fused_moe_qpa, inverse_prepack_qpa, prepack_qpa, qpa_mma_gemm
from mosaicvm.kernels.xsimt import inverse_prepack_xsimt, prepack_xsimt, xsimt_gemm


def test_xsimt_prepack_inverse_bit_exact():
    torch.manual_seed(0)
    n, k = 64, 128
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    codes, scales = prepack_xsimt(p, s)
    p2, s2 = inverse_prepack_xsimt(codes, scales, n, k)
    assert torch.equal(p, p2)
    assert torch.equal(s.view(torch.uint8), s2)


def test_qpa_k32_prepack_inverse_bit_exact():
    torch.manual_seed(0)
    n, k = 64, 128
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    tiles, _ = prepack_qpa(p, s, "qpa_k32_fragment")
    p2, s2 = inverse_prepack_qpa(tiles, n, k)
    assert torch.equal(p, p2)
    assert torch.equal(s.view(torch.uint8), s2)


def test_gu_unpair_bit_exact():
    torch.manual_seed(0)
    n, k = 32, 128
    wg = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    wu = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    pg, sg = pack_mxfp4_nk(wg)
    pu, su = pack_mxfp4_nk(wu)
    packed, scale = pair_gate_up(pg, sg, pu, su)
    g2, sg2, u2, su2 = unpair_gate_up(packed, scale)
    assert torch.equal(pg, g2)
    assert torch.equal(pu, u2)
    assert torch.equal(sg.view(torch.uint8), sg2)
    assert torch.equal(su.view(torch.uint8), su2)


def test_gu_swiglu_vs_reference():
    torch.manual_seed(1)
    m, n, k = 2, 32, 128
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    pg, sg = pack_mxfp4_nk((torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16))
    pu, su = pack_mxfp4_nk((torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16))
    packed, scale = pair_gate_up(pg, sg, pu, su)
    g2, sg2, u2, su2 = unpair_gate_up(packed, scale)
    g = xsimt_gemm(x, g2, sg2.view(torch.float8_e8m0fnu), n)
    u = xsimt_gemm(x, u2, su2.view(torch.float8_e8m0fnu), n)
    z = torch.nn.functional.silu(g.float()) * u.float()
    ref = swiglu_ref(x, pg, sg, pu, su)
    err = (z - ref).abs().max().item()
    assert err < 0.05, f"GU SwiGLU max abs {err}"


def test_fused_moe_qpa_two_run_equal():
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
        pgu.append(pg)
        sgu.append(sg)
        pdu.append(pd)
        sdu.append(sd)
    packed_gu = torch.stack(pgu)
    scale_gu = torch.stack(sgu)
    packed_d = torch.stack(pdu)
    scale_d = torch.stack(sdu)
    y1 = fused_moe_qpa(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d)
    y2 = fused_moe_qpa(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d)
    assert torch.equal(y1, y2)
    assert y1.shape == (k, h)


def test_fused_moe_qpa_no_item_in_dispatch():
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
    from mosaicvm.kernels.qpa_mma import prepack_expert_bank_qpa
    bank = prepack_expert_bank_qpa(packed_gu, scale_gu, packed_d=packed_d, scale_d=scale_d)
    n = {"n": 0}
    orig = torch.Tensor.item
    def counted(self):
        n["n"] += 1
        return orig(self)
    torch.Tensor.item = counted
    try:
        fused_moe_qpa(x, ids, weights, bank=bank)
    finally:
        torch.Tensor.item = orig
    assert n["n"] == 0, n["n"]

