from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.fp4_source import ACT_BLOCK, act_quant_ref, pack_mxfp4_nk
from mosaicvm.kernels.mxint import (
    fused_moe_mxint,
    mxint_probe_e4m3,
    mxint_probe_silu,
    mxint_routed_reduce,
    mxint_swiglu_quant_down,
    prepack_expert_bank_mxint,
)
from mosaicvm.bench.floors import HIDDEN as _H  # noqa: F401  (constant sanity only)


def test_e4m3_encoder_bitwise_clamped_domain():
    """Device e4m3fn encoder must match torch cast bitwise on [-448, 448]."""
    torch.manual_seed(0)
    vals = []
    # every representable e4m3 value and its +-1ulp fp32 neighbors
    all_bytes = torch.arange(256, dtype=torch.uint8, device="cuda")
    dec = all_bytes.view(torch.float8_e4m3fn).float()
    finite = dec[torch.isfinite(dec)]
    for d in (finite, -finite):
        vals.append(d)
        vals.append(torch.nextafter(d, torch.full_like(d, 1e30)))
        vals.append(torch.nextafter(d, torch.full_like(d, -1e30)))
    # midpoints between consecutive representable values +- 1ulp
    pos = torch.sort(finite)[0]
    mids = (pos[:-1] + pos[1:]) / 2
    for d in (mids, -mids):
        vals.append(d)
        vals.append(torch.nextafter(d, torch.full_like(d, 1e30)))
        vals.append(torch.nextafter(d, torch.full_like(d, -1e30)))
    # dense random sweep across the domain (incl. subnormals)
    r = torch.empty(1 << 21, dtype=torch.float32, device="cuda").uniform_(-448.0, 448.0)
    lg = torch.randn(1 << 20, dtype=torch.float32, device="cuda") * 30.0
    tiny = torch.randn(1 << 20, dtype=torch.float32, device="cuda") * 1e-2
    vals += [r, lg, tiny,
             torch.zeros(4, device="cuda"),
             torch.tensor([448.0, -448.0, 0.0, -0.0], device="cuda")]
    x = torch.cat([v.reshape(-1) for v in vals]).float().contiguous()
    x = x[torch.isfinite(x) & (x.abs() <= 448.0)]
    got = mxint_probe_e4m3(x)
    want = x.to(torch.float8_e4m3fn).view(torch.uint8)
    bad = (got != want)
    assert not bad.any(), (
        f"e4m3 encoder mismatches torch on {int(bad.sum())}/{x.numel()} values; "
        f"first: x={x[bad][:8].tolist()} got={got[bad][:8].tolist()} "
        f"want={want[bad][:8].tolist()}")


def test_silu_device_bitwise():
    torch.manual_seed(1)
    x = torch.cat([
        torch.randn(1 << 22, dtype=torch.float32, device="cuda") * 20.0,
        torch.randn(1 << 20, dtype=torch.float32, device="cuda") * 1.0,
        torch.linspace(-30.0, 30.0, 1 << 16, device="cuda"),
        torch.tensor([0.0, -0.0, 1e-20, -1e-20], device="cuda"),
    ]).contiguous()
    got = mxint_probe_silu(x)
    want = torch.nn.functional.silu(x)
    bad = got.view(torch.int32) != want.view(torch.int32)
    assert not bad.any(), (
        f"silu mismatches on {int(bad.sum())}/{x.numel()}; "
        f"x={x[bad][:8].tolist()} got={got[bad][:8].tolist()} want={want[bad][:8].tolist()}")


def _torch_swiglu_quant(gu, w_a, inter, limit):
    gate = gu[:, :inter].float()
    up = gu[:, inter:].float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    z = torch.nn.functional.silu(gate) * up
    z = z * w_a[:, None]
    zd = z.to(torch.bfloat16)
    xf8, sA = act_quant_ref(zd, ACT_BLOCK)
    return zd, xf8, sA


def test_swiglu_quant_down_bitwise():
    torch.manual_seed(2)
    A, inter = 24, 256
    gu = (torch.randn(A, 2 * inter, device="cuda") * 3.0).to(torch.bfloat16)
    w_a = torch.rand(A, device="cuda", dtype=torch.float32) * 1.5
    limit = 10.0
    zd_ref, xf8_ref, sA_ref = _torch_swiglu_quant(gu, w_a, inter, limit)
    a0, a1, a2, sA = mxint_swiglu_quant_down(gu, w_a, inter, limit)
    assert torch.equal(sA, sA_ref), (sA - sA_ref).abs().max().item()
    # planes must equal decode_activation_int of the reference fp8 bytes
    from mosaicvm.kernels.mxint import decode_activation_int
    r0, r1, r2 = decode_activation_int((xf8_ref, sA_ref))
    assert torch.equal(a0, r0)
    assert torch.equal(a1, r1)
    assert torch.equal(a2, r2)


def test_swiglu_quant_down_clamp_engaged():
    torch.manual_seed(3)
    A, inter = 8, 128
    gu = (torch.randn(A, 2 * inter, device="cuda") * 30.0).to(torch.bfloat16)
    w_a = torch.rand(A, device="cuda", dtype=torch.float32) * 2.0
    zd_ref, xf8_ref, sA_ref = _torch_swiglu_quant(gu, w_a, inter, 10.0)
    a0, a1, a2, sA = mxint_swiglu_quant_down(gu, w_a, inter, 10.0)
    from mosaicvm.kernels.mxint import decode_activation_int
    r0, r1, r2 = decode_activation_int((xf8_ref, sA_ref))
    assert torch.equal(sA, sA_ref)
    assert torch.equal(a0, r0) and torch.equal(a1, r1) and torch.equal(a2, r2)


def test_routed_reduce_bitwise():
    torch.manual_seed(4)
    k_tok, top_k, h, e = 5, 6, 320, 16
    a = k_tok * top_k
    ids = torch.stack([torch.randperm(e)[:top_k] for _ in range(k_tok)]).to("cuda")
    weights = torch.rand(k_tok, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(-1, keepdim=True)
    yd = (torch.randn(a, h, device="cuda") * 2.0).to(torch.bfloat16)
    # reference path (copy of fused_moe_mxint reduction)
    from mosaicvm.kernels.moe_layout import layout_assignments
    flat_ids = ids.reshape(-1)
    token_of = torch.arange(k_tok, device="cuda").repeat_interleave(top_k)
    slot_of = torch.arange(top_k, device="cuda").repeat(k_tok)
    _, perm, offsets, counts = layout_assignments(flat_ids, e)
    partials = torch.zeros(k_tok, top_k, h, dtype=torch.float32, device="cuda")
    partials[token_of[perm], slot_of[perm]] = yd.float()
    order = ids.long().argsort(dim=1)
    ref = torch.zeros(k_tok, h, dtype=torch.float32, device="cuda")
    for s in range(top_k):
        slot = order[:, s]
        ref = ref + partials.gather(1, slot.view(k_tok, 1, 1).expand(k_tok, 1, h)).squeeze(1)
    ref = ref.to(torch.bfloat16)
    # kernel path
    row_of_flat = torch.empty(a, dtype=torch.int64, device="cuda")
    row_of_flat[perm] = torch.arange(a, device="cuda")
    slot_rows = row_of_flat.view(k_tok, top_k)
    got = mxint_routed_reduce(yd, slot_rows, order, h)
    assert torch.equal(got, ref), (got.float() - ref.float()).abs().max().item()


def _make_bank(e, h, inter):
    pgu, sgu, pdu, sdu = [], [], [], []
    for _ in range(e):
        pg, sg = pack_mxfp4_nk((torch.randn(2 * inter, h, device="cuda") / math.sqrt(h)).to(torch.bfloat16))
        pd, sd = pack_mxfp4_nk((torch.randn(h, inter, device="cuda") / math.sqrt(inter)).to(torch.bfloat16))
        pgu.append(pg); sgu.append(sg); pdu.append(pd); sdu.append(sd)
    return (torch.stack(pgu), torch.stack(sgu), torch.stack(pdu), torch.stack(sdu))


@pytest.mark.parametrize("k,top_k", [(4, 2), (1, 6), (8, 6)])
def test_fused_epilogue_bitwise_vs_torch(k, top_k):
    torch.manual_seed(5)
    e, h, inter = 8, 256, 128
    if top_k > e:
        e = top_k
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.stack([torch.randperm(e)[:top_k] for _ in range(k)]).to("cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(-1, keepdim=True)
    pgu, sgu, pdu, sdu = _make_bank(e, h, inter)
    for repr_ in ("hot8", "packed4"):
        y_t = fused_moe_mxint(x, ids, weights, pgu, sgu, pdu, sdu,
                              weight_repr=repr_, epilogue="torch")
        y_f = fused_moe_mxint(x, ids, weights, pgu, sgu, pdu, sdu,
                              weight_repr=repr_, epilogue="fused")
        assert torch.equal(y_t, y_f), (
            repr_, k, top_k, (y_t.float() - y_f.float()).abs().max().item())


def test_fused_epilogue_two_run_and_no_item():
    torch.manual_seed(6)
    k, top_k, e, h, inter = 4, 2, 8, 256, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(-1, keepdim=True)
    pgu, sgu, pdu, sdu = _make_bank(e, h, inter)
    bank = prepack_expert_bank_mxint(pgu, sgu, packed_d=pdu, scale_d=sdu)
    y1 = fused_moe_mxint(x, ids, weights, bank=bank, epilogue="fused")
    y2 = fused_moe_mxint(x, ids, weights, bank=bank, epilogue="fused")
    assert torch.equal(y1, y2)
    n = {"n": 0}
    orig = torch.Tensor.item
    def counted(self):
        n["n"] += 1
        return orig(self)
    torch.Tensor.item = counted
    try:
        fused_moe_mxint(x, ids, weights, bank=bank, epilogue="fused")
    finally:
        torch.Tensor.item = orig
    assert n["n"] == 0, n["n"]


def test_fused_epilogue_stage_times_keys():
    torch.manual_seed(7)
    k, top_k, e, h, inter = 4, 2, 8, 256, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(-1, keepdim=True)
    pgu, sgu, pdu, sdu = _make_bank(e, h, inter)
    st = {}
    fused_moe_mxint(x, ids, weights, pgu, sgu, pdu, sdu,
                    weight_repr="hot8", epilogue="fused", stage_times=st)
    keys = {"layout_sec", "act_quant_sec", "decode_int_sec", "gu_sec",
            "swiglu_sec", "down_quant_sec", "down_sec", "reduction_sec", "total_sec"}
    assert keys == set(st.keys())
