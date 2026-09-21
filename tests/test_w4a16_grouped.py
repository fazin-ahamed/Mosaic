from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.w4a16_gemm import pack_gate_up, pack_w4, w4a16_grouped_moe_sequential
from mosaicvm.kernels.w4a16_grouped import fused_moe_w4a16, last_launch_count, stack_gate_up


def _pack_experts(e, h, inter):
    scale_h = 1.0 / math.sqrt(h)
    scale_i = 1.0 / math.sqrt(inter)
    wg = (torch.randn(e, h, inter, device="cuda") * scale_h).to(torch.bfloat16)
    wu = (torch.randn(e, h, inter, device="cuda") * scale_h).to(torch.bfloat16)
    wd = (torch.randn(e, inter, h, device="cuda") * scale_i).to(torch.bfloat16)
    pg = [pack_w4(wg[i]) for i in range(e)]
    pu = [pack_w4(wu[i]) for i in range(e)]
    pd = [pack_w4(wd[i]) for i in range(e)]
    packed_gate = [p for p, _ in pg]
    packed_up = [p for p, _ in pu]
    packed_down = [p for p, _ in pd]
    scales = (
        [s for _, s in pg],
        [s for _, s in pu],
        [s for _, s in pd],
    )
    return packed_gate, packed_up, packed_down, scales


def _weighted_sequential(x, ids, weights, packed_gate, packed_up, packed_down, scales):
    k, top_k = ids.shape
    token_of = torch.arange(k, device=x.device).repeat_interleave(top_k)
    y_exp = w4a16_grouped_moe_sequential(
        x[token_of], packed_gate, packed_up, packed_down, scales, ids.reshape(-1)
    )
    y_exp = y_exp * weights.reshape(-1, 1).to(y_exp.dtype)
    y = torch.zeros_like(x)
    y.index_add_(0, token_of, y_exp)
    return y


def test_fused_moe_w4a16_parity_vs_sequential():
    torch.manual_seed(0)
    k, top_k, e, h, inter = 8, 6, 32, 4096, 2048
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    packed_gate, packed_up, packed_down, scales = _pack_experts(e, h, inter)

    out = fused_moe_w4a16(x, ids, weights, packed_gate, packed_up, packed_down, scales)
    ref = _weighted_sequential(x, ids, weights, packed_gate, packed_up, packed_down, scales)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.15, f"max abs err {err} >= 0.15"
    assert last_launch_count() <= 4


def test_fused_moe_gate_up_concat_parity():
    torch.manual_seed(1)
    k, top_k, e, h, inter = 8, 6, 32, 4096, 2048
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    packed_gate, packed_up, packed_down, scales = _pack_experts(e, h, inter)
    pgu, sgu = stack_gate_up((packed_gate, scales[0], packed_up, scales[1], packed_down, scales[2]))
    out = fused_moe_w4a16(
        x, ids, weights, None, None, packed_down,
        scales={"gate_up": sgu, "down": scales[2]},
        packed_gate_up=pgu,
    )
    ref = _weighted_sequential(x, ids, weights, packed_gate, packed_up, packed_down, scales)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.15, f"concat max abs err {err} >= 0.15"
    assert last_launch_count() <= 3


def test_pack_gate_up_matches_separate_pack():
    torch.manual_seed(2)
    h, inter = 4096, 2048
    wg = torch.randn(h, inter, dtype=torch.bfloat16, device="cuda")
    wu = torch.randn(h, inter, dtype=torch.bfloat16, device="cuda")
    p, s = pack_gate_up(wg, wu)
    pg, sg = pack_w4(wg)
    pu, su = pack_w4(wu)
    assert torch.equal(p, torch.cat([pg, pu], dim=-1))
    assert torch.equal(s, torch.cat([sg, su], dim=-1))
