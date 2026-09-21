from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.mxfp4 import dequant_mxfp4, pack_mxfp4
from mosaicvm.kernels.skinny_fp4 import fused_moe_skinny, last_launch_count, skinny_mxfp4_gemm


@pytest.mark.parametrize("m,k,n", [(1, 4096, 4096), (2, 4096, 4096), (3, 2048, 4096)])
def test_skinny_mxfp4_gemm_single_expert_parity(m, k, n):
    torch.manual_seed(0)
    w = (torch.randn(k, n, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    packed, scale = pack_mxfp4(w)
    out = skinny_mxfp4_gemm(x, packed, scale, n)
    ref = x @ dequant_mxfp4(packed, scale)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.05, f"max abs err {err} >= 0.05 (skinny vs dequant ref)"


def test_fused_moe_skinny_parity_vs_sequential():
    torch.manual_seed(1)
    k, top_k, e, h, inter = 8, 6, 32, 4096, 2048
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Pack experts to MXFP4
    scale_h = 1.0 / math.sqrt(h)
    scale_i = 1.0 / math.sqrt(inter)
    wg = (torch.randn(e, h, inter, device="cuda") * scale_h).to(torch.bfloat16)
    wu = (torch.randn(e, h, inter, device="cuda") * scale_h).to(torch.bfloat16)
    wd = (torch.randn(e, inter, h, device="cuda") * scale_i).to(torch.bfloat16)

    pgu, sgu = [], []
    for i in range(e):
        p, s = pack_mxfp4(torch.cat([wg[i], wu[i]], dim=1))
        pgu.append(p)
        sgu.append(s)
    packed_gate_up = torch.stack(pgu, dim=0)
    scale_gate_up = torch.stack(sgu, dim=0)

    pdu, sdu = [], []
    for i in range(e):
        p, s = pack_mxfp4(wd[i])
        pdu.append(p)
        sdu.append(s)
    packed_down = torch.stack(pdu, dim=0)
    scale_down = torch.stack(sdu, dim=0)

    out = fused_moe_skinny(
        x, ids, weights, packed_gate_up, scale_gate_up, packed_down, scale_down,
    )

    # Reference: sequential unique-expert loop calling skinny_mxfp4_gemm per expert
    token_of = torch.arange(k, device="cuda").repeat_interleave(top_k)
    flat_ids = ids.reshape(-1)
    x_exp = x[token_of]
    ref = torch.zeros_like(x_exp)
    for e_idx in flat_ids.unique():
        mask = flat_ids == e_idx
        x_e = x_exp[mask]
        gu = skinny_mxfp4_gemm(x_e, packed_gate_up[e_idx], scale_gate_up[e_idx], 2 * inter)
        z = torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:]
        y_e = skinny_mxfp4_gemm(z, packed_down[e_idx], scale_down[e_idx], h)
        ref[mask] = y_e
    ref = ref * weights.reshape(-1, 1).to(ref.dtype)
    y_ref = torch.zeros(k, h, dtype=x.dtype, device="cuda")
    y_ref.index_add_(0, token_of, ref)

    err = (out.float() - y_ref.float()).abs().max().item()
    assert err < 0.05, f"fused vs sequential max abs err {err}"

    out2 = fused_moe_skinny(
        x, ids, weights, packed_gate_up, scale_gate_up, packed_down, scale_down,
    )
    assert last_launch_count() <= 4, f"launches {last_launch_count()} > 4"


def test_exact_moe_requires_exact_target():
    import os
    from mosaicvm.kernels.exact_moe import exact_moe
    from mosaicvm.kernels.fp4_source import pack_mxfp4_nk

    torch.manual_seed(2)
    k, top_k, e, h, inter = 4, 6, 8, 256, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    pgu = torch.stack([pack_mxfp4_nk(torch.randn(2 * inter, h, dtype=torch.bfloat16, device="cuda"))[0] for _ in range(e)])
    sgu = torch.stack([pack_mxfp4_nk(torch.randn(2 * inter, h, dtype=torch.bfloat16, device="cuda"))[1] for _ in range(e)])
    pdu = torch.stack([pack_mxfp4_nk(torch.randn(h, inter, dtype=torch.bfloat16, device="cuda"))[0] for _ in range(e)])
    sdu = torch.stack([pack_mxfp4_nk(torch.randn(h, inter, dtype=torch.bfloat16, device="cuda"))[1] for _ in range(e)])

    os.environ["MOSAICVM_EXACT_TARGET"] = "0"
    try:
        try:
            exact_moe(x, ids, weights, pgu, sgu, pdu, sdu)
            assert False, "exact_moe should raise when exact_target is off"
        except RuntimeError:
            pass
    finally:
        os.environ["MOSAICVM_EXACT_TARGET"] = "1"

    y = exact_moe(x, ids, weights, pgu, sgu, pdu, sdu)
    assert y.shape == (k, h)

