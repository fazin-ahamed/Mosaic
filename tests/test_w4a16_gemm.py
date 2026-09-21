from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.w4a16_gemm import dequant_w4, pack_w4, w4a16_gemm, w4a16_grouped_moe


@pytest.mark.parametrize("k_in,n_out", [(4096, 2048), (2048, 4096)])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8])
def test_w4a16_gemm_parity(k_in, n_out, m):
    torch.manual_seed(0)
    w = (torch.randn(k_in, n_out, device="cuda") / math.sqrt(k_in)).to(torch.bfloat16)
    x = torch.randn(m, k_in, dtype=torch.bfloat16, device="cuda")
    packed, scale = pack_w4(w)
    out = w4a16_gemm(x, packed, scale)
    w_hat = dequant_w4(packed, scale)
    ref = x @ w_hat
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 0.15, f"max abs err {err} >= 0.15 (kernel vs dequant W)"
    ref_orig = x @ w
    orig_err = (out.float() - ref_orig.float()).abs().max().item()
    assert orig_err < 1.0, f"quantization vs dense W blew up: {orig_err}"


def test_w4a16_grouped_moe_unique_loop():
    torch.manual_seed(1)
    m, h, inter, e = 8, 256, 128, 4
    x = torch.randn(m, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([0, 0, 1, 2, 2, 2, 3, 1], device="cuda")
    scale = 1.0 / math.sqrt(h)
    wg = (torch.randn(e, h, inter, device="cuda") * scale).to(torch.bfloat16)
    wu = (torch.randn(e, h, inter, device="cuda") * scale).to(torch.bfloat16)
    wd = (torch.randn(e, inter, h, device="cuda") / math.sqrt(inter)).to(torch.bfloat16)
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
    y = w4a16_grouped_moe(x, packed_gate, packed_up, packed_down, scales, ids)
    ref = torch.zeros_like(x)
    for i in range(m):
        ei = ids[i].item()
        g = torch.nn.functional.silu(x[i:i + 1] @ dequant_w4(*pg[ei]))
        u = x[i:i + 1] @ dequant_w4(*pu[ei])
        ref[i] = ((g * u) @ dequant_w4(*pd[ei])).squeeze(0)
    err = (y.float() - ref.float()).abs().max().item()
    assert err < 0.15, f"grouped max abs err {err}"
