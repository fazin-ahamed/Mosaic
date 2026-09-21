from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_dsa_parity():
    import sys
    sys.path.insert(0, ".")
    from mosaicvm.kernels.dsa_attention import dsa_decode_attention, reference_topk_attention

    torch.manual_seed(0)
    H, d, ctx, topk = 64, 128, 4096, 256
    q = torch.randn(H, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(ctx, H, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(ctx, H, d, dtype=torch.bfloat16, device="cuda")

    out_ref, idx_ref = reference_topk_attention(q, k, v, topk)
    out, idx = dsa_decode_attention(q, k, v, topk)

    assert torch.equal(idx.sort(dim=-1).values, idx_ref.sort(dim=-1).values.long()), "selected indices differ"
    err = (out.float() - out_ref.float()).abs().max().item()
    assert err < 2e-2, f"max abs err {err}"
