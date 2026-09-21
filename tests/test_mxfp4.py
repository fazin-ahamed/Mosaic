from __future__ import annotations

import pytest
import torch

from mosaicvm.kernels.mxfp4 import E2M1_LUT, MX_BLOCK, dequant_mxfp4, pack_mxfp4

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def test_pack_mxfp4_shapes_and_dtypes():
    torch.manual_seed(0)
    k, n = 256, 256
    w = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4(w)
    assert p.dtype == torch.uint8, f"packed dtype {p.dtype}"
    assert p.shape == (k, n // 2), f"packed shape {tuple(p.shape)}"
    assert s.shape == (k, n // MX_BLOCK), f"scale shape {tuple(s.shape)}"


def test_pack_mxfp4_quantization_error_finite():
    torch.manual_seed(0)
    k, n = 256, 256
    w = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4(w)
    dq = dequant_mxfp4(p, s)
    err = (dq.float() - w.float()).abs()
    assert torch.isfinite(err).all(), "non-finite quantization error"
    # Quantization error, not a kernel error: bounded by the E2M1 grid.
    assert err.max().item() < 1.0, f"max abs err {err.max().item()} too large"


def test_pack_mxfp4_roundtrip_codes():
    """pack -> dequant -> pack preserves the original packed codes."""
    torch.manual_seed(0)
    k, n = 256, 256
    w = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
    p0, s0 = pack_mxfp4(w)
    dq = dequant_mxfp4(p0, s0)
    p1, s1 = pack_mxfp4(dq.to(torch.bfloat16))
    assert torch.equal(p0, p1), "round-trip codes differ"
    assert torch.equal(s0.view(torch.uint8), s1.view(torch.uint8)), "round-trip scales differ"


def test_e2m1_lut_matches_locked_table():
    assert E2M1_LUT == (
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    )


def test_dequant_mxfp4_matches_lut():
    """A zero-scale block dequantizes to the E2M1 codebook values directly."""
    k, n = 1, 32
    codes = torch.arange(16, dtype=torch.uint8, device="cuda").repeat(2)[:n]
    # Pack: even index = low nibble, odd index = high nibble.
    packed = (codes[0::2] | (codes[1::2] << 4)).view(k, n // 2)
    scale = torch.tensor([[127]], dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)  # 2**0 = 1x
    dq = dequant_mxfp4(packed, scale).float().reshape(-1)
    want = torch.tensor(E2M1_LUT, dtype=torch.float32, device="cuda")[:n].repeat(2)[:n]
    assert torch.allclose(dq, want, atol=1e-6), f"dq {dq} != want {want}"
