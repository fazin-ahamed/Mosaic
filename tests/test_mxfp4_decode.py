from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.fp4_source import decode_e2m1_nk, e8m0_to_fp32
from mosaicvm.kernels.mxfp4_decode import (
    BITWISE_OK,
    decode_e4m3_device,
    exhaustive_e2m1_table,
    reference_exhaustive_nk,
    verify_exhaustive,
)


def test_exhaustive_e2m1_and_e4m3():
    r = verify_exhaustive()
    assert r["e2m1_lut_reg_ok"]
    assert r["e4m3_ok"]
    assert r["n_e2m1"] == 4096
    assert r["n_e4m3"] == 256
    ref = reference_exhaustive_nk()
    lut = exhaustive_e2m1_table("lut_reg")
    finite = torch.isfinite(ref)
    assert torch.allclose(lut[finite], ref[finite], atol=1e-6)
    if not BITWISE_OK:
        pytest.skip("bitwise decoder discarded (not all 4096 pairs match)")
    bit = exhaustive_e2m1_table("bitwise")
    assert torch.allclose(bit[finite], ref[finite], atol=1e-6)


def test_e4m3_all_256_vs_torch():
    b = torch.arange(256, dtype=torch.uint8, device="cuda")
    want = b.view(torch.float8_e4m3fn).float()
    got = decode_e4m3_device(b)
    nan_w = torch.isnan(want)
    assert torch.equal(got[~nan_w], want[~nan_w]) or torch.allclose(got[~nan_w], want[~nan_w], atol=1e-6)
    if nan_w.any():
        assert torch.isnan(got[nan_w]).all()


def test_direct_packed_not_via_pack_mxfp4():
    """Construct packed directly: all code x e8m0, compare to NK dequant of that layout."""
    device = "cuda"
    codes = torch.arange(16, device=device, dtype=torch.uint8)
    e8 = torch.arange(256, device=device, dtype=torch.uint8)
    code_g = codes.view(16, 1).expand(16, 256).reshape(-1)
    e8_g = e8.view(1, 256).expand(16, 256).reshape(-1)
    packed = (code_g | (code_g << 4)).view(-1, 1).expand(-1, 16).contiguous()
    scale = e8_g.view(-1, 1)
    vals = decode_e2m1_nk(packed) * e8m0_to_fp32(scale.view(torch.float8_e8m0fnu)).repeat_interleave(32, dim=-1)
    lut = torch.tensor(
        __import__("mosaicvm.kernels.mxfp4", fromlist=["E2M1_LUT"]).E2M1_LUT,
        dtype=torch.float32, device=device,
    )
    sw = e8m0_to_fp32(e8.view(torch.float8_e8m0fnu))
    want = (lut.view(16, 1) * sw.view(1, 256)).reshape(-1, 1).expand(-1, 32)
    finite = torch.isfinite(want)
    assert torch.allclose(vals[finite], want[finite], atol=1e-6)
