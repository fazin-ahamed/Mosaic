from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.kernels.fp4_source import decode_e2m1_nk, decode_e4m3, pack_mxfp4_nk
from mosaicvm.kernels.mxint import (
    qint_encode_a,
    qint_encode_w,
    split_qA_device,
    _split_host,
)


def test_qint_encode_w_equals_two_e2m1():
    torch.manual_seed(0)
    w = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
    p, _ = pack_mxfp4_nk(w)
    qw = qint_encode_w(p)
    want = (2.0 * decode_e2m1_nk(p)).round()
    assert torch.equal(qw.float(), want)


def test_qint_encode_a_reconstructs_512A():
    torch.manual_seed(1)
    vals = [0.0, 1e-5, -1e-5, 1e-4, 448.0, -448.0, 1.0, -1.0]
    vals += [2.0 ** e for e in (-8, -1, 0, 2, 4)]
    vals += [-(2.0 ** e) for e in (-8, 0, 2)]
    x = torch.tensor(vals, dtype=torch.float32, device="cuda").view(1, -1)
    # pad to multiple of 16 for e4m3 roundtrip via act-like cast
    pad = (16 - x.shape[1] % 16) % 16
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    x_fp8 = x.to(torch.float8_e4m3fn)
    a0, a1, a2 = qint_encode_a(x_fp8)
    recon = a0.to(torch.int32) + 256 * a1.to(torch.int32) + 65536 * a2.to(torch.int32)
    want = (512.0 * decode_e4m3(x_fp8)).round().to(torch.int32)
    assert torch.equal(recon, want)

    # randn activations
    xr = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    a0, a1, a2 = qint_encode_a(xr)
    recon = a0.to(torch.int32) + 256 * a1.to(torch.int32) + 65536 * a2.to(torch.int32)
    want = (512.0 * decode_e4m3(xr)).round().to(torch.int32)
    assert torch.equal(recon, want)


def test_split_qA_s8_host_and_device():
    qs = torch.tensor(
        [-229376, -1, 0, 1, 229376, 512, -512],
        dtype=torch.int32, device="cuda",
    )
    a0h, a1h, a2h = _split_host(qs)
    recon_h = a0h.to(torch.int32) + 256 * a1h.to(torch.int32) + 65536 * a2h.to(torch.int32)
    assert torch.equal(recon_h, qs)

    a0d, a1d, a2d = split_qA_device(qs)
    recon_d = a0d.to(torch.int32) + 256 * a1d.to(torch.int32) + 65536 * a2d.to(torch.int32)
    assert torch.equal(recon_d, qs)
    assert torch.equal(a0h, a0d) and torch.equal(a1h, a1d) and torch.equal(a2h, a2d)

    # Reverting the carry (plain two's-complement byte slice) must fail -229376.
    q = torch.tensor([-229376], dtype=torch.int32, device="cuda")
    naive0 = ((q & 0xFF) - 256 * ((q & 0xFF) >= 128).to(torch.int32)).to(torch.int8)
    naive1 = (((q >> 8) & 0xFF) - 256 * (((q >> 8) & 0xFF) >= 128).to(torch.int32)).to(torch.int8)
    naive2 = (((q >> 16) & 0xFF) - 256 * (((q >> 16) & 0xFF) >= 128).to(torch.int32)).to(torch.int8)
    naive = naive0.to(torch.int32) + 256 * naive1.to(torch.int32) + 65536 * naive2.to(torch.int32)
    assert not torch.equal(naive, q)
    assert int(naive.item()) == -294912
