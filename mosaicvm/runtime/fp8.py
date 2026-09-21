"""FP8/FP4 block-scaled dequant (checkpoint quantization_config)."""
from __future__ import annotations

import torch


def unpack_fp4_last_dim(packed: torch.Tensor) -> torch.Tensor:
    """Two fp4 values per uint8 along the last dimension."""
    p = packed.to(torch.int32)
    lo = p & 0x0F
    hi = p >> 4
    return torch.stack((lo, hi), dim=-1).reshape(*p.shape[:-1], p.shape[-1] * 2).float()


_E2M1_LUT = (0., .5, 1., 1.5, 2., 3., 4., 6.,
              -0., -.5, -1., -1.5, -2., -3., -4., -6.)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor,
                rows: int, cols: int) -> torch.Tensor:
    """Packed FP4 E2M1 [rows, cols/2] uint8 + E8M0 scale [rows, cols/32]
    -> bf16 [rows, cols].

    Nibble order: low nibble first (repo convention, matches
    unpack_fp4_last_dim and dspark_wire.dequant_fp4). Scale applies per
    32 elements along the input dim AFTER LUT unpacking.
    """
    lut = torch.tensor(_E2M1_LUT, device=packed.device)
    p = packed.view(torch.uint8).to(torch.int64).reshape(rows, -1)
    lo = lut[p & 0x0F]
    hi = lut[p >> 4]
    vals = torch.stack((lo, hi), dim=-1).reshape(rows, cols)
    s = scale.float().reshape(rows, -1).repeat_interleave(32, dim=1)
    return (vals * s).to(torch.bfloat16)


def maybe_unpack_fp4(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if t.numel() == rows * cols:
        return t.reshape(rows, cols)
    if t.numel() * 2 == rows * cols:
        t2 = unpack_fp4_last_dim(t.reshape(t.shape[0], -1))
        if t2.numel() != rows * cols:
            return t
        if t2.shape[0] == rows and t2.shape[1] == cols:
            return t2
        if t2.shape[0] == cols and t2.shape[1] == rows:
            return t2.T
        return t2.reshape(rows, cols)
    return t


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """weight with per-block scale -> bf16.

    Uses the scale tensor's grid when it tiles the weight.
    """
    w = weight.reshape(weight.shape[0], -1).float()
    n, k = w.shape
    s = scale.float()
    if s.ndim == 2 and s.shape[0] > 0 and s.shape[1] > 0 and n % s.shape[0] == 0 and k % s.shape[1] == 0:
        bn, bk = n // s.shape[0], k // s.shape[1]
        w = w.view(s.shape[0], bn, s.shape[1], bk) * s[:, None, :, None]
        return w.reshape(n, k).to(torch.bfloat16)
    return (w * s.reshape(-1)[0]).to(torch.bfloat16)


def fit_linear_weight(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Coerce a packed/transposed expert weight into [rows, cols] bf16."""
    t = maybe_unpack_fp4(t, rows, cols)
    if t.shape == (rows, cols):
        return t.to(torch.bfloat16)
    if t.shape == (cols, rows):
        return t.T.contiguous().to(torch.bfloat16)
    if t.numel() == rows * cols:
        return t.reshape(rows, cols).to(torch.bfloat16)
    raise RuntimeError(f"cannot fit weight shape {tuple(t.shape)} into ({rows}, {cols})")
