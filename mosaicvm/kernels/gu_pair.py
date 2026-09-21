"""Source-preserving gate/up output-row pairing.

Permute output rows: g0..15, u0..15, g16..31, u16..31, ...
Same permutation on scale rows. No pack_mxfp4. No dequant.
"""
from __future__ import annotations

import torch

from mosaicvm.kernels.fp4_source import reference_fp4_linear_source_semantics

PAIR = 16


def _perm(n: int, device) -> torch.Tensor:
    """n must be even; n/2 output rows per matrix, n total after concat."""
    if n % (2 * PAIR) != 0:
        raise ValueError(f"N={n} must be a multiple of {2 * PAIR}")
    half = n // 2
    n_groups = half // PAIR
    idx = torch.arange(n, device=device, dtype=torch.int64)
    g = torch.arange(n_groups, device=device)
    # paired block b: rows [b*32, b*32+16) = gate[b*16:b*16+16]
    #                  rows [b*32+16, b*32+32) = up[b*16:b*16+16]
    gate_pos = (torch.arange(half, device=device) // PAIR) * (2 * PAIR) + (
        torch.arange(half, device=device) % PAIR
    )
    up_pos = gate_pos + PAIR
    perm = torch.empty(n, dtype=torch.int64, device=device)
    perm[gate_pos] = torch.arange(half, device=device)
    perm[up_pos] = torch.arange(half, device=device) + half
    return perm


def pair_gate_up(packed_g, scale_g, packed_u, scale_u):
    """Permute output rows: g0..15, u0..15, g16..31, u16..31, ...

    packed_* [N, K/2], scale_* [N, K/32] with N=2048 typically.
    Returns packed [2N, K/2], scale [2N, K/32] in paired row order.
    """
    n = packed_g.shape[0]
    packed = torch.cat([packed_g, packed_u], dim=0)
    scale = torch.cat([scale_g.view(torch.uint8), scale_u.view(torch.uint8)], dim=0)
    perm = _perm(2 * n, packed.device)
    return packed[perm].contiguous(), scale[perm].contiguous()


def unpair_gate_up(packed, scale):
    """Bit-exact inverse of pair_gate_up. packed [2N,K/2] -> (g, sg, u, su)."""
    n2 = packed.shape[0]
    n = n2 // 2
    perm = _perm(n2, packed.device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n2, device=packed.device)
    p = packed[inv]
    s = scale.view(torch.uint8)[inv]
    return p[:n].contiguous(), s[:n].contiguous(), p[n:].contiguous(), s[n:].contiguous()


def concat_gate_up(packed_g, scale_g, packed_u, scale_u):
    """Byte-concat already packed tensors along N. No requant."""
    return (
        torch.cat([packed_g, packed_u], dim=0).contiguous(),
        torch.cat([scale_g.view(torch.uint8), scale_u.view(torch.uint8)], dim=0).contiguous(),
    )


def swiglu_ref(x, packed_g, scale_g, packed_u, scale_u, swiglu_limit: float = 0.0):
    g = reference_fp4_linear_source_semantics(x, packed_g, scale_g).float()
    u = reference_fp4_linear_source_semantics(x, packed_u, scale_u).float()
    if swiglu_limit > 0:
        g = g.clamp(max=swiglu_limit)
        u = u.clamp(-swiglu_limit, swiglu_limit)
    return torch.nn.functional.silu(g) * u
