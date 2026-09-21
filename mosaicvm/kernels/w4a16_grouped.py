"""Grouped W4A16 GEMM: one launch covering all experts via a GPU tile list."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mosaicvm.kernels.moe_layout import layout_assignments
from mosaicvm.kernels.w4a16_gemm import BLOCK_K, BLOCK_N, GROUP

BLOCK_M = 16
_LAUNCHES = 0


def last_launch_count() -> int:
    return _LAUNCHES


def _stack(t):
    if torch.is_tensor(t):
        return t if t.dim() == 3 else t.unsqueeze(0)
    return torch.stack(list(t), dim=0)


def _n_out_from_packed(packed) -> int:
    if torch.is_tensor(packed):
        return packed.shape[-1] * 2
    return packed[0].shape[-1] * 2


def _n_experts(*banks) -> int:
    for src in banks:
        if src is None:
            continue
        if torch.is_tensor(src):
            return src.shape[0]
        return len(src)
    raise ValueError("no expert bank")


@triton.jit
def _emit_tiles_kernel(
    COUNTS,
    TILE_OFFSETS,
    EXPERT_INDEX,
    M_START,
    TILE_ROWS,
    BLOCK_M: tl.constexpr,
    MAX_TILES: tl.constexpr,
):
    e = tl.program_id(0)
    c = tl.load(COUNTS + e)
    n_tiles = (c + BLOCK_M - 1) // BLOCK_M
    off = tl.load(TILE_OFFSETS + e)
    for t in range(MAX_TILES):
        do_store = t < n_tiles
        rows = tl.minimum(BLOCK_M, c - t * BLOCK_M)
        tl.store(EXPERT_INDEX + off + t, e, mask=do_store)
        tl.store(M_START + off + t, t * BLOCK_M, mask=do_store)
        tl.store(TILE_ROWS + off + t, rows, mask=do_store)


@triton.jit
def _w4a16_grouped_kernel(
    X,
    PACKED,
    SCALE,
    OUT,
    OFFSETS,
    EXPERT_INDEX,
    M_START,
    N_TILES,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_pe,
    stride_pk,
    stride_pn,
    stride_se,
    stride_sk,
    stride_sg,
    stride_om,
    stride_on,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_tiles = tl.load(N_TILES)
    if pid_t >= n_tiles:
        return

    e = tl.load(EXPERT_INDEX + pid_t)
    m0 = tl.load(M_START + pid_t)
    row0 = tl.load(OFFSETS + e) + m0
    row1 = tl.load(OFFSETS + e + 1)

    offs_m = row0 + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        mask_x = (offs_m[:, None] < row1) & (k_idx[None, :] < K)
        a = tl.load(
            X + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk,
            mask=mask_x,
            other=0.0,
        ).to(tl.float16)

        packed_n = offs_n // 2
        mask_p = (k_idx[:, None] < K) & (offs_n[None, :] < N)
        p = tl.load(
            PACKED
            + e * stride_pe
            + k_idx[:, None] * stride_pk
            + packed_n[None, :] * stride_pn,
            mask=mask_p,
            other=0,
        )
        nibble = tl.where((offs_n[None, :] % 2) == 0, p & 0x0F, (p >> 4) & 0x0F)
        q = nibble.to(tl.int8)
        q = tl.where(q >= 8, q - 16, q).to(tl.float32)

        group_idx = offs_n // GROUP
        mask_s = (k_idx[:, None] < K) & (offs_n[None, :] < N)
        s = tl.load(
            SCALE
            + e * stride_se
            + k_idx[:, None] * stride_sk
            + group_idx[None, :] * stride_sg,
            mask=mask_s,
            other=1.0,
        ).to(tl.float32)
        b = (q * s).to(tl.float16)
        acc += tl.dot(a, b)

    mask_o = (offs_m[:, None] < row1) & (offs_n[None, :] < N)
    tl.store(
        OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=mask_o,
    )


def emit_tiles(counts: torch.Tensor, block_m: int = BLOCK_M, max_tiles: int | None = None):
    """counts [E] int32 -> (expert_index [Tcap], m_start [Tcap], n_tiles [1], tile_rows [Tcap]) on GPU.

    tile_rows[t] = min(block_m, counts[expert_index[t]] - m_start[t]).
    """
    global _LAUNCHES
    e = counts.numel()
    n_tiles_e = (counts.to(torch.int32) + block_m - 1) // block_m
    zeros = torch.zeros(1, dtype=torch.int32, device=counts.device)
    tile_offsets = torch.cat([zeros, torch.cumsum(n_tiles_e, dim=0).to(torch.int32)])
    t_cap = int(max_tiles) if max_tiles is not None else max(e, 1)
    expert_index = torch.empty(t_cap, dtype=torch.int32, device=counts.device)
    m_start = torch.empty(t_cap, dtype=torch.int32, device=counts.device)
    tile_rows = torch.empty(t_cap, dtype=torch.int32, device=counts.device)
    _LAUNCHES += 1
    _emit_tiles_kernel[(e,)](
        counts, tile_offsets, expert_index, m_start, tile_rows, BLOCK_M=block_m, MAX_TILES=32,
    )
    return expert_index, m_start, tile_offsets[-1:], tile_rows


def w4a16_grouped_gemm(
    x_packed: torch.Tensor,
    packed_w: torch.Tensor,
    scale_w: torch.Tensor,
    offsets: torch.Tensor,
    n_out: int,
    expert_index: torch.Tensor | None = None,
    m_start: torch.Tensor | None = None,
    n_tiles: torch.Tensor | None = None,
    block_m: int = BLOCK_M,
) -> torch.Tensor:
    """One launch covering all experts. Empty experts (count=0) are skipped by the grid."""
    global _LAUNCHES
    packed_w = _stack(packed_w)
    scale_w = _stack(scale_w)
    a, k = x_packed.shape
    if k % 128 != 0 or n_out % 128 != 0:
        raise ValueError(f"K={k} and N={n_out} must be multiples of 128")
    counts = (offsets[1:] - offsets[:-1]).to(torch.int32)
    if expert_index is None or m_start is None or n_tiles is None:
        expert_index, m_start, n_tiles, _tile_rows = emit_tiles(counts, block_m)
    t_cap = expert_index.numel()
    out = torch.zeros(a, n_out, dtype=torch.bfloat16, device=x_packed.device)
    if t_cap == 0:
        return out
    _LAUNCHES += 1
    grid = (t_cap, triton.cdiv(n_out, BLOCK_N))
    _w4a16_grouped_kernel[grid](
        x_packed,
        packed_w,
        scale_w,
        out,
        offsets,
        expert_index,
        m_start,
        n_tiles,
        n_out,
        k,
        x_packed.stride(0),
        x_packed.stride(1),
        packed_w.stride(0),
        packed_w.stride(1),
        packed_w.stride(2),
        scale_w.stride(0),
        scale_w.stride(1),
        scale_w.stride(2),
        out.stride(0),
        out.stride(1),
        GROUP=GROUP,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


def stack_gate_up(pool):
    """Concatenate already-packed gate|up along N. GROUP=128 tiles do not cross 2048."""
    pg, sg, pu, su, pd, sd = pool
    n_e = len(pg)
    pgu = torch.stack([torch.cat([pg[e], pu[e]], dim=-1) for e in range(n_e)], dim=0)
    sgu = torch.stack([torch.cat([sg[e], su[e]], dim=-1) for e in range(n_e)], dim=0)
    return pgu, sgu


def fused_moe_w4a16(
    x,
    expert_ids,
    expert_weights,
    packed_gate,
    packed_up,
    packed_down,
    scales,
    packed_gate_up=None,
):
    """2-4 launches, zero Python expert loops.

    x [k, H] with expert_ids [k, top_k] -> y [k, H]
    x [M, H] with expert_ids [M] (already expanded) -> y [M, H]
    """
    global _LAUNCHES
    _LAUNCHES = 0
    h = x.shape[1]
    if expert_ids.ndim == 1:
        k = x.shape[0]
        flat_ids = expert_ids.reshape(-1)
        token_of = torch.arange(k, device=x.device)
        if expert_weights is None:
            weights = torch.ones(k, device=x.device, dtype=torch.float32)
        else:
            weights = expert_weights.reshape(-1).to(dtype=torch.float32)
        out_rows = k
    else:
        k = x.shape[0]
        top_k = expert_ids.shape[1]
        flat_ids = expert_ids.reshape(-1)
        token_of = torch.arange(k, device=x.device).repeat_interleave(top_k)
        if expert_weights is None:
            weights = torch.ones(flat_ids.numel(), device=x.device, dtype=torch.float32)
        else:
            weights = expert_weights.reshape(-1).to(dtype=torch.float32)
        out_rows = k

    n_experts = _n_experts(packed_gate_up, packed_down, packed_gate)
    _sorted, perm, offsets, counts = layout_assignments(flat_ids, n_experts)
    del _sorted
    x_g = x[token_of][perm]
    expert_index, m_start, n_tiles, _tile_rows = emit_tiles(
        counts, BLOCK_M, max_tiles=max(n_experts, x_g.shape[0]),
    )

    if packed_gate_up is not None:
        sgu = scales["gate_up"] if isinstance(scales, dict) else scales
        n_gu = _n_out_from_packed(packed_gate_up)
        gu = w4a16_grouped_gemm(
            x_g, packed_gate_up, sgu, offsets, n_gu,
            expert_index=expert_index, m_start=m_start, n_tiles=n_tiles,
        )
        half = n_gu // 2
        z = torch.nn.functional.silu(gu[:, :half]) * gu[:, half:]
        sd = scales["down"] if isinstance(scales, dict) else None
    else:
        if isinstance(scales, dict):
            sg, su, sd = scales["gate"], scales["up"], scales["down"]
        else:
            sg, su, sd = scales
        n_inter = _n_out_from_packed(packed_gate)
        g = torch.nn.functional.silu(
            w4a16_grouped_gemm(
                x_g, packed_gate, sg, offsets, n_inter,
                expert_index=expert_index, m_start=m_start, n_tiles=n_tiles,
            )
        )
        u = w4a16_grouped_gemm(
            x_g, packed_up, su, offsets, n_inter,
            expert_index=expert_index, m_start=m_start, n_tiles=n_tiles,
        )
        z = g * u

    y_g = w4a16_grouped_gemm(
        z, packed_down, sd, offsets, h,
        expert_index=expert_index, m_start=m_start, n_tiles=n_tiles,
    )
    y_g = y_g * weights[perm].to(y_g.dtype)[:, None]
    y = torch.zeros(out_rows, h, dtype=x.dtype, device=x.device)
    y.index_add_(0, token_of[perm], y_g)
    return y
