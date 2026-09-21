"""Skinny SIMT MXFP4 GEMM: M=1-8, no tl.dot, no BLOCK_M=16 padding.

Ampere ``tl.dot`` requires M >= 16; the Exact-30 MoE path has M=1-8 tokens per
expert.  This kernel does the matvec with pure SIMT FMA in registers — a Python
``for mi in range(MAX_M)`` unrolled at compile time, accumulating
``acc[mi] += sum_k(x[mi, k] * w[k, n])`` without tensor cores.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mosaicvm.kernels.mxfp4 import E2M1_LUT, MX_BLOCK

BLOCK_N = 64
BLOCK_K = 128  # four MX blocks per iter
_LAUNCHES = 0


def last_launch_count() -> int:
    return _LAUNCHES


def _pick_max_m(m: int) -> int:
    # Triton tl.arange requires a power-of-2 length; always pad to 8 and mask
    # the inactive rows. Specializing per-M saves nothing for a SIMT kernel.
    return 8


def _scale_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Convert float8_e8m0fnu (or uint8 bitcast) scale to fp32 power-of-two."""
    if scale.dtype == torch.float8_e8m0fnu:
        s_biased = scale.view(torch.uint8).to(torch.int32)
    elif scale.dtype == torch.uint8:
        s_biased = scale.to(torch.int32)
    else:
        return scale.to(torch.float32)  # already fp32/fp16
    s_exp = s_biased - 127
    s_exp = s_exp.clamp(-127, 128)
    s_bits = (s_exp + 127) << 23
    return s_bits.view(torch.float32)


def _lut_tensor(device) -> torch.Tensor:
    return torch.tensor(E2M1_LUT, dtype=torch.float32, device=device)


@triton.jit
def _skinny_m_kernel(
    X, PACKED, SCALE, LUT, OUT,
    M, N, K,
    stride_xm, stride_xk,
    stride_pk, stride_pn,
    stride_sk, stride_sg,
    stride_om, stride_on,
    MX_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        mask_k = k_idx < K

        # Load packed weights [BLOCK_K, BLOCK_N]
        packed_n = offs_n // 2
        mask_p = mask_k[:, None] & mask_n[None, :]
        p = tl.load(
            PACKED + k_idx[:, None] * stride_pk + packed_n[None, :] * stride_pn,
            mask=mask_p, other=0,
        )

        # Nibble unpack: low nibble = even index, high nibble = odd index
        nibble = tl.where((offs_n[None, :] % 2) == 0, p & 0x0F, (p >> 4) & 0x0F)
        code = nibble.to(tl.int32)
        w_vals = tl.load(LUT + code)

        # Load e8m0 scale (pre-converted to fp32 on host)
        scale_idx = offs_n // MX_BLOCK
        mask_s = mask_k[:, None] & mask_n[None, :]
        s = tl.load(
            SCALE + k_idx[:, None] * stride_sk + scale_idx[None, :] * stride_sg,
            mask=mask_s, other=1.0,
        )

        w = w_vals * s  # [BLOCK_K, BLOCK_N] fp32

        # Load one activation row [BLOCK_K]
        mask_x = k_idx < K
        x_mi = tl.load(
            X + pid_m * stride_xm + k_idx * stride_xk,
            mask=mask_x, other=0.0,
        ).to(tl.float32)

        # acc[n] += sum_k x[k] * w[k, n] — dot-free reduction.
        acc += tl.sum(x_mi[:, None] * w, axis=0)

    tl.store(
        OUT + pid_m * stride_om + offs_n * stride_on,
        acc.to(tl.bfloat16),
        mask=mask_n,
    )


def skinny_mxfp4_gemm(
    x: torch.Tensor,          # [M, K] bf16, M in 1..8
    packed: torch.Tensor,     # [K, N/2] uint8
    scale: torch.Tensor,      # [K, N/32] float8_e8m0fnu or uint8 bitcast
    n_out: int,
) -> torch.Tensor:            # [M, N] bf16
    """Single-expert skinny MXFP4 GEMM. No tl.dot, no BLOCK_M=16 padding."""
    global _LAUNCHES
    m, k = x.shape
    k_p, n_half = packed.shape
    n = n_half * 2
    if n != n_out:
        raise ValueError(f"n_out={n_out} != packed N={n}")
    if k_p != k:
        raise ValueError(f"packed K={k_p} != x K={k}")

    scale_fp32 = _scale_to_fp32(scale)
    lut = _lut_tensor(x.device)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)

    _LAUNCHES += 1
    grid = (m, triton.cdiv(n, BLOCK_N))
    _skinny_m_kernel[grid](
        x, packed, scale_fp32, lut, out,
        m, n, k,
        x.stride(0), x.stride(1),
        packed.stride(0), packed.stride(1),
        scale_fp32.stride(0), scale_fp32.stride(1),
        out.stride(0), out.stride(1),
        MX_BLOCK=MX_BLOCK, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


# ---------------------------------------------------------------------------
# Grouped variable-M dispatcher (2-3 launches, zero Python expert loops)
# ---------------------------------------------------------------------------

@triton.jit
def _skinny_grouped_kernel(
    X, PACKED, SCALE, LUT, OUT,
    OFFSETS, EXPERT_INDEX, M_START, N_TILES,
    N, K,
    stride_xm, stride_xk,
    stride_pe, stride_pk, stride_pn,
    stride_se, stride_sk, stride_sg,
    stride_om, stride_on,
    MX_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_M: tl.constexpr,
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
    m = tl.minimum(MAX_M, row1 - row0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N

    # Per-row: each row has its own 1D accumulator, stored immediately.
    for mi in range(MAX_M):
        if mi < m:
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

            for k0 in range(0, K, BLOCK_K):
                k_idx = k0 + offs_k
                mask_k = k_idx < K

                packed_n = offs_n // 2
                mask_p = mask_k[:, None] & mask_n[None, :]
                p = tl.load(
                    PACKED + e * stride_pe + k_idx[:, None] * stride_pk + packed_n[None, :] * stride_pn,
                    mask=mask_p, other=0,
                )

                nibble = tl.where((offs_n[None, :] % 2) == 0, p & 0x0F, (p >> 4) & 0x0F)
                code = nibble.to(tl.int32)
                w_vals = tl.load(LUT + code)

                scale_idx = offs_n // MX_BLOCK
                mask_s = mask_k[:, None] & mask_n[None, :]
                s = tl.load(
                    SCALE + e * stride_se + k_idx[:, None] * stride_sk + scale_idx[None, :] * stride_sg,
                    mask=mask_s, other=1.0,
                )

                w = w_vals * s

                mask_x = k_idx < K
                x_mi = tl.load(
                    X + (row0 + mi) * stride_xm + k_idx * stride_xk,
                    mask=mask_x, other=0.0,
                ).to(tl.float32)

                acc += tl.sum(x_mi[:, None] * w, axis=0)

            tl.store(
                OUT + (row0 + mi) * stride_om + offs_n * stride_on,
                acc.to(tl.bfloat16),
                mask=mask_n,
            )


def _stack(t):
    if torch.is_tensor(t):
        return t if t.dim() == 3 else t.unsqueeze(0)
    return torch.stack(list(t), dim=0)


def pack_gate_up_mxfp4(w_gate: torch.Tensor, w_up: torch.Tensor):
    """Concatenate gate|up along N then pack_mxfp4. 2048 % 32 == 0."""
    from mosaicvm.kernels.mxfp4 import pack_mxfp4
    return pack_mxfp4(torch.cat([w_gate, w_up], dim=1))


def _grouped_skinny_gemm(
    x_g, packed_w, scale_w, offsets, expert_index, m_start, n_tiles, n_out,
):
    """One Triton launch covering all experts via a tile list."""
    global _LAUNCHES
    packed_w = _stack(packed_w)
    scale_w = _stack(scale_w)
    scale_fp32 = _scale_to_fp32(scale_w)
    a, k = x_g.shape
    n = n_out
    t_cap = expert_index.numel()
    out = torch.zeros(a, n, dtype=torch.bfloat16, device=x_g.device)
    if t_cap == 0:
        return out
    _LAUNCHES += 1
    lut = _lut_tensor(x_g.device)
    grid = (t_cap, triton.cdiv(n, BLOCK_N))
    _skinny_grouped_kernel[grid](
        x_g, packed_w, scale_fp32, lut, out,
        offsets, expert_index, m_start, n_tiles,
        n, k,
        x_g.stride(0), x_g.stride(1),
        packed_w.stride(0), packed_w.stride(1), packed_w.stride(2),
        scale_fp32.stride(0), scale_fp32.stride(1), scale_fp32.stride(2),
        out.stride(0), out.stride(1),
        MX_BLOCK=MX_BLOCK, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, MAX_M=8,
    )
    return out


def fused_moe_skinny(
    x, expert_ids, expert_weights,
    packed_gate_up, scale_gate_up,   # [E, K, N/2] and [E, K, N/32], N=4096 concat
    packed_down, scale_down,         # [E, 2048, 2048] packed along N=4096
) -> torch.Tensor:
    """2-3 launches, zero Python expert loops, no BLOCK_M=16 padding."""
    global _LAUNCHES
    _LAUNCHES = 0
    from mosaicvm.kernels.moe_layout import layout_assignments
    from mosaicvm.kernels.w4a16_grouped import emit_tiles

    h = x.shape[1]
    if expert_ids.ndim == 2:
        k = x.shape[0]
        top_k = expert_ids.shape[1]
        flat_ids = expert_ids.reshape(-1)
        token_of = torch.arange(k, device=x.device).repeat_interleave(top_k)
        if expert_weights is None:
            weights = torch.ones(flat_ids.numel(), device=x.device, dtype=torch.float32)
        else:
            weights = expert_weights.reshape(-1).to(dtype=torch.float32)
        out_rows = k
    else:
        k = x.shape[0]
        flat_ids = expert_ids.reshape(-1)
        token_of = torch.arange(k, device=x.device)
        if expert_weights is None:
            weights = torch.ones(flat_ids.numel(), device=x.device, dtype=torch.float32)
        else:
            weights = expert_weights.reshape(-1).to(dtype=torch.float32)
        out_rows = k

    n_experts = packed_gate_up.shape[0] if torch.is_tensor(packed_gate_up) else len(packed_gate_up)
    _sorted, perm, offsets, counts = layout_assignments(flat_ids, n_experts)
    del _sorted
    x_g = x[token_of][perm]

    expert_index, m_start, n_tiles, _tile_rows = emit_tiles(
        counts, block_m=8, max_tiles=max(n_experts, x_g.shape[0]),
    )

    n_gu = packed_gate_up.shape[-1] * 2 if torch.is_tensor(packed_gate_up) else packed_gate_up[0].shape[-1] * 2
    gu = _grouped_skinny_gemm(
        x_g, packed_gate_up, scale_gate_up, offsets, expert_index, m_start, n_tiles, n_gu,
    )
    half = n_gu // 2
    z = torch.nn.functional.silu(gu[:, :half]) * gu[:, half:]

    y_g = _grouped_skinny_gemm(
        z, packed_down, scale_down, offsets, expert_index, m_start, n_tiles, h,
    )

    y_g = y_g * weights[perm].to(y_g.dtype)[:, None]
    y = torch.zeros(out_rows, h, dtype=x.dtype, device=x.device)
    y.index_add_(0, token_of[perm], y_g)
    return y
