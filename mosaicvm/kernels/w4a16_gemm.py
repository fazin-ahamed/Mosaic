"""Fused W4A16 GEMM: never materializes a [K, N] bf16 weight.

pack_w4 stores two consecutive N-values per uint8 (low nibble first) and a
per-group fp16 scale (GROUP=128, max-abs / 7). The Triton kernel dequantizes
tiles in registers and accumulates with tl.dot in fp32.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

GROUP = 128
BLOCK_M = 16
BLOCK_N = 64
BLOCK_K = 128


def pack_w4(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w_bf16 [K, N] -> packed uint8 [K, N/2] (low nibble first) and scale fp16 [K, N/GROUP].

    Scale is max-abs per group / 7. Quantized values live in [-8, 7].
    """
    if w_bf16.dim() != 2:
        raise ValueError(f"pack_w4 expects [K, N], got {tuple(w_bf16.shape)}")
    k, n = w_bf16.shape
    if n % GROUP != 0:
        raise ValueError(f"N={n} must be a multiple of GROUP={GROUP}")
    w = w_bf16.float()
    groups = n // GROUP
    w_g = w.view(k, groups, GROUP)
    amax = w_g.abs().amax(dim=-1).clamp_min(1e-8)  # [K, groups]
    scale = (amax / 7.0).to(torch.float16)
    q = torch.round(w_g / scale.float().unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    q = q.view(k, n)
    even = q[:, 0::2].to(torch.uint8) & 0x0F
    odd = q[:, 1::2].to(torch.uint8) & 0x0F
    packed = (even | (odd << 4)).contiguous()
    return packed, scale.contiguous()


def pack_gate_up(w_gate: torch.Tensor, w_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate gate and up along N, then pack_w4. N_out = 2 * MOE_INTER."""
    if w_gate.shape != w_up.shape:
        raise ValueError(f"gate {tuple(w_gate.shape)} != up {tuple(w_up.shape)}")
    return pack_w4(torch.cat([w_gate, w_up], dim=1))


def dequant_w4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """packed [K, N/2] uint8 + scale [K, N/GROUP] fp16 -> [K, N] bf16."""
    k, n_half = packed.shape
    n = n_half * 2
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    q = torch.stack((lo, hi), dim=-1).reshape(k, n).float()
    s = scale.float().repeat_interleave(GROUP, dim=-1)
    return (q * s).to(torch.bfloat16)


@triton.jit
def _w4a16_kernel(
    X, PACKED, SCALE, OUT,
    M, N, K,
    stride_xm, stride_xk,
    stride_pk, stride_pn,
    stride_sk, stride_sg,
    stride_om, stride_on,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        mask_x = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        a = tl.load(
            X + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk,
            mask=mask_x,
            other=0.0,
        ).to(tl.float16)

        packed_n = offs_n // 2
        mask_p = (k_idx[:, None] < K) & (offs_n[None, :] < N)
        p = tl.load(
            PACKED + k_idx[:, None] * stride_pk + packed_n[None, :] * stride_pn,
            mask=mask_p,
            other=0,
        )
        nibble = tl.where((offs_n[None, :] % 2) == 0, p & 0x0F, (p >> 4) & 0x0F)
        q = nibble.to(tl.int8)
        q = tl.where(q >= 8, q - 16, q).to(tl.float32)

        group_idx = offs_n // GROUP
        mask_s = (k_idx[:, None] < K) & (offs_n[None, :] < N)
        s = tl.load(
            SCALE + k_idx[:, None] * stride_sk + group_idx[None, :] * stride_sg,
            mask=mask_s,
            other=1.0,
        ).to(tl.float32)
        b = (q * s).to(tl.float16)

        acc += tl.dot(a, b)

    mask_o = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=mask_o,
    )


def w4a16_gemm(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x [M, K] bf16 @ W[K, N] (W4 packed along N) -> [M, N] bf16.

    Never materializes a [K, N] bf16 weight. M is padded to BLOCK_M by the
    launcher (K=1..10 fits in one BLOCK_M). N and K must be multiples of 128.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [M, K], got {tuple(x.shape)}")
    m, k = x.shape
    k_p, n_half = packed.shape
    n = n_half * 2
    if k_p != k:
        raise ValueError(f"packed K={k_p} != x K={k}")
    if k % 128 != 0 or n % 128 != 0:
        raise ValueError(f"K={k} and N={n} must be multiples of 128")
    if scale.shape != (k, n // GROUP):
        raise ValueError(f"scale shape {tuple(scale.shape)} != ({k}, {n // GROUP})")

    m_pad = triton.cdiv(m, BLOCK_M) * BLOCK_M
    if m_pad != m:
        x_in = torch.zeros(m_pad, k, dtype=x.dtype, device=x.device)
        x_in[:m] = x
    else:
        x_in = x.contiguous()

    out = torch.empty(m_pad, n, dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(m_pad, BLOCK_M), triton.cdiv(n, BLOCK_N))
    _w4a16_kernel[grid](
        x_in, packed, scale, out,
        m_pad, n, k,
        x_in.stride(0), x_in.stride(1),
        packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1),
        GROUP=GROUP, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out[:m]


def w4a16_grouped_moe_sequential(x, packed_gate, packed_up, packed_down, scales, expert_ids):
    """x [M, H], expert_ids [M] -> y [M, H]. Unique-expert loop, one w4a16_gemm per unique e.

    packed_gate/up: list or stacked [E, H, I/2]; packed_down: [E, I, H/2].
    scales: dict with keys 'gate','up','down' each [E, K, N/GROUP], or a tuple of three.
    Floor B backend A only; production path is w4a16_grouped_moe -> fused_moe_w4a16.
    """
    m, h = x.shape
    y = torch.zeros(m, h, dtype=x.dtype, device=x.device)
    unique = expert_ids.unique()
    pg, pu, pd = packed_gate, packed_up, packed_down
    if isinstance(scales, dict):
        sg, su, sd = scales["gate"], scales["up"], scales["down"]
    else:
        sg, su, sd = scales
    for e in unique.tolist():
        mask = expert_ids == e
        xe = x[mask]
        g = torch.nn.functional.silu(w4a16_gemm(xe, pg[e], sg[e]))
        u = w4a16_gemm(xe, pu[e], su[e])
        y[mask] = w4a16_gemm(g * u, pd[e], sd[e])
    return y


def w4a16_grouped_moe(x, packed_gate, packed_up, packed_down, scales, expert_ids):
    """Production grouped MoE: fused GPU path (2-4 launches)."""
    from mosaicvm.kernels.w4a16_grouped import fused_moe_w4a16

    return fused_moe_w4a16(
        x, expert_ids, None, packed_gate, packed_up, packed_down, scales
    )
