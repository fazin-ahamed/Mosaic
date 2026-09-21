"""Triton grouped GEMM over selected experts — one launch per MoE layer.

Phase 1 correctness fallback uses torch bmm (grouped_expert_forward in
layers.py); this module holds the faster Triton path for Phase 2+.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemm_kernel(
    X, W, OUT, EXPERTS,
    stride_xm, stride_xk,
    stride_we, stride_wk, stride_wn,
    stride_om, stride_on,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    exp = tl.load(EXPERTS + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(X + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk)
        mask_e = exp[:, None] * 0 + tl.arange(0, 1)[None, :]
        b = tl.load(
            W + exp[:, None, None] * stride_we + (k0 + offs_k)[None, :, None] * stride_wk + offs_n[None, None, :] * stride_wn
        )
        acc += tl.sum(a[:, :, None].to(tl.float32) * b.to(tl.float32), axis=1)

    tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc.to(tl.bfloat16))


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """y[m,:] = x[m,:] @ w[expert_ids[m]]

    x: [M, K]; w: [E, K, N]; expert_ids: [M]
    """
    M, K = x.shape
    _, _, N = w.shape
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _grouped_gemm_kernel[grid](
        x, w, out, expert_ids,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1),
        N=N, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out
