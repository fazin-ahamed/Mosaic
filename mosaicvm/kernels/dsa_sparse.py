"""Fused exact-DSA sparse attention — round-2 Triton kernel (AGENTS §42–46).

Replaces the eager `sparse_attn_ref` in PRODUCTION bf16 mode. Semantics
follow the checkpoint's sparse_attn_kernel (tilelang, .mosaicvm/source/
inference_kernel.py:277+) block-for-block:

  * per (seq_pos, head-tile) program; gathered-KV traversal in blocks of
    64 (source `block = 64`), topk indices -1-masked;
  * scores = tl.dot(bf16 q, bf16 k) with FP32 accumulation — exactly the
    source T.gemm on BF16 shared tiles into FP32 fragments;
  * online max/sum across blocks (source scores_max/scores_scale/sum_exp);
  * p is rounded to bf16 before the value dot (source acc_s_cast), the
    denominator sums the FP32 p (source reduce_sum on acc_s);
  * sink contributes exp(sink - running_max) to the DENOMINATOR only;
  * final o = acc_o / sum_exp, cast to q dtype.

Head tiling (BLOCK_H=16) exists to give tl.dot non-degenerate MMA shapes;
per-head math is unchanged. Classification: NUMERIC_PARITY vs the source
kernel (same block-64 traversal and accumulator precision; tensor-core dot
splitting and online-rescale ordering are not bit-identical to the fp32
eager reference). The eager path remains the BITWISE_MOSAIC_REFERENCE;
fp32 parity suites keep asserting against it. Workspace: none beyond the
output — no fp32 KV materialization (the eager path's ~0.25-GB transient).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_fused(
    q_ptr,            # [s, h, d] bf16
    kv_ptr,           # [n, d]   bf16
    idx_ptr,          # [s, t]   int32 (-1 masked)
    sink_ptr,         # [h]      fp32
    o_ptr,            # [s, h, d] bf16
    scale,
    s, t, d,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)          # [BH]
    offs_d = tl.arange(0, D)                                  # [D]
    offs_bt = tl.arange(0, BLOCK_T)                           # [BT]

    # q tile [BH, D] bf16 (heads of this tile)
    q = tl.load(q_ptr + pid_s * H * D + offs_h[:, None] * D
                + offs_d[None, :])                            # [BH, D] bf16

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)       # running max
    l_i = tl.zeros([BLOCK_H], tl.float32)                     # running sum
    acc = tl.zeros([BLOCK_H, D], tl.float32)                  # value acc

    for tb in range(0, t, BLOCK_T):
        idxs = tl.load(idx_ptr + pid_s * t + tb + offs_bt,
                       mask=tb + offs_bt < t, other=-1)
        valid = idxs != -1                                    # [BT]
        safe = tl.where(valid, idxs, 0)
        # gather KV rows as [BT, D] then transpose for the score dot
        k = tl.load(kv_ptr + safe[:, None] * d + offs_d[None, :])  # [BT,D] bf16
        kt = tl.trans(k)                                      # [D, BT] bf16
        # scores: [BH, D] x [D, BT] -> [BH, BT] fp32 accumulate (MMA)
        s_ij = tl.dot(q, kt) * scale
        s_ij = tl.where(valid[None, :], s_ij, -float("inf"))
        # online max / rescale (source scores_max_prev/scores_scale)
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))         # [BH]
        alpha = tl.exp(m_i - m_new)                           # [BH]
        p = tl.exp(s_ij - m_new[:, None])                     # [BH, BT] fp32
        # denominator accumulates FP32 p (source reduce_sum acc_s)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        # value dot uses bf16-cast p (source acc_s_cast): [BH,BT]x[BT,D]
        p_bf = p.to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p_bf, k)          # fp32 acc
        m_i = m_new

    # sink: denominator-only (source sum_exp += exp(sink - scores_max))
    sink = tl.load(sink_ptr + offs_h)                         # [BH] fp32
    l_i = l_i + tl.exp(sink - m_i)

    o = acc / l_i[:, None]                                    # [BH, D] fp32
    tl.store(o_ptr + pid_s * H * D + offs_h[:, None] * D + offs_d[None, :],
             o.to(o_ptr.dtype.element_ty))


def sparse_attn_fused(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor,
                      topk_idxs: torch.Tensor, scale: float) -> torch.Tensor:
    """q [s,h,d] bf16, kv [n,d] bf16, topk_idxs [s,t] int32 (-1 masked),
    sink [h] fp32 -> o [s,h,d] bf16. Production path (bf16 only)."""
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16, \
        "fused path is bf16-only; fp32 parity uses sparse_attn_ref"
    s, h, d = q.shape
    t = topk_idxs.shape[-1]
    assert d in (64, 128, 256, 512), f"unsupported head dim {d}"
    assert h % 16 == 0, f"unsupported head count {h}"
    o = torch.empty_like(q)
    grid = (s, h // 16)
    _sparse_attn_fused[grid](
        q, kv, topk_idxs.to(torch.int32), sink.float(), o,
        scale, s, t, d,
        H=h, D=d, BLOCK_H=16, BLOCK_T=64,     # BLOCK_T = source block size
        num_warps=4, num_stages=2,
    )
    return o
