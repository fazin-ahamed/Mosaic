"""Decode-time top-k sparse attention (DSA) Triton kernel.

Doc §7: never materialize all-64-head intermediate state. Heads are processed
in groups of HEAD_GROUP (8, tunable to 16); top-k selection over the candidate
set runs in chunks of CHUNK with a running top-k kept in registers.
Workspace is only groups-sized (H x topk), never heads x context.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_GROUP = 8
CHUNK = 64
CANDIDATES = 512


@triton.jit
def _dsa_select_kernel(
    Q, K, IDX, SCORES,
    qk_scale,
    stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    n_ctx,
    TOPK: tl.constexpr, HEADS: tl.constexpr, HEAD_GROUP: tl.constexpr,
    INDEX_DIM: tl.constexpr, CHUNK: tl.constexpr, N_CHUNKS: tl.constexpr,
):
    """Stage 1: per-head running top-k. Grid: (heads // HEAD_GROUP, batch).

    Never materializes heads x context scores; workspace is H x topk.
    """
    pid_g = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_d = tl.arange(0, INDEX_DIM)
    slots = tl.arange(0, TOPK)

    for hh in range(HEAD_GROUP):
        h = pid_g * HEAD_GROUP + hh
        q_h = tl.load(Q + h * stride_qh + offs_d * stride_qd)
        best_s = tl.full((TOPK,), float("-inf"), dtype=tl.float32)
        best_i = tl.zeros((TOPK,), dtype=tl.int32)

        for c in range(N_CHUNKS):
            offs_n = c * CHUNK + tl.arange(0, CHUNK)
            mask_n = offs_n < n_ctx
            kk = tl.load(
                K + offs_n[:, None] * stride_kn + h * stride_kh + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None], other=0.0,
            )
            scores = tl.sum(kk.to(tl.float32) * q_h.to(tl.float32)[None, :], axis=1) * qk_scale
            scores = tl.where(mask_n, scores, float("-inf"))
            for _ in range(CHUNK):
                m_chunk = tl.max(scores, axis=0)
                i_chunk = tl.argmax(scores, axis=0)
                take_new = m_chunk > tl.min(best_s, axis=0)
                slot = tl.argmin(best_s, axis=0)
                old_i = tl.sum(tl.where(slots == slot, best_i, 0))
                old_s = tl.sum(tl.where(slots == slot, best_s, 0.0))
                new_s = tl.where(take_new, m_chunk, old_s)
                new_i = tl.where(take_new, c * CHUNK + i_chunk, old_i)
                best_s = tl.where(slots == slot, new_s, best_s)
                best_i = tl.where(slots == slot, new_i, best_i)
                scores = tl.where(tl.arange(0, CHUNK) == i_chunk, float("-inf"), scores)

        out_base = (pid_b * HEADS + h) * TOPK + slots
        tl.store(IDX + out_base, best_i)
        tl.store(SCORES + out_base, best_s)


@triton.jit
def _dsa_attend_kernel(
    Q, K, V, IDX, OUT,
    softmax_scale,
    stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    n_ctx,
    HEADS: tl.constexpr, HEAD_GROUP: tl.constexpr, TOPK: tl.constexpr,
    QDIM: tl.constexpr, VDIM: tl.constexpr,
):
    """Stage 2: gather top-k KV per head and compute attention output."""
    pid_g = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_k = tl.arange(0, TOPK)

    for hh in range(HEAD_GROUP):
        h = pid_g * HEAD_GROUP + hh
        idx_h = tl.load(IDX + (pid_b * HEADS + h) * TOPK + offs_k)  # [TOPK]
        q_h = tl.load(Q + h * stride_qh + tl.arange(0, QDIM) * stride_qd)  # [QDIM]

        scores = tl.full((TOPK,), float("-inf"), dtype=tl.float32)
        for t in range(TOPK):
            pos = tl.sum(tl.where(offs_k == t, idx_h, 0), axis=0)
            kt = tl.load(K + pos * stride_kn + h * stride_kh + tl.arange(0, QDIM) * stride_kd)
            s = tl.sum(q_h.to(tl.float32) * kt.to(tl.float32), axis=0) * softmax_scale
            scores = tl.where(offs_k == t, s, scores)

        m = tl.max(scores, axis=0)
        p = tl.exp(scores - m)
        l = tl.sum(p, axis=0)

        acc = tl.zeros((VDIM,), dtype=tl.float32)
        offs_dv = tl.arange(0, VDIM)
        for t in range(TOPK):
            pos = tl.sum(tl.where(offs_k == t, idx_h, 0), axis=0)
            vt = tl.load(V + pos * stride_vn + h * stride_vh + offs_dv * stride_vd)
            acc += tl.sum(tl.where(offs_k == t, p, 0.0), axis=0) * vt.to(tl.float32)

        tl.store(OUT + (pid_b * HEADS + h) * VDIM + offs_dv, (acc / l).to(tl.bfloat16))


def dsa_decode_attention(q, k_index, v_full, topk=CANDIDATES, scale=None):
    """Public API.

    q: [H, index_dim] bf16 queries for one decoding token (batch=1).
    k_index: [ctx, H, index_dim] bf16 candidate keys.
    v_full: [ctx, H, v_dim] bf16 values.
    Returns (out [H, v_dim], idx [H, topk]).
    """
    assert q.dim() == 2, "batch=1 decode path"
    H, index_dim = q.shape
    ctx = k_index.shape[0]
    v_dim = v_full.shape[-1]
    assert H % HEAD_GROUP == 0
    assert ctx >= topk, "context must cover candidate set"
    n_chunks = triton.cdiv(ctx, CHUNK)
    if scale is None:
        scale = 1.0 / (index_dim ** 0.5)

    idx = torch.empty(1, H, topk, dtype=torch.int32, device=q.device)
    scores_ws = torch.empty(1, H, topk, dtype=torch.float32, device=q.device)

    _dsa_select_kernel[(H // HEAD_GROUP, 1)](
        q, k_index, idx, scores_ws,
        scale,
        q.stride(0), q.stride(1),
        k_index.stride(0), k_index.stride(1), k_index.stride(2),
        ctx,
        TOPK=topk, HEADS=H, HEAD_GROUP=HEAD_GROUP,
        INDEX_DIM=index_dim, CHUNK=CHUNK, N_CHUNKS=n_chunks,
    )

    out = torch.empty(H, v_dim, dtype=torch.bfloat16, device=q.device)
    _dsa_attend_kernel[(H // HEAD_GROUP, 1)](
        q, k_index, v_full, idx, out,
        scale,
        q.stride(0), q.stride(1),
        k_index.stride(0), k_index.stride(1), k_index.stride(2),
        v_full.stride(0), v_full.stride(1), v_full.stride(2),
        ctx,
        HEADS=H, HEAD_GROUP=HEAD_GROUP, TOPK=topk, QDIM=index_dim, VDIM=v_dim,
    )
    return out, idx[0]


def reference_topk_attention(q, k, v, topk, scale=None):
    """Plain PyTorch reference implementing the same top-k sparse attention."""
    H, index_dim = q.shape
    if scale is None:
        scale = 1.0 / (index_dim ** 0.5)
    scores = torch.einsum("hd,chd->ch", q.float(), k.float()) * scale  # [ctx, H]
    top_scores, idx = scores.topk(topk, dim=0)  # [topk, H]
    v_t = v.permute(1, 0, 2).float()  # [H, ctx, vdim]
    gathered = v_t.gather(1, idx.T.unsqueeze(-1).expand(-1, -1, v.shape[-1]))  # [H, topk, vdim]
    probs = torch.softmax(top_scores.T, dim=-1)  # [H, topk]
    out = (probs.unsqueeze(-1) * gathered).sum(dim=1)
    return out.to(v.dtype), idx.T
