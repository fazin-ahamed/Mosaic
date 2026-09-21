"""Exact V4 DSA primitives — ported from the checkpoint's own reference
implementation (.mosaicvm/source/inference_model.py + inference_kernel.py).

These are the exactness source of truth for production kernels. Everything
matches the source op-for-op:

  * Attention: per-head q RMS-norm, rope on the LAST 64 dims, 64-block
    FP8-sim on KV non-rope dims, sliding-window-128 gather + top-k compressed
    gather, sink added to the softmax DENOMINATOR only, inverse-rope on the
    output, grouped low-rank O projection.
  * Compressor: fp32 wkv/wgate, ape-positioned softmax pooling over
    compress_ratio chunks (overlap interleaving at ratio 4), RMSNorm, rope,
    64-block FP8-sim on non-rope dims (indexer variant: Hadamard + fp4-sim).
  * Indexer: 64 heads x 128, rope + Hadamard + fp4-sim q, relu scoring
    weighted per head, causal mask, top-512, offset past the window.
  * act_quant_sim: pow2 scale 2^ceil(log2(amax/448)), amax >= 1e-4, e4m3
    round-trip, bf16 storage (exact port of act_quant_kernel inplace).

0xFF scale interpretation is parameterized ('pow2' | 'saturate' | 'nan') —
OPEN_SEMANTICS_CASE (artifacts/ff_semantics.md); default pow2 matches the
hardware tensor-core decode (2^(E-127)) used by every observed runtime.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------- config --
WINDOW = 128
INDEX_TOPK = 512
ROPE_DIM = 64
NORM_EPS = 1e-6
HC_EPS = 1e-6
HC_MULT = 4
SINKHORN_ITERS = 20
ROPE_THETA = 10000
COMPRESS_ROPE_THETA = 160000
ORIG_SEQ = 65536
ROPE_FACTOR = 16
BETA_FAST, BETA_SLOW = 32, 1


# ------------------------------------------------------------- scale/quant -
def _bits_to_f32(exp_int: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, (exp_int.float() - 127.0).clamp(-126, 128))


def dequant_scale(s: torch.Tensor, ff: str = "pow2") -> torch.Tensor:
    """e8m0 scale bytes -> f32. 0xFF: OPEN_SEMANTICS_CASE (AGENTS.md;
    artifacts/ff_semantics.md). Documented decodes:
      'pow2'     -> 2^(E-127); E=255 -> 2^128 = +inf (tcgen05
                    cutlass::float_ue8m0_t and vLLM _upcast_e8m0_to_fp32
                    bit-insertion semantics)
      'saturate' -> 2^127
      'nan'      -> torch e8m0fnu .float() (E=255 -> NaN)
    The production loader (runtime.fp8.dequant_fp8, s.float()) matches the
    P2-measured core (core_timing._load_layer) — NaN decode; parity
    comparisons use mutual-finite masking (AGENTS.md §36-38).
    """
    if s.dtype != torch.float8_e8m0fnu:
        return s.float()
    if ff == "nan":
        return s.float()
    b = s.view(torch.uint8).to(torch.int32)
    if ff == "saturate":
        b = b.clamp(max=254)
    return _bits_to_f32(b)


def dequant_weight(w_fp8: torch.Tensor, s: torch.Tensor,
                   ff: str = "pow2") -> torch.Tensor:
    n, k = w_fp8.shape
    s32 = dequant_scale(s, ff).to(w_fp8.device)
    nb, kb = s32.shape
    return (w_fp8.float().view(nb, n // nb, kb, k // kb) * s32[:, None, :, None]) \
        .reshape(n, k).to(torch.bfloat16)


def act_quant_sim(x: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Exact port of act_quant(inplace=True, round_scale=True): bf16 in/out."""
    orig = x.shape
    xf = x.float().reshape(-1, orig[-1])
    N = xf.shape[-1]
    xb = xf.view(xf.shape[0], N // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * s).to(torch.bfloat16).view(orig)


_E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def fp4_act_quant_sim(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Exact port of fp4_act_quant(inplace=True): pow2 scale (max 6), e2m1 grid."""
    orig = x.shape
    xf = x.float().reshape(-1, orig[-1])
    N = xf.shape[-1]
    xb = xf.view(xf.shape[0], N // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(6 * 2**-126)
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    q = (xb / s).clamp(-6.0, 6.0)
    grid = torch.tensor(_E2M1_GRID, device=x.device)
    mid = (grid[1:] + grid[:-1]) / 2.0
    idx = torch.bucketize(q.abs(), mid)
    return (q.sign() * grid[idx] * s).to(torch.bfloat16).view(orig)


_HAD: dict = {}


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """fast_hadamard_transform, scale d^-0.5 (Sylvester construction)."""
    d = x.shape[-1]
    assert d & (d - 1) == 0
    H = _HAD.get(d)
    if H is None or H.device != x.device:
        h = torch.tensor([[1.0]], device=x.device, dtype=torch.float32)
        while h.shape[0] < d:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)])
        _HAD[d] = H = h / math.sqrt(d)
    return (x.float() @ H).to(x.dtype)


# ------------------------------------------------------------------- rope --
def precompute_freqs_cis(dim: int, seqlen: int, original_seq_len: int,
                         base: float, factor: float, beta_fast: int,
                         beta_slow: int, device) -> torch.Tensor:
    def correction_dim(num_rot, dim, base, max_seq):
        return dim * math.log(max_seq / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    lo = max(int(math.floor(correction_dim(beta_fast, dim, base, max_seq=original_seq_len))), 0) \
        if original_seq_len > 0 else 0
    hi = min(int(math.ceil(correction_dim(beta_slow, dim, base, max_seq=original_seq_len))), dim - 1) \
        if original_seq_len > 0 else 0

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        if lo == hi:
            hi += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - lo) / (hi - lo), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    return torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs)).to(device)


_FREQS_CACHE: dict = {}


def get_freqs_table(dim: int, seqlen: int, original_seq_len: int,
                    base: float, device) -> torch.Tensor:
    """Shared rope table. Tables are value-identical across all layers of
    the same (theta, YaRN) class — sharing is exactness-preserving and
    mandatory at long contexts (105 per-layer tables would be ~28 GB at
    1M; deduped it is 2 tables / ~0.8 GB). Tables are read-only in use.
    Always complex64 (polar of f32)."""
    key = (dim, seqlen, original_seq_len, base, str(device))
    t = _FREQS_CACHE.get(key)
    if t is None:
        t = precompute_freqs_cis(dim, seqlen, original_seq_len, base,
                                 16.0, 32, 1, device)
        _FREQS_CACHE[key] = t
    return t


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor,
                     inverse: bool = False) -> torch.Tensor:
    y = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    fc = freqs_cis.conj() if inverse else freqs_cis
    if x.ndim == 4:
        fc = fc.view(1, fc.size(0), 1, x.size(-1) // 2)
    else:
        # local layouts: [s, h, d] (3-D) or [s, d] (2-D); freqs on dim 0
        if x.ndim == 3:
            fc = fc.view(fc.size(0), 1, x.size(-1) // 2)
        else:
            fc = fc.view(fc.size(0), x.size(-1) // 2)
    out = torch.view_as_real(y * fc).flatten(-2)
    return out.to(x.dtype)

# ---------------------------------------------------------------- indexes --
def window_topk_idxs(seqlen: int, device, start_pos: int = 0) -> torch.Tensor:
    """Causal sliding-window gather indices. Prefill: [s, window].
    Decode (start_pos>0, seqlen=1): [1, window] into the ring buffer."""
    if start_pos >= WINDOW - 1:
        sp = start_pos % WINDOW
        m = torch.cat([torch.arange(sp + 1, WINDOW, device=device),
                       torch.arange(0, sp + 1, device=device)])
        return m.int().view(1, -1)
    if start_pos > 0:
        m = F.pad(torch.arange(start_pos + 1, device=device),
                  (0, WINDOW - start_pos - 1), value=-1)
        return m.int().view(1, -1)
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    m = (base - WINDOW + 1).clamp(0) + torch.arange(min(seqlen, WINDOW), device=device)
    return torch.where(m > base, -1, m).int()


def compress_topk_idxs_static(ratio: int, seqlen: int, device,
                              offset: int, start_pos: int = 0) -> torch.Tensor:
    """Causal static compress gather. Prefill [s, s//ratio]; decode [1, n]."""
    if start_pos > 0:
        n = (start_pos + 1) // ratio
        return (torch.arange(n, device=device) + offset).int().view(1, -1)
    m = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
    mask = m >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
    return torch.where(mask, -1, m + offset).int()


# ------------------------------------------------------------ sparse attn --
def sparse_attn_ref(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor,
                    topk_idxs: torch.Tensor, scale: float) -> torch.Tensor:
    """sparse_attn_kernel exact math: gathered-KV online-softmax with the sink
    as a denominator-only virtual token. q [s,h,d] bf16, kv [n,d] bf16,
    topk_idxs [s,t] int32 (-1 masked), sink [h] fp32."""
    valid = topk_idxs >= 0
    idxc = topk_idxs.clamp(min=0)
    kv_g = kv[idxc]                                            # [s,t,d]
    # single fp32 materialization of the gathered KV (the kernel gemms
    # bf16 into fp32 accumulators; .float() preserves exactly those
    # accumulate semantics) — previously materialized TWICE (once per
    # einsum), doubling the dominant attention transient
    kv_gf = kv_g.float()
    del kv_g
    qf = q.float()
    scores = torch.einsum("shd,std->sht", qf, kv_gf) * scale
    scores = torch.where(valid.unsqueeze(1), scores,
                         torch.full_like(scores, float("-inf")))
    m = scores.max(dim=-1, keepdim=True).values                # [s,h,1]
    p = torch.exp(scores - m)
    # kernel truth (sparse_attn_kernel): the denominator sums the FP32 p
    # (reduce_sum over acc_s), while the VALUE gemm uses the bf16-cast p
    # (acc_s_cast). Denominator therefore uses unrounded p:
    denom = p.sum(-1) + torch.exp(sink.unsqueeze(0) - m.squeeze(-1))
    # value gemm: bf16-cast p (acc_s_cast) against the fp32 KV copy
    p_bf = p.to(q.dtype).float()
    del scores, qf
    o = torch.einsum("sht,std->shd", p_bf, kv_gf) / denom.unsqueeze(-1)
    return o.to(q.dtype)


# ------------------------------------------------------------- HC mixing --
def hc_split_sinkhorn_ref(mixes: torch.Tensor, hc_scale, hc_base):
    """Exact port of hc_split_sinkhorn_kernel. mixes [n, mix_hc] fp32 ->
    (pre [n,hc], post [n,hc], comb [n,hc,hc])."""
    hc = HC_MULT
    eps = HC_EPS
    pre_ = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[:, 2 * hc:].view(-1, hc, hc) * hc_scale[2] + hc_base[2 * hc:].view(hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre_, post, comb


def hc_pre_ref(x4: torch.Tensor, hc_fn, hc_scale, hc_base):
    """x4 [s, hc, d] -> (y [s,d], post, comb). Source Block.hc_pre."""
    shape = x4.shape
    xf = x4.reshape(shape[0], -1).float()
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(xf, hc_fn) * rs
    pre_, post, comb = hc_split_sinkhorn_ref(mixes, hc_scale, hc_base)
    y = (pre_.unsqueeze(-1) * x4.float()).sum(dim=1)
    return y.to(x4.dtype), post, comb


def hc_post_ref(y: torch.Tensor, residual: torch.Tensor, post, comb) -> torch.Tensor:
    """Source Block.hc_post. y [s,d]; residual [s,hc,d]."""
    out = post.unsqueeze(-1) * y.float().unsqueeze(-2) + \
        torch.einsum("sjk,skd->sjd", comb, residual.float())
    return out.to(y.dtype)
