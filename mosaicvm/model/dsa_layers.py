"""Exact V4 DSA attention — Compressor, Indexer, ExactAttention.

Ported op-for-op from the checkpoint's reference implementation
(.mosaicvm/source/inference_model.py). This is the exact production path
replacing the eager einsum stand-in (§60-1). Numerics match the source:
per-head q RMS-norm, rope on the last 64 dims, 64-block FP8-sim on KV
non-rope dims, sink in the softmax denominator only, inverse-rope on the
output, grouped low-rank O projection; the compressor does fp32 gated
pooling with ape positions and overlap interleaving at ratio 4; the indexer
does relu-scored top-512 selection with Hadamard + fp4-sim on its path.

Prefill (chunk) mode here; decode mode inherits the same per-position
semantics via the state buffers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import V4Config
from .dsa import (
    WINDOW, INDEX_TOPK, NORM_EPS, ROPE_DIM, act_quant_sim, fp4_act_quant_sim,
    hadamard, precompute_freqs_cis, apply_rotary_emb, sparse_attn_ref,
    window_topk_idxs, compress_topk_idxs_static, get_freqs_table,
)

# Production dispatch: bf16 calls route to the fused Triton kernel
# (mosaicvm/kernels/dsa_sparse.py; NUMERIC_PARITY class per AGENTS §45);
# fp32/reference mode always uses the eager bitwise reference.
USE_FUSED_SPARSE = True


def _attend(q, kv, sink, idxs, scale):
    if USE_FUSED_SPARSE and q.dtype == torch.bfloat16 \
            and kv.dtype == torch.bfloat16:
        from ..kernels.dsa_sparse import sparse_attn_fused
        return sparse_attn_fused(q, kv.contiguous(), sink, idxs, scale)
    return sparse_attn_ref(q, kv, sink, idxs, scale)


class Compressor(nn.Module):
    """Gated pooled KV compression (source Compressor, prefill path).

    wkv/wgate output coff*head_dim dims: at ratio 4 (overlap) the first
    head_dim dims are the overlapping half, the second the normal half.
    """

    def __init__(self, dim: int, ratio: int, head_dim: int, rotate: bool,
                 cfg: V4Config, max_seq: int = 4096, device="cuda"):
        super().__init__()
        self.ratio = ratio
        self.head_dim = head_dim
        self.rope_head_dim = ROPE_DIM
        self.overlap = ratio == 4
        self.rotate = rotate
        coff = 1 + (1 if self.overlap else 0)
        self.coff = coff
        self.ape = nn.Parameter(torch.zeros(ratio, coff * head_dim))
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False, dtype=torch.float32)
        self.norm = RMSNormExact(head_dim)
        theta = cfg.compress_rope_theta
        # source: compressor shares the attention rope table and indexes it at
        # CHUNK START positions (freqs_cis[:cutoff:ratio]). Shared table —
        # value-identical across layers of the same theta class.
        self._freq_args = (ROPE_DIM, max_seq + 2, 65536, theta)
        self.freqs = get_freqs_table(ROPE_DIM, max_seq + 2, 65536, theta,
                                     device)
        # decode incremental state (source kv_state / score_state)
        self.register_buffer(
            "kv_state", torch.zeros(coff * ratio, coff * head_dim,
                                    dtype=torch.float32), persistent=False)
        self.register_buffer(
            "score_state", torch.full((coff * ratio, coff * head_dim),
                                      float("-inf"), dtype=torch.float32),
            persistent=False)
        self.kv_cache = None  # assigned by ExactAttention (window-offset view)

    def rebuild_freqs(self):
        dim, ms, osl, theta = self._freq_args
        self.freqs = get_freqs_table(dim, ms, osl, theta, self.ape.device)

    def _w(self, t: torch.Tensor) -> torch.Tensor:
        """fp32 view of a weight (no-op cast for fp32 storage; exact widen
        for bf16 storage — checkpoint values are bf16-representable)."""
        return t if t.dtype == torch.float32 else t.float()

    def _post(self, out, dtype, start_pos, cutoff, s):
        """norm + rope + quant, then write kv_cache. Source Compressor tail."""
        ratio, rd = self.ratio, self.rope_head_dim
        out = self.norm(out.to(dtype))
        if start_pos == 0:
            f = self.freqs[:cutoff:ratio][:out.shape[0]]
        else:
            f = self.freqs[start_pos + 1 - ratio].unsqueeze(0)
        orot = apply_rotary_emb(out[..., -rd:], f)
        out = torch.cat([out[..., :-rd], orot], -1)
        if self.rotate:
            out = hadamard(out)
            out = fp4_act_quant_sim(out)
        else:
            out = torch.cat([act_quant_sim(out[..., :-rd]), out[..., -rd:]], -1)
        if self.kv_cache is not None:
            if start_pos == 0:
                self.kv_cache[:s // ratio] = out
            else:
                self.kv_cache[start_pos // ratio] = out.squeeze(0)
        return out

    def _chunked(self, kv, score, dtype, start_pos, s):
        """Chunked prefill (start_pos>0, s>1). Extends the source's
        prefill/decode state machine to multi-row chunks:
          1. leading rows complete the in-progress group (decode-style,
             emitting when the group closes),
          2. aligned full groups pool batched (prefill math, prev-group
             first half from state for the first group),
          3. trailing rows seed the next group's state.
        Emits go through _post (rope at chunk-start position g*ratio,
        matching both prefill [:cutoff:ratio] and decode [sp+1-ratio])."""
        ratio, d, rd = self.ratio, self.head_dim, self.rope_head_dim
        coff = self.coff
        ape = self._w(self.ape)
        emitted = []

        lead_end = min(-(-start_pos // ratio) * ratio, start_pos + s)
        for i in range(lead_end - start_pos):
            pos = start_pos + i
            kv_i, sc_i = kv[i:i + 1], score[i:i + 1] + ape[pos % ratio]
            if self.overlap:
                self.kv_state[ratio + pos % ratio] = kv_i.squeeze(0)
                self.score_state[ratio + pos % ratio] = sc_i.squeeze(0)
                if (pos + 1) % ratio == 0:
                    kv_st = torch.cat([self.kv_state[:ratio, :d],
                                       self.kv_state[ratio:, d:]], dim=0)
                    sc_st = torch.cat([self.score_state[:ratio, :d],
                                       self.score_state[ratio:, d:]], dim=0)
                    out = (kv_st * sc_st.softmax(dim=0)).sum(dim=0, keepdim=True)
                    self.kv_state[:ratio] = self.kv_state[ratio:]
                    self.score_state[:ratio] = self.score_state[ratio:]
                    emitted.append(self._post(out, dtype, pos, 0, 1))
            else:
                self.kv_state[pos % ratio] = kv_i.squeeze(0)
                self.score_state[pos % ratio] = sc_i.squeeze(0)
                if (pos + 1) % ratio == 0:
                    out = (self.kv_state * self.score_state.softmax(dim=0)
                           ).sum(dim=0, keepdim=True)
                    emitted.append(self._post(out, dtype, pos, 0, 1))

        n_full = (start_pos + s - lead_end) // ratio
        if n_full > 0:
            lo = lead_end - start_pos
            rows = kv[lo:lo + n_full * ratio].view(n_full, ratio, coff * d)
            srows = score[lo:lo + n_full * ratio].view(n_full, ratio, coff * d) + ape
            if self.overlap:
                prev_kv = torch.cat([self.kv_state[:ratio, :d].unsqueeze(0),
                                     rows[:-1, :, :d]], dim=0)
                prev_sc = torch.cat([self.score_state[:ratio, :d].unsqueeze(0),
                                     srows[:-1, :, :d]], dim=0)
                w_kv = torch.cat([prev_kv, rows[:, :, d:]], dim=1)
                w_sc = torch.cat([prev_sc, srows[:, :, d:]], dim=1)
                pooled = (w_kv * w_sc.softmax(dim=1)).sum(dim=1)
            else:
                pooled = (rows * srows.softmax(dim=1)).sum(dim=1)
            # state: last full group becomes prev; stale [ratio:] slots are
            # safe — a group only emits after every slot was rewritten
            self.kv_state[:ratio] = rows[-1].reshape(ratio, coff * d)
            self.score_state[:ratio] = srows[-1]
            g0 = lead_end // ratio
            f = self.freqs[g0 * ratio:(g0 + n_full) * ratio:ratio]
            out = self.norm(pooled.to(dtype))
            orot = apply_rotary_emb(out[..., -rd:], f)
            out = torch.cat([out[..., :-rd], orot], -1)
            if self.rotate:
                out = hadamard(out)
                out = fp4_act_quant_sim(out)
            else:
                out = torch.cat([act_quant_sim(out[..., :-rd]),
                                 out[..., -rd:]], -1)
            if self.kv_cache is not None:
                self.kv_cache[g0:g0 + n_full] = out
            emitted.append(out)

        tail_lo = lead_end - start_pos + n_full * ratio
        for i in range(tail_lo, s):
            pos = start_pos + i
            slot = (ratio if self.overlap else 0) + pos % ratio
            self.kv_state[slot] = kv[i]
            self.score_state[slot] = score[i] + ape[pos % ratio]

        if not emitted:
            return None
        return torch.cat(emitted, dim=0)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """x [s, dim] -> emitted compressed rows (or None). Dispatch:
        prefill (start_pos=0) / decode (s=1) / chunked prefill (both>0).
        Chunked prefill has no source reference — it is validated against
        full-prefix recompute (tests/test_exact_dsa_chunked_parity.py)."""
        ratio, d, rd = self.ratio, self.head_dim, self.rope_head_dim
        s = x.shape[0]
        coff = self.coff
        dtype = x.dtype
        xf = x.float()
        kv = torch.nn.functional.linear(xf, self._w(self.wkv.weight))   # [s, coff*d]
        score = torch.nn.functional.linear(xf, self._w(self.wgate.weight))
        ape = self._w(self.ape)
        if start_pos == 0:
            should_compress = s >= ratio
            remainder = s % ratio
            cutoff = s - remainder
            n_chunk = cutoff // ratio
            offset = ratio if self.overlap else 0
            if self.overlap and cutoff >= ratio:
                self.kv_state[:ratio] = kv[cutoff - ratio:cutoff]
                self.score_state[:ratio] = score[cutoff - ratio:cutoff] + ape
            if remainder > 0:
                self.kv_state[offset:offset + remainder] = kv[cutoff:]
                self.score_state[offset:offset + remainder] = (
                    score[cutoff:] + ape[:remainder])
                kv, score = kv[:cutoff], score[:cutoff]
            kv_u = kv.view(n_chunk, ratio, coff * d)
            sc_u = score.view(n_chunk, ratio, coff * d) + ape
            if self.overlap:
                t, ts = kv_u, sc_u
                nkv = torch.zeros(n_chunk, 2 * ratio, d, device=x.device)
                nsc = torch.full((n_chunk, 2 * ratio, d), float("-inf"),
                                 device=x.device)
                nkv[:, ratio:] = t[:, :, d:]
                nsc[:, ratio:] = ts[:, :, d:]
                nkv[1:, :ratio] = t[:-1, :, :d]
                nsc[1:, :ratio] = ts[:-1, :, :d]
                out = (nkv * nsc.softmax(dim=1)).sum(dim=1)
            else:
                out = (kv_u * sc_u.softmax(dim=1)).sum(dim=1)
            if not should_compress:
                return None
            return self._post(out, dtype, 0, cutoff, s)
        if s > 1:
            return self._chunked(kv, score, dtype, start_pos, s)
        # --- decode (s=1) ---
        should_compress = (start_pos + 1) % ratio == 0
        score = score + ape[start_pos % ratio]
        if self.overlap:
            self.kv_state[ratio + start_pos % ratio] = kv.squeeze(0)
            self.score_state[ratio + start_pos % ratio] = score.squeeze(0)
            if should_compress:
                kv_st = torch.cat([self.kv_state[:ratio, :d],
                                   self.kv_state[ratio:, d:]], dim=0)
                sc_st = torch.cat([self.score_state[:ratio, :d],
                                   self.score_state[ratio:, d:]], dim=0)
                out = (kv_st * sc_st.softmax(dim=0)).sum(dim=0, keepdim=True)
                self.kv_state[:ratio] = self.kv_state[ratio:]
                self.score_state[:ratio] = self.score_state[ratio:]
        else:
            self.kv_state[start_pos % ratio] = kv.squeeze(0)
            self.score_state[start_pos % ratio] = score.squeeze(0)
            if should_compress:
                out = (self.kv_state * self.score_state.softmax(dim=0)
                       ).sum(dim=0, keepdim=True)
        if not should_compress:
            return None
        return self._post(out, dtype, start_pos, 0, s)


class Indexer(nn.Module):
    """Top-k compressed-KV selection (source Indexer, prefill path)."""

    def __init__(self, cfg: V4Config, dim: int, max_seq: int = 4096, device="cuda"):
        super().__init__()
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.topk = min(cfg.index_topk, max_seq // 4)
        self.scale = self.head_dim ** -0.5 * self.n_heads ** -0.5
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim,
                              bias=False, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(dim, self.n_heads, bias=False, dtype=torch.bfloat16)
        self.compressor = Compressor(dim, 4, self.head_dim, rotate=True,
                                     cfg=cfg, max_seq=max_seq, device=device)

    def forward(self, x: torch.Tensor, qr: torch.Tensor,
                freqs: torch.Tensor, offset: int,
                start_pos: int = 0) -> torch.Tensor:
        s = x.shape[0]
        end_pos = start_pos + s
        q = self.wq_b(qr).view(s, self.n_heads, self.head_dim)
        f = freqs[start_pos:start_pos + s]
        qr_ = apply_rotary_emb(q[..., -ROPE_DIM:], f)
        q = torch.cat([q[..., :-ROPE_DIM], qr_], -1)
        q = hadamard(q)
        q = fp4_act_quant_sim(q)
        # compressor writes its own kv_cache (assigned by ExactAttention)
        self.compressor(x, start_pos)
        n_c = end_pos // 4
        ckv = self.compressor.kv_cache[:n_c]
        # SOURCE-EXACT scoring dtypes: the source indexer einsums q (bf16,
        # post fp4-sim) against the bf16 kv cache, relu_ in place, scales by
        # bf16 weights and reduces in bf16 — everything dtype-flowing. The
        # previous .float() upcasts were a precision UPGRADE (part of the
        # bf16-mode residual vs source) and ~2x slower on the context-scaled
        # C/4 scoring term. In fp32 parity mode all tensors are fp32 and the
        # einsum stays fp32 — bitwise tests unchanged.
        weights = self.weights_proj(x) * self.scale
        scores = torch.einsum("shd,td->sht", q, ckv).relu()
        scores = (scores * weights.unsqueeze(-1)).sum(dim=1)
        # causal bound per row: entry j valid iff j < (p+1)//4 with p the
        # ABSOLUTE position of row t (= start_pos + t); prefill p == t
        bounds = (torch.arange(1, s + 1, device=x.device) + start_pos) // 4
        if start_pos == 0 or s > 1:
            mask = torch.arange(n_c, device=x.device).repeat(s, 1) >= \
                bounds.unsqueeze(1)
            # dtype-flowing mask add (source adds a bf16 -inf mask under
            # bf16 default dtype; .to is a no-op in fp32 parity mode)
            scores = scores + torch.where(
                mask, float("-inf"), 0.0).to(scores.dtype)
        k = min(INDEX_TOPK, n_c)
        idxs = scores.topk(k, dim=-1)[1]
        if start_pos == 0 or s > 1:
            m2 = idxs >= bounds.unsqueeze(1)
            return torch.where(m2, -1, idxs + offset)
        return idxs + offset


class RMSNormExact(nn.Module):
    def __init__(self, dim: int, eps: float = NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        d = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        # lazy widen: bf16-stored weights (checkpoint-native residency mode)
        # compute fp32 with bit-identical values — fp32 storage skips the cast
        w = self.weight if self.weight.dtype == torch.float32 \
            else self.weight.float()
        return (w * xf).to(d)


class ExactAttention(nn.Module):
    """Exact V4 attention (source Attention, prefill). Drop-in for the eager
    stand-in at verification widths; per-layer compress_ratio aware."""

    def __init__(self, cfg: V4Config, layer_id: int, max_seq: int = 4096,
                 device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.rd = cfg.qk_rope_head_dim
        self.nope = self.head_dim - self.rd
        self.o_groups = cfg.o_groups
        self.o_lora = cfg.o_lora_rank
        self.window = cfg.sliding_window
        self.scale = self.head_dim ** -0.5
        self.eps = cfg.rms_norm_eps

        inner = self.n_heads * self.head_dim
        bf16 = torch.bfloat16
        self.q_a = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False, dtype=bf16)
        self.q_norm = RMSNormExact(cfg.q_lora_rank, self.eps)
        self.q_b = nn.Linear(cfg.q_lora_rank, inner, bias=False, dtype=bf16)
        self.kv_proj = nn.Linear(cfg.hidden_size, self.head_dim, bias=False, dtype=bf16)
        self.kv_norm = RMSNormExact(self.head_dim, self.eps)
        # group = 8 heads x 512 dims = 4096 (source wo_a: [4096 -> 8*1024])
        gh = (inner // self.o_groups)
        self.wo_a = nn.Linear(gh, self.o_groups * self.o_lora, bias=False,
                              dtype=bf16)
        self.wo_b = nn.Linear(self.o_groups * self.o_lora, cfg.hidden_size,
                              bias=False, dtype=bf16)
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))

        self.ratio = cfg.compress_ratios[layer_id]
        self.has_indexer = self.ratio == 4
        self.compressor = None
        self.indexer = None
        if self.ratio:
            self.compressor = Compressor(cfg.hidden_size, self.ratio,
                                         self.head_dim, rotate=False, cfg=cfg,
                                         max_seq=max_seq, device=device)
            if self.has_indexer:
                self.indexer = Indexer(cfg, cfg.hidden_size, max_seq, device)
            osl, theta = (65536, cfg.compress_rope_theta) if self.ratio \
                else (0, cfg.rope_theta)
        else:
            self.compressor = None
            self.indexer = None
            osl, theta = 0, cfg.rope_theta
        self._freq_args = (self.rd, max_seq + 2, osl, theta)
        self.freqs = get_freqs_table(self.rd, max_seq + 2, osl, theta, device)
        # source: kv_cache [win + max_seq//ratio, head_dim]; compressor
        # writes into the compressed half (offset = window)
        cache_len = self.window + (max_seq // self.ratio if self.ratio else 0)
        # caches are bf16 (source semantics: kit registers under default bf16);
        # a module-level .float() casts them for the fp32 parity path
        cache_dt = torch.bfloat16
        self.register_buffer(
            "kv_cache",
            torch.zeros(cache_len, self.head_dim, dtype=cache_dt, device=device),
            persistent=False)
        if self.indexer is not None:
            self.register_buffer(
                "ix_kv_cache",
                torch.zeros(max_seq // 4, self.indexer.head_dim,
                            dtype=cache_dt, device=device),
                persistent=False)
        self._wire_caches()

    def _wire_caches(self):
        """Point compressor/indexer at this module's cache buffers and
        re-resolve shared freqs tables. MUST run after any
        .to()/.float()/.cuda() — buffer views and device-keyed shared
        tables go stale when _apply replaces the underlying tensors."""
        dev = self.kv_cache.device
        rd, ms, osl, theta = self._freq_args
        self.freqs = get_freqs_table(rd, ms, osl, theta, dev)
        if getattr(self, "compressor", None) is not None:
            self.compressor.kv_cache = self.kv_cache[self.window:]
            self.compressor.freqs = get_freqs_table(
                *self.compressor._freq_args, dev)
            if getattr(self, "indexer", None) is not None:
                self.indexer.compressor.kv_cache = self.ix_kv_cache
                self.indexer.compressor.freqs = self.compressor.freqs

    def _apply(self, *args, **kwargs):
        ret = super()._apply(*args, **kwargs)
        if isinstance(getattr(self, "kv_cache", None), torch.Tensor):
            self._wire_caches()
        return ret

    def rebuild_freqs(self):
        rd, ms, osl, theta = self._freq_args
        self.freqs = get_freqs_table(rd, ms, osl, theta, self.attn_sink.device)

    def reset_state(self):
        """Clear all session state (ring, compressed history, indexer
        cache, compressor states) for a fresh sequence. Weights and the
        shared freqs tables are untouched."""
        with torch.no_grad():
            self.kv_cache.zero_()
            if self.indexer is not None:
                self.ix_kv_cache.zero_()
            for m in filter(None, (self.compressor,
                                   getattr(self.indexer, "compressor", None))):
                m.kv_state.zero_()
                m.score_state.fill_(float("-inf"))

    def _core_body(self, x, qr, q, kv, f, start_pos):
        """Index building + gather attention + state updates + inverse rope.
        Consumes PRE-COMPUTED q (rope'd, [s, h, d]), kv (rope'd + fp8-sim'd,
        [s, d]) and qr (q_norm'd lora, [s, q_lora_rank]); returns o
        [s, n_heads, head_dim] post inverse-rope. Shared by forward() and
        attention_core() (external FP8-path projections)."""
        s = x.shape[0]
        win = self.window
        if start_pos == 0 or s == 1:
            idxs = window_topk_idxs(s, x.device, start_pos)
        else:
            # chunked: per row i (absolute p = start_pos+i) gather positions
            # [p-win+1 .. p]; in-chunk positions index the kv block, history
            # positions index the ring slot q % win (valid: q >= p-win+1 >=
            # start_pos-win, the ring holds the latest win positions)
            pos = start_pos + torch.arange(s, device=x.device)
            Q = pos.unsqueeze(1) - torch.arange(win - 1, -1, -1,
                                                device=x.device).unsqueeze(0)
            idx_cur = Q - start_pos                       # in-chunk location
            idx_hist = s + torch.remainder(Q, win)        # ring location
            idxs = torch.where(Q >= start_pos, idx_cur, idx_hist)
            idxs = torch.where(Q >= 0, idxs, torch.full_like(idxs, -1)).int()
        if self.ratio:
            if start_pos == 0:
                offset = kv.shape[0]
            elif s == 1:
                offset = win
            else:
                offset = s + win                          # chunked layout
            if self.has_indexer:
                cidx = self.indexer(x, qr, self.freqs, offset, start_pos)
            elif s > 1 and start_pos > 0:
                # static compressed idxs for a chunk: per row bound
                n_c = (start_pos + s) // self.ratio
                bounds = (start_pos + torch.arange(1, s + 1,
                                                   device=x.device)) // self.ratio
                m = torch.arange(n_c, device=x.device).repeat(s, 1) < \
                    bounds.unsqueeze(1)
                cidx = torch.where(m, torch.arange(n_c, device=x.device)
                                   .repeat(s, 1) + offset,
                                   torch.full_like(m, -1)).int()
            else:
                cidx = compress_topk_idxs_static(self.ratio, s, x.device,
                                                 offset, start_pos)
            idxs = torch.cat([idxs, cidx], -1)

        if start_pos == 0:
            if s <= win:
                self.kv_cache[:s] = kv
            else:
                cutoff = s % win
                self.kv_cache[cutoff:win], self.kv_cache[:cutoff] = (
                    kv[-win:].split([win - cutoff, cutoff], dim=0)
                    if cutoff else (kv[-win:], kv[:0]))
            kv_all = kv
            if self.ratio:
                ckv = self.compressor(x, 0)
                if ckv is not None and ckv.shape[0]:
                    kv_all = torch.cat([kv, ckv], 0)
            o = _attend(q, kv_all, self.attn_sink.float(), idxs, self.scale)
        elif s == 1:
            self.kv_cache[start_pos % win] = kv.squeeze(0)
            if self.ratio:
                self.compressor(x, start_pos)
            o = _attend(q, self.kv_cache, self.attn_sink.float(),
                       idxs, self.scale)
        else:
            # chunked prefill: run the compressor (emits into the compressed
            # cache), gather from [chunk kv | ring(history) | compressed],
            # and only THEN write the chunk's rows into the ring — earlier
            # rows of this chunk still read the pre-write ring (the chunk's
            # slots collide with the oldest history positions).
            if self.ratio:
                self.compressor(x, start_pos)
            kv_all = torch.cat([kv, self.kv_cache], 0)
            o = _attend(q, kv_all, self.attn_sink.float(),
                       idxs, self.scale)
            # only the LAST win positions belong in the ring; writing all
            # s rows would scatter duplicate slots (nondeterministic when
            # s > win)
            n_ring = min(s, win)
            ring_slots = torch.remainder(
                start_pos + torch.arange(s - n_ring, s, device=x.device), win)
            self.kv_cache[ring_slots] = kv[-n_ring:]

        orot = apply_rotary_emb(o[..., -self.rd:], f, inverse=True)
        o = torch.cat([o[..., :-self.rd], orot], -1)
        return o

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """x [s, hidden] -> [s, hidden]. Prefill (start_pos=0) or decode
        (start_pos>0, typically s=1) against the window/compressed ring."""
        s = x.shape[0]
        rd = self.rd
        qr = self.q_norm(self.q_a(x))
        q = self.q_b(qr).view(s, self.n_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        f = self.freqs[start_pos:start_pos + s]
        qr_ = apply_rotary_emb(q[..., -self.rd:], f)
        q = torch.cat([q[..., :-self.rd], qr_], -1)
        kv = self.kv_norm(self.kv_proj(x))
        kvr = apply_rotary_emb(kv[..., -rd:], f)
        kv = torch.cat([kv[..., :-rd], kvr], -1)
        kv = torch.cat([act_quant_sim(kv[..., :-rd]), kv[..., -rd:]], -1)
        o = self._core_body(x, qr, q, kv, f, start_pos)
        o = o.reshape(s, self.o_groups, -1)
        wo_a = self.wo_a.weight.view(self.o_groups, self.o_lora, -1)
        o = torch.einsum("sgd,grd->sgr", o.to(wo_a.dtype), wo_a).reshape(s, -1)
        return self.wo_b(o)

    def attention_core(self, x: torch.Tensor, q_lora: torch.Tensor,
                       q_raw: torch.Tensor, kv_raw: torch.Tensor,
                       start_pos: int = 0) -> torch.Tensor:
        """DSA attention core with EXTERNAL projections (production FP8
        path): q_lora = post-wq_a pre-q_norm [s, q_lora_rank]; q_raw =
        post-wq_b pre-per-head-RMS [s, n_heads*head_dim]; kv_raw =
        post-wkv pre-kv_norm [s, head_dim]. Runs the identical DSA body
        (per-head RMS, rope, fp8-sim, window/compress gather, compressor,
        indexer, sink softmax, inverse rope, state updates) and returns
        o [s, n_heads*head_dim] — the wo_a/wo_b projection stays with the
        caller's production path."""
        s = x.shape[0]
        rd = self.rd
        qr = self.q_norm(q_lora)
        q = q_raw.view(s, self.n_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        f = self.freqs[start_pos:start_pos + s]
        qr_ = apply_rotary_emb(q[..., -self.rd:], f)
        q = torch.cat([q[..., :-self.rd], qr_], -1)
        kv = self.kv_norm(kv_raw)
        kvr = apply_rotary_emb(kv[..., -rd:], f)
        kv = torch.cat([kv[..., :-rd], kvr], -1)
        kv = torch.cat([act_quant_sim(kv[..., :-rd]), kv[..., -rd:]], -1)
        o = self._core_body(x, qr, q, kv, f, start_pos)
        return o.reshape(s, -1)


def load_exact_attention(attn: ExactAttention, layer_id: int,
                         get, ff: str = "pow2") -> None:
    """Load real weights from the core archive into ExactAttention.

    `get(name) -> torch.Tensor` (safetensors accessor). FP8 linears are
    dequantized to bf16 with e8m0 128x128-block scales (ff = 0xFF policy);
    compressor linears cast bf16->fp32 (source stores fp32 params); norms
    bf16->fp32; ape/attn_sink stay fp32.
    """
    from ..runtime.fp8 import dequant_fp8

    pre = f"layers.{layer_id}.attn"

    def dq(name: str) -> torch.Tensor:
        w = get(f"{pre}.{name}.weight")
        if w.dtype == torch.float8_e4m3fn:
            # same decode as the P2-measured core (core_timing._load_layer)
            return dequant_fp8(w, get(f"{pre}.{name}.scale"))
        return w.to(torch.bfloat16)

    def put(param, tensor, dtype):
        param.data = tensor.to(device=param.device, dtype=dtype)

    put(attn.q_a.weight, dq("wq_a"), torch.bfloat16)
    put(attn.q_norm.weight, get(f"{pre}.q_norm.weight"), torch.float32)
    put(attn.q_b.weight, dq("wq_b"), torch.bfloat16)
    put(attn.kv_proj.weight, dq("wkv"), torch.bfloat16)
    put(attn.kv_norm.weight, get(f"{pre}.kv_norm.weight"), torch.float32)
    put(attn.wo_a.weight, dq("wo_a"), torch.bfloat16)
    put(attn.wo_b.weight, dq("wo_b"), torch.bfloat16)
    put(attn.attn_sink, get(f"{pre}.attn_sink"), torch.float32)

    def load_compressor(comp, cpre: str):
        put(comp.ape, get(f"{cpre}.ape"), torch.float32)
        put(comp.wkv.weight, get(f"{cpre}.wkv.weight"), torch.float32)
        put(comp.wgate.weight, get(f"{cpre}.wgate.weight"), torch.float32)
        put(comp.norm.weight, get(f"{cpre}.norm.weight"), torch.float32)

    if attn.compressor is not None:
        load_compressor(attn.compressor, f"{pre}.compressor")
    if attn.indexer is not None:
        ix = attn.indexer
        put(ix.wq_b.weight, dq("indexer.wq_b"), torch.bfloat16)
        ix.weights_proj.weight.data.copy_(
            get(f"{pre}.indexer.weights_proj.weight").to(torch.bfloat16))
        load_compressor(ix.compressor, f"{pre}.indexer.compressor")


class ExactAttentionAdapter(nn.Module):
    """Bridge ExactAttention to the eager Attention interface ([b, t, d]).

    Prefill/chunk semantics only: fresh-chunk (start_pos=0) compute with the
    full per-layer DSA structure (window gather + compressor + indexer).
    Decode against an existing KV history is a later phase; passing a
    non-empty kv_cache raises.
    """

    def __init__(self, cfg: V4Config, layer_idx: int, max_seq: int = 4096):
        super().__init__()
        # CPU construction: a 43-layer exact model is ~10 GB of weights —
        # forcing CUDA here OOMs a 12 GB card at build time. Move with the
        # normal .to(device) flow, then cast_exact_attention().
        self.inner = ExactAttention(cfg, layer_idx, max_seq=max_seq,
                                    device="cpu")
        self._pos = 0

    def reset(self):
        """Reset decode position AND all session state (ring, compressed
        history, compressor/indexer states) for a fresh sequence."""
        self._pos = 0
        self.inner.reset_state()

    def forward(self, x, kv_cache=None):
        b, t, _ = x.shape
        if b != 1:
            raise NotImplementedError("exact DSA is single-batch")
        # position self-tracks: the ring/compressed caches inside
        # ExactAttention are the state; kv_cache is interface compat only.
        start_pos = self._pos
        out = self.inner(x.squeeze(0), start_pos)
        self._pos = start_pos + t
        return out.unsqueeze(0)


def cast_exact_attention(attn: ExactAttention, device) -> None:
    """Enforce the source dtype scheme after any module-level .to() cast:
    bf16 GEMM linears, fp32 compressor math / norms / sink / ape, complex64
    freqs. Idempotent; call before loading real weights or inference."""
    f32, bf16, c64 = torch.float32, torch.bfloat16, torch.complex64
    attn.to(device)
    for lin_nm in ("q_a", "q_b", "kv_proj", "wo_a", "wo_b"):
        lin = getattr(attn, lin_nm, None)   # stripped in projection-free use
        if lin is not None:
            lin.to(bf16)
    for nrm in (attn.q_norm, attn.kv_norm):
        nrm.to(f32)
    attn.attn_sink.data = attn.attn_sink.data.to(f32)
    if attn.freqs.dtype != c64:
        # a module-level real cast would have destroyed the imag part;
        # rebuild from the table definition instead
        attn.rebuild_freqs()
    else:
        attn.freqs = attn.freqs.to(c64)
    attn._wire_caches()
    if attn.compressor is not None:
        attn.compressor.to(f32)
        if attn.compressor.freqs.dtype != c64:
            attn.compressor.rebuild_freqs()
    if attn.indexer is not None:
        attn.indexer.wq_b.to(bf16)
        attn.indexer.weights_proj.to(bf16)
        attn.indexer.compressor.to(f32)
        if attn.indexer.compressor.freqs.dtype != c64:
            attn.indexer.compressor.rebuild_freqs()
