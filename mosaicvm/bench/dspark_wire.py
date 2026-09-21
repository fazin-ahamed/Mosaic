"""DSpark Stage-2: wire the drafter from the OBSERVED Stage-1 interface.

Runs on the Blackwell sandbox. Loads the main model via transformers
(device_map=auto), hooks layers 40/41/42 for main_hidden, loads the mtp.*
tensors from the HF shards, and implements the three DSpark stages in
PyTorch bf16 following the checkpoint's own source kit exactly:

  * Block hyper-connections (hc_pre sinkhorn, hc_post)
  * DSparkAttention (windowed ring KV over target hiddens, YaRN rope,
    sparse attention with denominator-only attn_sink)
  * MoE (sqrtsoftplus gate, bias-shifted top-6, SwIGLU experts with
    swiglu_limit clamp, shared expert)
  * MarkovHead (bigram logit bias), ConfidenceHead
  * draft block [committed, noise*4] -> 5 proposed tokens sampled at T=1.0

Kit reference: .mosaicvm/source/inference_model.py (DSparkBlock,
DSparkAttention, DSparkMarkovHead, DSparkConfidenceHead, Transformer.
forward_spec), inference_kernel.py (hc_split_sinkhorn, sparse_attn),
inference_config.json.

Numerics deviations from the source kit (recorded in the artifact):
FP8 weights dequantized to bf16; FP4 experts dequantized per-call; KV bf16;
activations not FP8-quantized.

Measurement: chain acceptance of drafted blocks vs greedy target (argmax
equality, first-mismatch stop, bonus token), batch verification with KV
cache crop. All timing/memory = BLACKWELL_SANDBOX_ONLY.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, "/marimo/MosaicVM")
from mosaicvm.bench.dspark_tree import analyze as analyze_d0d3

OUT_DIR = "/marimo/MosaicVM/artifacts"

# ---- constants from inference_config.json + Stage-1 inspection ----
N_STAGES = 3
TARGET_LAYERS = [40, 41, 42]
BLOCK_SIZE = 5
NOISE_TOKEN_ID = 128799
MARKOV_RANK = 256
HC_MULT = 4
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-6
WINDOW = 128
DIM = 4096
MOE_INTER = 2048
N_HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
Q_LORA = 1024
O_GROUPS = 8
O_LORA = 1024
N_EXPERTS = 256
TOPK = 6
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
NORM_EPS = 1e-6
VOCAB = 129280
TEMPERATURE = 1.0
ROPE_THETA = 10000
ROPE_FACTOR = 16
ORIGINAL_SEQ_LEN = 65536
BETA_FAST = 32
BETA_SLOW = 1
N_BLOCKS = 30                    # draft blocks measured per prompt
MAIN_GPU_GIB = 76                # leave VRAM for the MTP stages
TOPK_COVER = (1, 2, 4, 8, 16)    # D2 coverage k-set
TREE_BUDGETS = (16, 32, 48, 64, 80, 96, 112, 124, 128, 144, 160)
RNG_SEED = 20260829

PROMPTS = [
    ("coding", "def fibonacci(n):"),
    ("agent_tool_use",
     "Call the weather API for Tokyo tomorrow with units=celsius."),
    ("math_reasoning", "Prove that sqrt(2) is irrational:"),
    ("general_conversation", "The history of the silk road begins"),
    ("long_context_retrieval",
     "In a long-context retrieval setting, the key idea is"),
    ("structured_json",
     'Return a JSON object with fields "name", "age", "email" for a user '
     'profile schema, validating that'),
    ("long_generation",
     "Write a detailed technical explanation of how residual networks "
     "improved image classification, covering"),
]

FP4_TABLE_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ================================================================ rope ======
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor,
                         beta_fast, beta_slow):
    """Kit-exact YaRN rope (inference_model.py)."""
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) \
            / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_, max_, dim):
        if min_ == max_:
            max_ += 0.001
        lf = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
        return torch.clamp(lf, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32)
                             / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                          original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary(x, freqs_cis, inverse=False):
    """Kit-exact rotary on the last dim (complex pairs)."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    fc = freqs_cis
    if inverse:
        fc = fc.conj()
    if xc.ndim == 3:                     # [B, S, D/2]
        fc = fc.view(1, xc.size(1), xc.size(-1))
    else:                                # [B, S, H, D/2]
        fc = fc.view(1, xc.size(1), 1, xc.size(-1))
    out = torch.view_as_real(xc * fc).flatten(-2)
    return out.to(x.dtype)


def _rope_last(x, freqs_cis):
    """Apply rotary to the last ROPE_DIM of x in place-ish (clone)."""
    x = x.clone()
    x[..., -ROPE_DIM:] = apply_rotary(x[..., -ROPE_DIM:], freqs_cis)
    return x


def _rope_inv_last(x, freqs_cis):
    """Inverse rotary on the last ROPE_DIM (4D [B, S, H, D] input)."""
    x = x.clone()
    x[..., -ROPE_DIM:] = apply_rotary(x[..., -ROPE_DIM:], freqs_cis,
                                      inverse=True)
    return x


# ============================================================== dequant =====
def dequant_fp8(weight, scale, block=128):
    """FP8 [out, in] + scale [out, in/128] (or [out/128, in/128]) -> bf16."""
    out_dim, in_dim = weight.shape
    w = weight.to(torch.float32)
    # scales are float8_e8m0fnu (ue8m0) in this checkpoint
    s8 = scale.float()
    if s8.dim() == 2 and s8.shape == (out_dim, in_dim // block):
        s = s8.repeat_interleave(block, dim=1)
    elif s8.dim() == 2 and s8.shape == (out_dim // block,
                                         in_dim // block):
        s = s8.repeat_interleave(block, 0).repeat_interleave(block, 1)
    else:
        raise ValueError(f"unexpected scale {tuple(scale.shape)} for "
                         f"weight {tuple(weight.shape)}")
    return (w * s).to(torch.bfloat16)


def dequant_fp4(weight_i8, scale, out_dim, in_dim, fp4_table):
    """Packed FP4 int8 [out, in/2] + scale [out, in/32] -> bf16 [out, in]."""
    x = weight_i8.view(torch.uint8)
    low = (x & 0x0F).long()
    high = ((x >> 4) & 0x0F).long()
    vals = torch.stack([fp4_table[low], fp4_table[high]], dim=-1).flatten(-2)
    s = scale.float().repeat_interleave(32, dim=1)
    return (vals * s).to(torch.bfloat16)


# ============================================================== sinkhorn ====
def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=HC_MULT,
                      iters=HC_SINKHORN_ITERS, eps=HC_EPS):
    """Kit-exact hc_split_sinkhorn (inference_kernel.py lines 391-425)."""
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].unflatten(-1, (hc, hc)) * hc_scale[2] \
        + hc_base[2 * hc:].unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


# ============================================================== modules =====
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * xf).to(dtype)


class Lin:
    """FP8 linear dequantized to bf16 at load, on `device`.

    If `scale` is None the weight is unquantized (router/markov weights
    in this checkpoint ship as plain bf16).
    """

    def __init__(self, weight, scale, device):
        if scale is None:
            self.weight = weight.to(torch.bfloat16).to(device)
        else:
            self.weight = dequant_fp8(weight, scale).to(device)

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype))


class SwiGLUExpert:
    def __init__(self, w1, w2, w3):
        self.w1, self.w2, self.w3 = w1, w2, w3

    def __call__(self, x, weight=None):
        gate = torch.nn.functional.linear(x, self.w1.to(x.dtype)).float()
        up = torch.nn.functional.linear(x, self.w3.to(x.dtype)).float()
        up = torch.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        gate = torch.clamp(gate, max=SWIGLU_LIMIT)
        h = torch.nn.functional.silu(gate) * up
        if weight is not None:
            h = weight * h
        return torch.nn.functional.linear(h.to(x.dtype),
                                          self.w2.to(x.dtype))


class MoEDQ:
    """Kit-exact MoE for one MTP stage (learned router)."""

    def __init__(self, t, fp4_table, device):
        gate_scale = t.get("gate.scale")
        if gate_scale is not None:
            self.gate_w = dequant_fp8(t["gate.weight"], gate_scale) \
                .float().to(device)
        else:                       # router weight ships unquantized
            self.gate_w = t["gate.weight"].to(torch.bfloat16).float() \
                .to(device)
        self.gate_b = (t["gate.bias"].float().to(device)
                       if "gate.bias" in t else None)
        # shared experts ship FP8 with 2D 128x128-block scales (unlike
        # the routed experts, which are FP4-packed int8)
        self.shared = SwiGLUExpert(
            dequant_fp8(t["shared.w1.weight"], t["shared.w1.scale"])
            .to(device),
            dequant_fp8(t["shared.w2.weight"], t["shared.w2.scale"])
            .to(device),
            dequant_fp8(t["shared.w3.weight"], t["shared.w3.scale"])
            .to(device))
        # packed experts on GPU; dequant selected on demand
        self.packed = {}
        for i in range(N_EXPERTS):
            self.packed[i] = tuple(
                t[f"experts.{i}.w{n}.{k}"].to(device)
                for n in (1, 2, 3) for k in ("weight", "scale"))
        self.table = fp4_table
        self.cache = {}
        self.cache_limit = 32

    def _expert(self, i):
        if i not in self.cache:
            if len(self.cache) >= self.cache_limit:
                self.cache.pop(next(iter(self.cache)))
            p = self.packed[i]
            self.cache[i] = SwiGLUExpert(
                dequant_fp4(p[0], p[1], MOE_INTER, DIM, self.table),
                dequant_fp4(p[2], p[3], DIM, MOE_INTER, self.table),
                dequant_fp4(p[4], p[5], MOE_INTER, DIM, self.table))
        return self.cache[i]

    def __call__(self, x):
        shape = x.size()
        xf = x.reshape(-1, DIM).float()
        scores = xf @ self.gate_w.t()
        scores = torch.nn.functional.softplus(scores).sqrt()
        orig = scores
        if self.gate_b is not None:
            scores = scores + self.gate_b
        indices = scores.topk(TOPK, dim=-1)[1]
        weights = orig.gather(1, indices)
        weights = weights / weights.sum(-1, keepdim=True) * ROUTE_SCALE
        y = torch.zeros_like(xf)
        for e in indices.flatten().unique().tolist():
            sel = (indices == e)
            if not sel.any():
                continue
            idx, top = torch.where(sel)
            y[idx] += self._expert(e)(xf[idx].to(torch.bfloat16),
                                      weights[idx, top, None]).float()
        y += self.shared(x.reshape(-1, DIM)).float()
        return y.to(x.dtype).view(shape)


class DSparkAttention:
    """Kit-exact DSparkAttention in bf16.

    Ring KV cache (WINDOW) stores the TARGET's projected hidden KV; draft
    queries attend to ring slots [0, min(win, start_pos+1)) plus the
    current block's KV appended at ring offset (kit get_dspark_topk_idxs).
    attn_sink enters the softmax denominator only.
    """

    def __init__(self, t, freqs_cis, device):
        self.wq_a = Lin(t["wq_a.weight"], t["wq_a.scale"], device)
        self.wq_b = Lin(t["wq_b.weight"], t["wq_b.scale"], device)
        self.wkv = Lin(t["wkv.weight"], t["wkv.scale"], device)
        self.wo_a = dequant_fp8(t["wo_a.weight"], t["wo_a.scale"]).to(device)
        self.wo_b = Lin(t["wo_b.weight"], t["wo_b.scale"], device)
        self.q_norm = RMSNorm(Q_LORA)
        with torch.no_grad():
            self.q_norm.weight.copy_(t["q_norm.weight"].float().to(device))
        self.kv_norm = RMSNorm(HEAD_DIM)
        with torch.no_grad():
            self.kv_norm.weight.copy_(t["kv_norm.weight"].float().to(device))
        self.q_norm.to(device)
        self.kv_norm.to(device)
        self.attn_sink = t["attn_sink"].float().to(device)
        self.softmax_scale = HEAD_DIM ** -0.5
        self.freqs_cis = freqs_cis
        self.kv_cache = torch.zeros(WINDOW, HEAD_DIM,
                                    dtype=torch.bfloat16, device=device)

    def __call__(self, x, start_pos, main_x):
        return self.forward(x, start_pos, main_x)

    def forward(self, x, start_pos, main_x):
        """x [1, block, dim] draft tokens; main_x [1, S, dim] target proj."""
        win = WINDOW
        main_seqlen = main_x.size(1)

        # main KV (target hidden KV) with rope at its own positions
        main_kv = self.kv_norm(self.wkv(main_x.to(torch.bfloat16)))
        main_kv = _rope_last(
            main_kv, self.freqs_cis[start_pos:start_pos + main_seqlen])

        if start_pos == 0:
            # prefill: fill the ring (kit wrap semantics)
            if main_seqlen <= win:
                self.kv_cache[:main_seqlen] = main_kv[0]
            else:
                cutoff = main_seqlen % win
                tail = main_kv[0, -win:]
                if cutoff == 0:
                    self.kv_cache[:] = tail
                else:
                    self.kv_cache[cutoff:] = tail[:win - cutoff]
                    self.kv_cache[:cutoff] = tail[win - cutoff:]
            return x

        # ring store at the current committed position
        self.kv_cache[start_pos % win] = main_kv[0, -1]

        block = x.size(1)
        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (N_HEADS, HEAD_DIM))
        q = q * torch.rsqrt(
            q.float().square().mean(-1, keepdim=True)
            + NORM_EPS).to(q.dtype)
        draft_fc = self.freqs_cis[start_pos + main_seqlen:
                                  start_pos + main_seqlen + block]
        q = _rope_last(q, draft_fc)
        kv = self.kv_norm(self.wkv(x))
        kv = _rope_last(kv, draft_fc)

        # kit get_dspark_topk_idxs: ring slots [0, min(win, start_pos+1))
        # then block slots at ring offset
        n_attend = min(win, start_pos + 1)
        full_kv = torch.cat([self.kv_cache, kv[0]], dim=0)  # [win+block, d]
        sel = list(range(n_attend)) + list(range(win, win + block))
        kv_sel = full_kv[sel]                               # [T, d]

        qf = q[0].float()                                   # [block, H, d]
        scores = torch.einsum("bhd,td->bht", qf, kv_sel.float()) \
            * self.softmax_scale
        m = scores.amax(-1, keepdim=True)
        p = torch.exp(scores - m)
        denom = p.sum(-1, keepdim=True) \
            + torch.exp(self.attn_sink.view(1, -1, 1) - m)
        o = torch.einsum("bht,td->bhd", p / denom, kv_sel.float())
        # inverse rope on output at query positions (keep 4D kit shape)
        o4 = o.unsqueeze(0).to(torch.bfloat16)              # [1, b, H, d]
        o4 = _rope_inv_last(o4, draft_fc)
        o = o4[0].float()

        # grouped output projection (kit wo_a/wo_b)
        o_g = o.reshape(block, O_GROUPS, -1)                # [b, g, d]
        wo_a_w = self.wo_a.view(O_GROUPS, O_LORA, -1).float()
        o_r = torch.einsum("bgd,grd->bgr", o_g, wo_a_w)     # [b, g, r]
        out = self.wo_b(o_r.reshape(1, block, O_GROUPS * O_LORA)
                        .to(torch.bfloat16))
        return out


class DSparkStage:
    """One MTP stage: HC attn + HC MoE (kit Block + DSparkBlock)."""

    def __init__(self, stage_id, t, freqs_cis, device, fp4_table):
        self.stage_id = stage_id
        self.attn = DSparkAttention(t["attn"], freqs_cis, device)
        self.ffn = MoEDQ(t["ffn"], fp4_table, device)
        self.attn_norm = RMSNorm(DIM)
        with torch.no_grad():
            self.attn_norm.weight.copy_(t["attn_norm"].float())
        self.ffn_norm = RMSNorm(DIM)
        with torch.no_grad():
            self.ffn_norm.weight.copy_(t["ffn_norm"].float())
        self.attn_norm.to(device)
        self.ffn_norm.to(device)
        self.hc_attn_fn = t["hc_attn_fn"].float().to(device)
        self.hc_attn_base = t["hc_attn_base"].float().to(device)
        self.hc_attn_scale = t["hc_attn_scale"].float().to(device)
        self.hc_ffn_fn = t["hc_ffn_fn"].float().to(device)
        self.hc_ffn_base = t["hc_ffn_base"].float().to(device)
        self.hc_ffn_scale = t["hc_ffn_scale"].float().to(device)
        if stage_id == 0:
            self.main_proj = Lin(t["main_proj.weight"], t["main_proj.scale"],
                                 device)
            self.main_norm = RMSNorm(DIM)
            with torch.no_grad():
                self.main_norm.weight.copy_(t["main_norm"].float().to(device))
            self.main_norm.to(device)
        if stage_id == N_STAGES - 1:
            self.norm = RMSNorm(DIM)
            with torch.no_grad():
                self.norm.weight.copy_(t["norm"].float())
            self.norm.to(device)
            self.markov_w1 = t["markov_w1.weight"].to(torch.bfloat16).to(
                device)
            self.markov_w2 = Lin(t["markov_w2.weight"],
                                 t.get("markov_w2.scale"), device)
            self.conf_proj = t["confidence_proj.weight"].float().to(device)
            self.hc_head_fn = t["hc_head_fn"].float().to(device)
            self.hc_head_base = t["hc_head_base"].float().to(device)
            self.hc_head_scale = t["hc_head_scale"].float().to(device)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
        mixes = torch.nn.functional.linear(xf, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    @staticmethod
    def hc_post(x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def __call__(self, x, start_pos, main_x):
        return self.forward(x, start_pos, main_x)

    def forward(self, x, start_pos, main_x):
        if start_pos == 0:
            # kit DSparkBlock.forward: prefill only fills attention KV
            self.attn(x, start_pos, main_x)
            return x
        residual = x
        h, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale,
                                    self.hc_attn_base)
        h = self.attn_norm(h)
        h = self.attn(h, start_pos, main_x)
        x = self.hc_post(h, residual, post, comb)
        residual = x
        h, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale,
                                    self.hc_ffn_base)
        h = self.ffn_norm(h)
        h = self.ffn(h)
        x = self.hc_post(h, residual, post, comb)
        return x

    def forward_embed(self, main_hidden, input_ids, embed_weight):
        """Kit DSparkBlock.forward_embed. main_hidden [1, S, 3*dim]."""
        main_x = self.main_norm(self.main_proj(main_hidden))
        ids = input_ids.reshape(-1)             # [B]
        draft_ids = ids.new_full((ids.size(0), BLOCK_SIZE), NOISE_TOKEN_ID)
        draft_ids[:, 0] = ids
        x = torch.nn.functional.embedding(
            draft_ids, embed_weight.to(torch.bfloat16))
        x = x.unsqueeze(2).repeat(1, 1, HC_MULT, 1)
        return x, main_x

    def forward_head(self, x, input_ids, lm_head_weight):
        """Kit DSparkBlock.forward_head (last stage).

        Kit hc_head: hc_head_fn [hc_mult, hc_dim], hc_head_base [hc_mult],
        hc_head_scale [1] (scalar) — pre = sigmoid(mixes*scale + base)+eps;
        no sinkhorn (the head is terminal).
        """
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
        mixes = torch.nn.functional.linear(xf, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale
                            + self.hc_head_base) + HC_EPS
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2).to(dtype)
        logits = torch.nn.functional.linear(
            self.norm(y).to(lm_head_weight.dtype), lm_head_weight)

        # Two independent chains from the SAME logits:
        #   greedy  — always argmax (D1)
        #   sampled — T=1.0 gumbel (D0, source default)
        # Markov bias at depth i uses the PREVIOUS token of THAT chain.
        greedy_ids = input_ids.new_empty(input_ids.size(0), BLOCK_SIZE + 1)
        sampled_ids = input_ids.new_empty(input_ids.size(0), BLOCK_SIZE + 1)
        greedy_ids[:, 0] = input_ids
        sampled_ids[:, 0] = input_ids
        greedy_logits = []
        sampled_logits = []
        greedy_topk = []
        sampled_topk = []
        greedy_embs = []
        sampled_embs = []
        k_cover = 16
        for i in range(BLOCK_SIZE):
            # --- greedy chain ---
            g_emb = torch.nn.functional.embedding(greedy_ids[:, i],
                                                   self.markov_w1)
            g_bias = self.markov_w2(g_emb)
            g_log = (logits[:, i].float() + g_bias.float())
            greedy_logits.append(g_log)
            g_topv, g_topi = g_log.topk(k_cover, dim=-1)
            greedy_topk.append((g_topi[0].tolist(), g_topv[0].tolist()))
            greedy_ids[:, i + 1] = g_log.argmax(-1)
            greedy_embs.append(g_emb)
            # --- sampled chain ---
            s_emb = torch.nn.functional.embedding(sampled_ids[:, i],
                                                   self.markov_w1)
            s_bias = self.markov_w2(s_emb)
            s_log = (logits[:, i].float() + s_bias.float())
            sampled_logits.append(s_log)
            s_topv, s_topi = s_log.topk(k_cover, dim=-1)
            sampled_topk.append((s_topi[0].tolist(), s_topv[0].tolist()))
            gumbel = -torch.log(-torch.log(
                torch.rand_like(s_log) + 1e-9) + 1e-9)
            sampled_ids[:, i + 1] = (
                s_log / max(TEMPERATURE, 1e-5) + gumbel).argmax(-1)
            sampled_embs.append(s_emb)
        greedy_embed = torch.stack(greedy_embs, dim=1)
        sampled_embed = torch.stack(sampled_embs, dim=1)
        conf_g = torch.nn.functional.linear(
            torch.cat([y.float(), greedy_embed.float()], dim=-1),
            self.conf_proj).squeeze(-1)
        conf_s = torch.nn.functional.linear(
            torch.cat([y.float(), sampled_embed.float()], dim=-1),
            self.conf_proj).squeeze(-1)
        return {
            "greedy_ids": greedy_ids,
            "sampled_ids": sampled_ids,
            "greedy_topk": greedy_topk,     # list of (ids, scores) per depth
            "sampled_topk": sampled_topk,
            "confidence_greedy": conf_g,
            "confidence_sampled": conf_s,
            "base_logits": logits,          # pre-markov, shared
        }


# ================================================================ loader ====
def load_mtp_tensors(model_dir, device):
    """Extract all mtp.* tensors from the HF shards; organize per stage."""
    from safetensors import safe_open

    idx = json.load(open(os.path.join(model_dir,
                                      "model.safetensors.index.json")))
    weight_map = {k: v for k, v in idx["weight_map"].items()
                  if k.startswith("mtp.")}
    shards = sorted(set(weight_map.values()))
    log(f"mtp: {len(weight_map)} tensors across {len(shards)} shards")
    raw = {}
    for shard in shards:
        with safe_open(os.path.join(model_dir, shard),
                       framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    raw[k] = f.get_tensor(k)
    stages = []
    for s in range(N_STAGES):
        t = {}
        pre = f"mtp.{s}."
        for k, v in raw.items():
            if not k.startswith(pre):
                continue
            rel = k[len(pre):]
            if rel.startswith("attn."):
                t.setdefault("attn", {})[rel[5:]] = v
            elif rel.startswith("ffn.gate."):
                t.setdefault("ffn", {})[f"gate.{rel[9:]}"] = v
            elif rel.startswith("ffn.shared_experts."):
                t.setdefault("ffn", {})[
                    f"shared.{rel[len('ffn.shared_experts.'):]}"] = v
            elif rel.startswith("ffn.experts."):
                t.setdefault("ffn", {})[rel[4:]] = v
            elif rel in ("attn_norm.weight", "ffn_norm.weight",
                         "norm.weight", "main_norm.weight"):
                t[rel[:-len(".weight")]] = v
            elif rel.startswith("main_proj."):
                t[f"main_proj.{rel[len('main_proj.'):]}"] = v
            elif rel.startswith("markov_head.markov_w1."):
                t["markov_w1.weight"] = v
            elif rel.startswith("markov_head.markov_w2."):
                t[f"markov_w2.{rel[len('markov_head.markov_w2.'):]}"] = v
            elif rel.startswith("confidence_head.proj."):
                t["confidence_proj.weight"] = v
            elif rel.startswith("hc_"):
                t[rel] = v
            else:
                t[rel] = v
        stages.append(t)
    log(f"stage tensor counts: {[len(t) for t in stages]}")
    return stages


# ================================================================ main ======
def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download("deepseek-ai/DeepSeek-V4-Flash-0731")
    revision = os.path.basename(model_dir)

    log("loading tokenizer + main model "
        f"(gpu {MAIN_GPU_GIB}GiB + cpu)...")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=cfg, torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: f"{MAIN_GPU_GIB}GiB", "cpu": "100GiB"},
        offload_folder="/tmp/offload",
        trust_remote_code=True)
    model.eval()
    log(f"main model loaded ({time.time()-t_start:.0f}s)")

    # hooks: layers 40/41/42 outputs -> mean over hc -> captured
    captured = []

    def hook(mod, args, output):
        captured.append(output.detach().mean(dim=2))

    handles = [model.model.layers[li].register_forward_hook(hook)
               for li in TARGET_LAYERS]

    dev = next(p.device for p in model.parameters()
               if p.device.type == "cuda")

    def _from_ckpt(key):
        # load straight from the shards (model may hold it on meta)
        from safetensors import safe_open
        idx = json.load(open(os.path.join(
            model_dir, "model.safetensors.index.json")))
        shard = idx["weight_map"].get(key)
        if shard is None:
            return None
        with safe_open(os.path.join(model_dir, shard),
                       framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    # NOTE: the checkpoint ships a single tied `head.weight` (no
    # lm_head.weight / embed_tokens.weight) — kit ParallelHead/embed.
    head_weight = _from_ckpt("head.weight")
    assert head_weight is not None, "head.weight not found in checkpoint"
    head_weight = head_weight.to(dev)
    lm_head_weight = head_weight
    embed_weight = model.model.embed_tokens.weight.data
    if embed_weight.device.type != "cuda":
        embed_weight = head_weight      # tied
    log(f"embed@{embed_weight.device} lm_head@{lm_head_weight.device} "
        f"({time.time()-t_start:.0f}s)")

    # ---- MTP ----
    log("loading mtp tensors...")
    fp4_table = torch.tensor(FP4_TABLE_VALUES, dtype=torch.float32,
                             device=dev)
    raw_stages = load_mtp_tensors(model_dir, dev)
    freqs_cis = precompute_freqs_cis(
        ROPE_DIM, 8192, ORIGINAL_SEQ_LEN, ROPE_THETA, ROPE_FACTOR,
        BETA_FAST, BETA_SLOW).to(dev)
    stages = []
    for s in range(N_STAGES):
        stages.append(DSparkStage(s, raw_stages[s], freqs_cis, dev,
                                  fp4_table))
        log(f"stage {s} built ({time.time()-t_start:.0f}s, "
            f"VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB)")

    # ---- drafter API (kit Transformer.forward_spec) ----
    def forward_spec(input_ids, main_hidden, start_pos):
        """input_ids [1]; main_hidden [1, S, 3*dim]; start_pos int."""
        h, main_x = stages[0].forward_embed(main_hidden, input_ids,
                                            embed_weight)
        for st in stages:
            h = st(h, start_pos, main_x)
        if start_pos == 0:
            return None                    # prefill fills KV only
        return stages[-1].forward_head(h, input_ids, lm_head_weight)

    def target_forward(tokens, past=None):
        """Feed tokens [1, S] to the target; returns (logits, past, mh)."""
        captured.clear()
        with torch.no_grad():
            out = model(tokens, past_key_values=past, use_cache=True)
        mh = torch.cat(captured, dim=-1)   # [1, S, 3*dim]
        captured.clear()
        return out.logits, out.past_key_values, mh

    def crop(past, n):
        try:
            past.crop(n)
            return past, True
        except Exception:
            return past, False

    # ---- measurement ----
    results = {
        "phase": "DSpark-stage2-acceptance",
        "model_revision": revision,
        "classification": "BLACKWELL_SANDBOX_ONLY",
        "numerics_deviations": [
            "FP8 weights dequant to bf16 (value-exact; accumulation differs)",
            "FP4 experts dequant per-call (value-exact)",
            "drafter KV stored bf16 (source stores FP8)",
            "no activation FP8 quant in drafter (source act_quant skipped)",
            "drafter sampling T=1.0 gumbel-max (source default)",
        ],
        "interpretation": [
            "kit semantics: forward_spec(input_ids=token at position "
            "start_pos, main_hidden of that token, start_pos); proposals "
            "cover start_pos+1 .. start_pos+block_size",
            "chain verification: greedy argmax equality, first-mismatch "
            "stop, bonus token emitted",
            "drafter ring KV filled densely: intermediate committed tokens "
            "get forward_spec KV fills between blocks",
            "verification uses transformers target with KV cache + crop "
            "rollback",
        ],
        "per_prompt": {},
    }

    for cat, prompt_text in PROMPTS:
        log(f"[{cat}] measuring {N_BLOCKS} blocks...")
        t0 = time.time()
        input_ids = tok(prompt_text, return_tensors="pt").input_ids.to(dev)
        n_prompt = input_ids.shape[1]

        # target prefill
        logits, past, mh_full = target_forward(input_ids)
        g_next = logits[0, -1].argmax().item()
        cur_mh = mh_full[0, -1:]            # [1, 3*dim] at last prompt pos

        # drafter prefill: fill ring caches from the whole prompt
        with torch.no_grad():
            forward_spec(input_ids[:, -1:], mh_full, 0)

        depths = []
        streams = []
        committed = n_prompt            # cache length; last token at pos n-1
        # last COMMITTED token (position committed-1) and its main hidden
        last_tok = input_ids[0, -1].item()
        last_mh = cur_mh                # hidden of the token at committed-1

        for block_i in range(N_BLOCKS):
            # --- draft: kit semantics — input_ids is the token AT POSITION
            # start_pos (= committed-1, the last committed token); proposals
            # cover positions committed .. committed+4.
            torch.manual_seed(RNG_SEED + block_i)
            with torch.no_grad():
                spec = forward_spec(
                    torch.tensor([[last_tok]], device=dev),
                    last_mh.unsqueeze(0), committed - 1)
            sampled = spec["sampled_ids"][0, 1:].tolist()
            greedy_d = spec["greedy_ids"][0, 1:].tolist()
            topk_ids = [ids for ids, _ in spec["greedy_topk"]]
            topk_scores = [sc for _, sc in spec["greedy_topk"]]
            conf_g = spec["confidence_greedy"][0].float().tolist()
            conf_s = spec["confidence_sampled"][0].float().tolist()

            # --- D1 verify: sequential greedy-chain vs target, then crop
            # back so D0 can reuse the same cache. target_greedy[0] = g_next
            # is path-independent; later depths require feeding greedy[i-1].
            L0 = past.get_seq_length()
            target_greedy = [g_next]
            g_tok = g_next
            for i in range(BLOCK_SIZE):
                if greedy_d[i] != g_tok:
                    # mismatch at depth i; remaining target tokens unknown
                    # (not needed — D1 prefix stops here)
                    break
                # feed the matching greedy token to get the NEXT target
                gl, past, _ = target_forward(
                    torch.tensor([[greedy_d[i]]], device=dev), past)
                g_tok = gl[0, -1].argmax().item()
                target_greedy.append(g_tok)
            past, _ = crop(past, L0)

            # --- D0 verify: batch-feed the SAMPLED chain
            draft_t = torch.tensor([sampled], device=dev)
            vlogits, past, vmh = target_forward(draft_t, past)
            target_sampled = [g_next] + vlogits[0].argmax(-1).tolist()
            k = 0
            while k < BLOCK_SIZE and sampled[k] == target_sampled[k]:
                k += 1
            depths.append(k)
            target_toks = target_sampled   # D0 path; D1 uses target_greedy

            # --- commit: k accepted + 1 bonus (target_toks[k]); rollback
            bonus = target_toks[k]
            L = past.get_seq_length() - BLOCK_SIZE   # cache len pre-verify
            past, ok = crop(past, L + k)
            if not ok:
                log("  WARNING: cache crop unavailable; results degraded")
            # feed the bonus token at position L+k
            logits, past, mh_b = target_forward(
                torch.tensor([[bonus]], device=dev), past)
            g_next = logits[0, -1].argmax().item()
            # dense drafter ring fill: accepted drafts d_1..d_k occupied
            # positions L..L+k-1 (hiddens vmh[0, 0..k-1]); the bonus at L+k
            # is stored by the NEXT draft call itself.
            with torch.no_grad():
                for i in range(k):
                    forward_spec(
                        torch.tensor([[sampled[i]]], device=dev),
                        vmh[0, i:i + 1].unsqueeze(0), L + i)
            committed = L + k + 1
            last_tok = bonus
            last_mh = mh_b[0, -1:]
            streams.append({
                "block": block_i,
                "start_pos": committed - k - 2,  # last committed pos
                "target": target_toks[:BLOCK_SIZE],
                "target_greedy_path": target_greedy[:BLOCK_SIZE],
                "sampled": sampled,
                "greedy": greedy_d,
                "topk_ids": topk_ids,
                "topk_scores": topk_scores,
                "confidence": conf_g,
                "confidence_sampled": conf_s,
                "bonus": bonus,
                "accepted_sampled": k,
                "rng_seed": RNG_SEED + block_i,
            })
            gk = 0
            while gk < BLOCK_SIZE and greedy_d[gk] == target_toks[gk]:
                gk += 1
            streams[-1]["accepted_greedy"] = gk

        elapsed = time.time() - t0
        ea = sum(depths) / len(depths) if depths else 0.0
        hist = {str(i): depths.count(i) for i in range(BLOCK_SIZE + 1)}
        d0d3 = analyze_d0d3(streams)
        results["per_prompt"][cat] = {
            "n_blocks": len(depths),
            "elapsed_s": round(elapsed, 1),
            "classification": "BLACKWELL_SANDBOX_ONLY",
            "D0": d0d3["D0"],
            "D1": d0d3["D1"],
            "D2": d0d3["D2"],
            "D3": {n: d0d3["D3"]["budgets"][n]["E[A]"]
                   for n in d0d3["D3"]["budgets"]},
            "streams": streams,          # FULL raw records
        }
        log(f"[{cat}] D0 E[A]={d0d3['D0']['E[A]']} "
            f"D1 E[A]={d0d3['D1']['E[A]']} "
            f"D2 top1={d0d3['D2']['P']['1'][0]} "
            f"({elapsed:.0f}s)")
        # incremental save (sandbox-lifetime resilience)
        with open(os.path.join(OUT_DIR,
                               "dspark_fullstream.json"), "w") as f:
            json.dump(results, f, indent=1, default=str)

    # aggregate D0-D3 across all prompts
    all_streams = []
    for v in results["per_prompt"].values():
        all_streams.extend(v.get("streams", []))
    if all_streams:
        results["aggregate"] = analyze_d0d3(all_streams)
        a = results["aggregate"]
        log(f"AGGREGATE D0 E[A]={a['D0']['E[A]']} "
            f"D1 E[A]={a['D1']['E[A]']} "
            f"over {a['n_records']} blocks")
        d3 = a["D3"]["budgets"]
        log("D3 E[A_tree(N)]: " + " ".join(
            f"N{n}={d3[str(n)]['E[A]']}" for n in TREE_BUDGETS))

    for h in handles:
        h.remove()
    path = os.path.join(OUT_DIR, "dspark_fullstream.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    log(f"wrote {path}")
    log(f"=== DONE in {(time.time()-t_start)/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    import torch  # noqa: F401  (module-level torch used by classes)
    sys.exit(main())
