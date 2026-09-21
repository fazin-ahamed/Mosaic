"""V4 layer implementations (Phase 1 correctness path).

Attention uses the DSA Triton kernel for decode and plain SDPA for prefill.
Router stays bf16 (doc §22 route-sensitivity). HashRouter follows checkpoint
weight shapes; MoELayer executes selected experts via a grouped-GEMM op.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import V4Config
from .dsa_layers import ExactAttentionAdapter


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Attention(nn.Module):
    """64-head attention with sliding window 128; DSA kernel at decode."""

    def __init__(self, cfg: V4Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.o_groups = cfg.o_groups
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        inner = self.n_heads * self.head_dim
        self.q_a = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.q_b = nn.Linear(cfg.q_lora_rank, inner, bias=False)
        self.kv_proj = nn.Linear(cfg.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        group_head_dim = inner // self.o_groups
        self.o_lora_a = nn.Linear(group_head_dim, self.o_groups * cfg.o_lora_rank, bias=False)
        self.o_lora_b = nn.Linear(self.o_groups * cfg.o_lora_rank, cfg.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))
        self.window = cfg.sliding_window

    def forward(self, x, kv_cache=None):
        b, t, _ = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.q_b(self.q_norm(self.q_a(x))).view(b, t, h, hd)
        kv = self.kv_norm(self.kv_proj(x)).unsqueeze(2)  # [b, t, 1, hd]

        if kv_cache is not None:
            if kv_cache[0] is not None and torch.is_tensor(kv_cache[0]) and kv_cache[0].numel() > 0:
                kv = torch.cat([kv_cache[0], kv], dim=1)
            kv_cache[0], kv_cache[1] = kv, kv

        k = kv
        v = kv
        dsa_topk = self.cfg.index_topk
        dsa_capable = (t == 1 and x.is_cuda
                       and (dsa_topk & (dsa_topk - 1)) == 0
                       and k.shape[1] >= dsa_topk)
        if dsa_capable:
            from ..kernels.dsa_attention import dsa_decode_attention

            ctx = k.shape[1]
            q_last = q[:, -1]
            k_c = k.expand(-1, -1, h, -1)
            v_c = v.expand(-1, -1, h, -1)
            out = torch.empty(b, h, hd, dtype=q.dtype, device=q.device)
            for bi in range(b):
                o, _sel = dsa_decode_attention(
                    q_last[bi].to(torch.bfloat16),
                    k_c[bi].to(torch.bfloat16),
                    v_c[bi].to(torch.bfloat16),
                    topk=dsa_topk,
                )
                out[bi] = o
            attn_out = out.reshape(b, 1, h * hd)
        else:
            scale = hd ** -0.5
            qh = q.transpose(1, 2)
            kh = k.expand(-1, -1, h, -1).permute(0, 2, 1, 3)
            vh = v.expand(-1, -1, h, -1).permute(0, 2, 1, 3)
            mask = None
            if t > self.window:
                idx = torch.arange(t, device=x.device)
                dist = idx[:, None] - idx[None, :]
                allow = (dist <= self.window) & (dist >= 0)
                mask = torch.zeros(t, t, device=x.device, dtype=qh.dtype)
                mask = mask.masked_fill(~allow, float("-inf"))
            attn_out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
            attn_out = attn_out.transpose(1, 2).reshape(b, t, h * hd)

        attn_out = attn_out.reshape(b, t, self.o_groups, -1)
        wo_a_w = self.o_lora_a.weight.view(self.o_groups, self.cfg.o_lora_rank, -1)
        attn_out = torch.einsum("btgd,grd->btgr", attn_out, wo_a_w)
        attn_out = attn_out.flatten(2)
        return self.o_lora_b(attn_out)


def _router_scores(logits: torch.Tensor, method: str) -> torch.Tensor:
    """scoring_func=sqrtsoftplus per checkpoint config."""
    if method == "sqrtsoftplus":
        s = F.softplus(logits.float())
        return s.sqrt()
    raise ValueError(f"unknown scoring_func {method}")


class Router(nn.Module):
    """Linear + sqrtsoftplus scores + noaux_tc top-k, always bf16 (doc §22)."""

    def __init__(self, cfg: V4Config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(cfg.n_routed_experts, cfg.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(cfg.n_routed_experts))
        self.cfg = cfg

    def forward(self, x):
        from ..runtime.exact import exact_target

        logits = F.linear(x.float(), self.weight.float())
        original = _router_scores(logits, self.cfg.scoring_func)
        scores = original + self.e_score_correction_bias.float()
        if exact_target():
            # Exact-30: always top num_experts_per_tok (6), no AED.
            max_k = self.cfg.num_experts_per_tok
        else:
            max_k = getattr(self, "active_k", None) or self.cfg.num_experts_per_tok
        topk_idx = torch.topk(scores, max_k, dim=-1).indices
        topk_scores = original.gather(-1, topk_idx)
        if not exact_target() and getattr(self, "aed", False):
            from ..runtime.aed import adaptive_k

            decision = adaptive_k(original, max_k=max_k)
            slot = torch.arange(max_k, device=x.device)
            keep = slot.unsqueeze(0) < decision.k.unsqueeze(-1)
            topk_scores = topk_scores * keep.to(topk_scores.dtype)
        if self.cfg.norm_topk_prob:
            topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topk_scores = topk_scores * self.cfg.routed_scaling_factor
        return topk_scores.float(), topk_idx


class HashRouter(nn.Module):
    """Hash-routed layers: no bias, real gate-projection scores, literal normalize.

    Expert ids come from ``tid2eid``; weights are the true router scores
    gathered at those ids, normalized when scoring_func != 'softmax', then
    scaled by routed_scaling_factor. Under exact target a missing table is an
    ExactContractError — no salted fallback.
    """

    def __init__(self, cfg: V4Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.weight = nn.Parameter(torch.zeros(cfg.n_routed_experts, cfg.hidden_size))
        self.tid2eid = None

    def forward(self, x, token_ids=None):
        from ..runtime.exact import ExactContractError, exact_target

        n = x.shape[0]
        if token_ids is None:
            token_ids = torch.arange(n, device=x.device)
        else:
            token_ids = token_ids.reshape(-1)
        if self.tid2eid is None:
            if exact_target():
                raise ExactContractError(f"HashRouter layer {self.layer_idx}: tid2eid required under exact target")
            salt = self.layer_idx * 2654435761 % (2**31)
            hashed = ((token_ids.int() + salt) * 2654435761) % (2**31)
            idx = (hashed % self.cfg.n_routed_experts).long().view(n, 1)
            scores = torch.ones_like(idx, dtype=torch.float32) * self.cfg.routed_scaling_factor
            return scores, idx
        table = self.tid2eid.to(device=x.device).long()
        idx = table[token_ids.long() % table.shape[0]]
        if idx.dim() == 1:
            idx = idx.view(n, 1)
        else:
            idx = idx.view(n, -1)
        logits = F.linear(x.float(), self.weight.float())
        original = _router_scores(logits, self.cfg.scoring_func)
        w = original.gather(-1, idx)
        if self.cfg.scoring_func != "softmax":
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        w = w * self.cfg.routed_scaling_factor
        return w.float(), idx


class ExpertMLP(nn.Module):
    """gate/up/down SwiGLU at moe_intermediate, FP32 with optional swiglu_limit clamp."""

    def __init__(self, hidden: int, inter: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x):
        gate = self.gate_proj(x).float()
        up = self.up_proj(x).float()
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        z = F.silu(gate) * up
        return self.down_proj(z.to(x.dtype))


class SharedExpertMLP(ExpertMLP):
    pass


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    pre_raw = mixes[..., :hc_mult]
    post_raw = mixes[..., hc_mult:2 * hc_mult]
    comb_raw = mixes[..., 2 * hc_mult:].reshape(*mixes.shape[:-1], hc_mult, hc_mult)
    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
    comb = comb_raw * hc_scale[2] + hc_base[2 * hc_mult:].reshape(hc_mult, hc_mult)
    comb = F.softmax(comb, dim=-1) + eps
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class HyperConnections(nn.Module):
    """Checkpoint-faithful HC: hc_mult residual copies with Sinkhorn mix.

    Weight names match the snapshot: hc_{attn,ffn}_{fn,base,scale}.
    """

    def __init__(self, cfg: V4Config, kind: str):
        super().__init__()
        self.mult = cfg.hc_mult
        self.iters = cfg.hc_sinkhorn_iters
        self.eps = cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.hidden_size
        self.fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.base = nn.Parameter(torch.zeros(mix_hc))
        self.scale = nn.Parameter(torch.ones(3))
        nn.init.normal_(self.fn, std=0.01)
        self.kind = kind

    def pre(self, x):
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.fn.float()) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.scale.float(), self.base.float(),
            self.mult, self.iters, self.eps,
        )
        y = (pre.unsqueeze(-1) * x.float()).sum(dim=2)
        return y.to(x.dtype), post, comb

    def post(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(2).float() + torch.einsum("bsmn,bsnd->bsmd", comb, residual.float())
        return y.to(residual.dtype)


class MoELayer(nn.Module):
    """Shared expert + gather/scatter around top-6 routed experts."""

    def __init__(self, cfg: V4Config, is_hash: bool = False):
        super().__init__()
        self.cfg = cfg
        self.is_hash = is_hash
        self.router = Router(cfg) if not is_hash else None
        self.shared_expert = SharedExpertMLP(cfg.hidden_size, cfg.moe_intermediate_size, cfg.swiglu_limit)

    def forward(self, x):
        from ..runtime.exact import exact_target

        b, t, _ = x.shape
        xf = x.view(-1, x.shape[-1])
        scores, idx = self.router(xf)  # [bt, k]
        out_shared = self.shared_expert(xf)
        routed = grouped_expert_forward(
            xf, scores, idx,
            self.experts_gate_up(), self.experts_down(),
            self.cfg.n_routed_experts,
            swiglu_limit=self.cfg.swiglu_limit,
        )
        y = out_shared + routed
        if not exact_target() and getattr(self, "use_aec", False) and getattr(self, "surrogate", None) is not None:
            scale = self.cfg.routed_scaling_factor
            kept = (scores.float().sum(dim=-1) / scale).clamp(0, 1)
            omitted = (1.0 - kept).unsqueeze(-1)
            y = y + self.surrogate(xf, omitted)
        return y.view(b, t, -1)

    def experts_gate_up(self):
        raise NotImplementedError("subclass provides stacked expert weights")


class DenseMoELayer(MoELayer):
    """All-experts-resident variant used for tests and golden parity runs."""

    def __init__(self, cfg: V4Config, resident_experts: bool = True):
        super().__init__(cfg, is_hash=False)
        e, i, h = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
        if resident_experts:
            self.w_gate = nn.Parameter(torch.randn(e, h, i) / math.sqrt(h))
            self.w_up = nn.Parameter(torch.randn(e, h, i) / math.sqrt(h))
            self.w_down = nn.Parameter(torch.randn(e, i, h) / math.sqrt(i))
        else:
            self.w_gate = self.w_up = self.w_down = None

    def experts_gate_up(self):
        return self.w_gate, self.w_up

    def experts_down(self):
        return self.w_down


def grouped_expert_forward(x, scores, idx, gate_up, down, n_experts, swiglu_limit: float = 0.0):
    """Dispatch each token to its selected experts (V4-0731 source semantics).

    FP32 SwiGLU with ``swiglu_limit`` clamps (gate max-only, up both-sided);
    router weight applied to the FP32 pre-down activation (down re-quantizes
    its input, so Q(r·x) != r·Q(x)); per-slot partial summed sequentially in
    ascending expert-id order — deterministic, no ``index_add_`` atomics.
    gate_up: tuple (w_gate [E,h,i], w_up [E,h,i]); down: w_down [E,i,h].
    """
    wg, wu = gate_up
    wd = down
    bt, h = x.shape
    k = idx.shape[-1]

    w = scores.reshape(bt, k).float()
    keep = w != 0
    if not bool(keep.any()):
        return torch.zeros(bt, h, dtype=x.dtype, device=x.device)
    partials = torch.zeros(bt, k, h, dtype=torch.float32, device=x.device)
    for slot in range(k):
        fidx = idx[:, slot].long()
        ws = w[:, slot]
        if not bool(ws.ne(0).any()):
            continue
        sel_wg = wg[fidx].float()
        sel_wu = wu[fidx].float()
        xs = x.float().unsqueeze(1)
        gate = torch.bmm(xs, sel_wg)
        up = torch.bmm(xs, sel_wu)
        if swiglu_limit > 0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        z = F.silu(gate) * up  # FP32
        z = z * ws[:, None, None]
        zd = z.to(x.dtype)
        sel_wd = wd[fidx].float()
        y = torch.bmm(zd.float(), sel_wd).squeeze(1)
        partials[:, slot] = y
    # ascending expert-id order per token, sequential FP32 accumulation
    order = idx.long().argsort(dim=1)
    out = torch.zeros(bt, h, dtype=torch.float32, device=x.device)
    for s in range(k):
        slot = order[:, s]
        out = out + partials.gather(1, slot.view(bt, 1, 1).expand(bt, 1, h)).squeeze(1)
    return out.to(x.dtype)


class HashMoELayer(DenseMoELayer):
    """MoE layer with hash routing (the num_hash_layers=3 target layers)."""

    def __init__(self, cfg: V4Config, layer_idx: int, resident_experts: bool = True):
        MoELayer.__init__(self, cfg, is_hash=True)
        e, i, h = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
        if resident_experts:
            self.w_gate = nn.Parameter(torch.randn(e, h, i) / math.sqrt(h))
            self.w_up = nn.Parameter(torch.randn(e, h, i) / math.sqrt(h))
            self.w_down = nn.Parameter(torch.randn(e, i, h) / math.sqrt(i))
        else:
            self.w_gate = self.w_up = self.w_down = None
        self.hash_router = HashRouter(cfg, layer_idx)

    def forward(self, x, token_ids=None):
        b, t, _ = x.shape
        xf = x.view(-1, x.shape[-1])
        scores, idx = self.hash_router(xf, token_ids)
        out_shared = self.shared_expert(xf)
        routed = grouped_expert_forward(
            xf, scores, idx,
            self.experts_gate_up(), self.experts_down(),
            self.cfg.n_routed_experts,
            swiglu_limit=self.cfg.swiglu_limit,
        )
        return (out_shared + routed).view(b, t, -1)


class V4Block(nn.Module):
    def __init__(self, cfg: V4Config, layer_idx: int, is_hash: bool, resident_experts: bool | None = None,
                 exact_dsa: bool = False):
        super().__init__()
        if resident_experts is None:
            resident_experts = cfg.n_routed_experts <= 8
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if exact_dsa:
            self.attn = ExactAttentionAdapter(cfg, layer_idx)
        else:
            self.attn = Attention(cfg)
        self.is_hash = is_hash
        self.hc_attn = HyperConnections(cfg, "attn")
        self.hc_ffn = HyperConnections(cfg, "ffn")
        if is_hash:
            self.moe: MoELayer = HashMoELayer(cfg, layer_idx, resident_experts=resident_experts)
        else:
            self.moe = DenseMoELayer(cfg, resident_experts=resident_experts)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, token_ids=None, kv_cache=None):
        # x: [B, S, hc_mult, D]
        residual = x
        y, post, comb = self.hc_attn.pre(x)
        y = self.attn_norm(y)
        y = self.attn(y, kv_cache)
        x = self.hc_attn.post(y, residual, post, comb)
        residual = x
        y, post, comb = self.hc_ffn.pre(x)
        y = self.ffn_norm(y)
        y = self.moe(y, token_ids) if self.is_hash else self.moe(y)
        x = self.hc_ffn.post(y, residual, post, comb)
        return x
