"""§43 rungs 2-5 — multi-layer parity, 43-layer + head golden, greedy
tokens vs the Blackwell-authoritative overlap corpus.

Modes:
  parity  — run BOTH chains (kit source Block vs exact runtime layer)
            through all 43 layers, per-layer divergence localization
            (prefill of the corpus coding prompt).
  golden  — run OUR exact chain only: 43 layers + hc_head + norm +
            lm_head, greedy continuation, compared token-by-token
            against the corpus expected ids.

Memory design (12 GB card):
  - embed + lm_head + hc_head resident (~2 GB)
  - per-layer weights moved CPU→GPU on demand (parity: both sides;
    golden: ours only), freed after each layer
  - attention STATE (window ring + compressor/indexer state) persists
    per layer in lightweight CPU-side holders, restored on re-entry
  - routed experts via the shared lazy dequant cache (page-cache backed)
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, ".mosaicvm/source")

import mosaicvm.model.dsa as mdsa  # noqa: E402

fht = types.ModuleType("fast_hadamard_transform")
fht.hadamard_transform = lambda x, scale=None: mdsa.hadamard(x)
sys.modules["fast_hadamard_transform"] = fht

import inference_kernel  # noqa: E402
sys.modules["kernel"] = inference_kernel

DEV = "cuda"
MAXSEQ = 512
HIDDEN = 4096
INTER = 2048
N_EXPERTS = 256
TOPK = 6
N_LAYERS = 43
HC = 4
VOCAB = 129280

torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(DEV)

import inference_model as SM  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
_E2M1_MID = (_E2M1[1:] + _E2M1[:-1]) / 2


def _act_quant(x, block_size=128, scale_fmt=None,
               scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    z = x.contiguous().float()
    xb = z.view(*z.shape[:-1], N // block_size, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0))) \
        if scale_fmt is not None else amax / 448.0
    qf = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        return x.copy_((qf.float() * s).to(x.dtype).reshape(*x.shape))
    return qf.reshape(*z.shape), s.reshape(*z.shape[:-1], N // block_size)


def _fp4_act_quant(x, block_size=32, inplace=False):
    N = x.size(-1)
    z = x.contiguous().float()
    xb = z.view(*z.shape[:-1], N // block_size, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(6 * 2 ** -126)
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    q = (xb / s).clamp(-6.0, 6.0)
    idx = torch.bucketize(q.abs(), _E2M1_MID.to(q.device))
    dq = q.sign() * _E2M1.to(q.device)[idx] * s
    if inplace:
        return x.copy_(dq.to(x.dtype).reshape(*x.shape))
    return dq.to(torch.float8_e4m3fn).reshape(*z.shape), s


def _sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, s, h, d = q.size()
    valid = topk_idxs >= 0
    idxc = topk_idxs.clamp(min=0)
    g = kv.unsqueeze(1).expand(b, s, *kv.shape[1:]) \
        .gather(2, idxc.unsqueeze(-1).expand(-1, -1, -1, d))
    scores = torch.einsum("bshd,bstd->bsht", q.float(), g.float()) * softmax_scale
    scores = torch.where(valid.unsqueeze(2), scores,
                         torch.full_like(scores, float("-inf")))
    m = scores.max(-1, keepdim=True).values
    p = torch.exp(scores - m)
    p_bf = p.to(torch.bfloat16).float()
    denom = p_bf.sum(-1) + torch.exp(
        attn_sink.view(1, 1, h).float() - m.squeeze(-1))
    o = torch.einsum("bsht,bstd->bshd", p_bf, g.float()) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4,
                       sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre_ = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].view(*mixes.shape[:-1], hc, hc) \
        * hc_scale[2] + hc_base[2 * hc:].view(hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre_, post, comb


def _fp4_gemm(a, a_s, w, w_s, scale_dtype=None):
    M, K = a.shape
    n = w.shape[0]
    A = a.float()
    if a_s is not None and a_s.dim() == 2:
        A = A * a_s.float().repeat_interleave(
            K // a_s.shape[-1], dim=-1).reshape(A.shape)
    from mosaicvm.runtime.fp8 import dequant_fp4
    W = dequant_fp4(w, w_s, n, K).float()
    return (A @ W.t()).to(torch.bfloat16)


SM.act_quant = _act_quant
SM.fp4_act_quant = _fp4_act_quant
SM.sparse_attn = _sparse_attn
SM.hc_split_sinkhorn = _hc_split_sinkhorn
SM.fp4_gemm = _fp4_gemm
SM.default_dtype = torch.bfloat16
SM.scale_fmt = "ue8m0"
SM.scale_dtype = torch.float8_e8m0fnu

_orig_linear = SM.linear


def _linear_nofp8act(x, weight, bias=None):
    if weight.dtype == torch.float4_e2m1fn_x2:
        return _fp4_gemm(x.float().to(torch.bfloat16), None, weight,
                         weight.scale, None)
    return _orig_linear(x, weight, bias)


SM.linear = _linear_nofp8act

from safetensors import safe_open  # noqa: E402

ARCH = safe_open(".mosaicvm/core/core.safetensors", framework="pt")
SNAP = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731"
    "/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
IDX = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]


def snap_get(name):
    with safe_open(str(SNAP / IDX[name]), framework="pt") as f:
        return f.get_tensor(name)


def arch_get(name):
    return ARCH.get_tensor(name)


from mosaicvm.runtime.fp8 import dequant_fp4  # noqa: E402
from mosaicvm.kernels.mxint import (  # noqa: E402
    fused_moe_mxint, prepack_expert_bank_mxint_packed4)
from mosaicvm.model.config import V4Config  # noqa: E402
from mosaicvm.model.dsa_layers import (  # noqa: E402
    ExactAttention, cast_exact_attention)
from mosaicvm.model.layers import Router, HashRouter  # noqa: E402
import mosaicvm.model.dsa_layers as _DL  # noqa: E402

_DL.USE_FUSED_SPARSE = False
cfg = V4Config()

_EXPERT_CACHE: dict = {}


def _get_expert_weights(layer, e):
    key = (layer, e)
    if key not in _EXPERT_CACHE:
        pre = f"layers.{layer}.ffn.experts.{e}."
        _EXPERT_CACHE[key] = (
            dequant_fp4(snap_get(pre + "w1.weight"),
                        snap_get(pre + "w1.scale"), INTER, HIDDEN),
            dequant_fp4(snap_get(pre + "w3.weight"),
                        snap_get(pre + "w3.scale"), INTER, HIDDEN),
            dequant_fp4(snap_get(pre + "w2.weight"),
                        snap_get(pre + "w2.scale"), HIDDEN, INTER))
    return _EXPERT_CACHE[key]


class LazyExpertLinear(torch.nn.Module):
    def __init__(self, layer, e, slot):
        super().__init__()
        self.layer, self.e, self.slot = layer, e, slot

    def forward(self, x):
        w = _get_expert_weights(self.layer, self.e)[self.slot]
        return F.linear(x, w.to(x.device))


def dq_fp8_arch(layer, proj):
    w = arch_get(f"layers.{layer}.attn.{proj}.weight")
    s = arch_get(f"layers.{layer}.attn.{proj}.scale")
    rows, cols = w.shape
    bs = s.shape
    return (w.float() * s.float().repeat_interleave(rows // bs[0], 0)
            .repeat_interleave(cols // bs[1], 1)).to(torch.bfloat16)


def dq_fp8_shared(layer, which):
    w = arch_get(f"layers.{layer}.ffn.shared_experts.{which}.weight")
    s = arch_get(f"layers.{layer}.ffn.shared_experts.{which}.scale")
    rows, cols = w.shape
    bs = s.shape
    return (w.float() * s.float().repeat_interleave(rows // bs[0], 0)
            .repeat_interleave(cols // bs[1], 1)).to(torch.bfloat16)


src_cfg = json.load(open(".mosaicvm/source/inference_config.json"))


def build_args():
    args = SM.ModelArgs()
    args.__dict__.update({
        "dim": src_cfg["dim"], "n_heads": src_cfg["n_heads"],
        "head_dim": src_cfg["head_dim"],
        "rope_head_dim": src_cfg["rope_head_dim"],
        "q_lora_rank": src_cfg["q_lora_rank"],
        "o_groups": src_cfg["o_groups"], "o_lora_rank": src_cfg["o_lora_rank"],
        "window_size": src_cfg["window_size"],
        "original_seq_len": src_cfg["original_seq_len"],
        "rope_theta": src_cfg["rope_theta"],
        "rope_factor": src_cfg["rope_factor"],
        "beta_fast": src_cfg["beta_fast"], "beta_slow": src_cfg["beta_slow"],
        "index_n_heads": src_cfg["index_n_heads"],
        "index_head_dim": src_cfg["index_head_dim"],
        "index_topk": src_cfg["index_topk"], "norm_eps": 1e-6,
        "max_batch_size": 1, "max_seq_len": MAXSEQ,
        "compress_rope_theta": src_cfg["compress_rope_theta"],
        "compress_ratios": src_cfg["compress_ratios"],
        "n_routed_experts": N_EXPERTS, "n_activated_experts": TOPK,
        "moe_inter_dim": INTER, "score_func": "sqrtsoftplus",
        "route_scale": 1.5, "n_shared_experts": 1, "swiglu_limit": 10.0,
        "expert_dtype": "fp4", "vocab_size": VOCAB, "n_hash_layers": 3,
    })
    return args


# ---------------- per-layer state holders (persist across tokens) ------------
class LayerState:
    """Persistent DSA state for one layer (survives weight swapping)."""

    def __init__(self):
        self.kv_cache = None       # [window, 512] bf16
        self.comp_kv = None        # compressor kv_state
        self.comp_score = None     # compressor score_state
        self.ix_kv = None          # indexer compressor kv_state
        self.ix_score = None       # indexer compressor score_state

    def save(self, attn: ExactAttention):
        self.kv_cache = attn.kv_cache.clone()
        if attn.compressor is not None:
            self.comp_kv = attn.compressor.kv_state.clone()
            self.comp_score = attn.compressor.score_state.clone()
        if attn.indexer is not None:
            self.ix_kv = attn.indexer.compressor.kv_state.clone()
            self.ix_score = attn.indexer.compressor.score_state.clone()

    def restore(self, attn: ExactAttention):
        attn.kv_cache.copy_(self.kv_cache)
        if attn.compressor is not None and self.comp_kv is not None:
            attn.compressor.kv_state.copy_(self.comp_kv)
            attn.compressor.score_state.copy_(self.comp_score)
        if attn.indexer is not None and self.ix_kv is not None:
            attn.indexer.compressor.kv_state.copy_(self.ix_kv)
            attn.indexer.compressor.score_state.copy_(self.ix_score)


STATES = [LayerState() for _ in range(N_LAYERS)]


def build_ours_layer(layer):
    attn = ExactAttention(cfg, layer_id=layer, max_seq=MAXSEQ, device=DEV)
    cast_exact_attention(attn, DEV)
    with torch.no_grad():
        for s_nm, o_nm in [("wq_a", "q_a"), ("wq_b", "q_b"),
                           ("wkv", "kv_proj"), ("wo_a", "wo_a"),
                           ("wo_b", "wo_b")]:
            getattr(attn, o_nm).weight.copy_(dq_fp8_arch(layer, s_nm))
        for nm in ("q_norm", "kv_norm"):
            getattr(attn, nm).weight.copy_(
                arch_get(f"layers.{layer}.attn.{nm}.weight").float())
        attn.attn_sink.data.copy_(
            arch_get(f"layers.{layer}.attn.attn_sink").float())

        def lc(scomp, pre):
            scomp.ape.copy_(arch_get(f"{pre}.ape").float())
            for nm in ("wkv", "wgate"):
                getattr(scomp, nm).weight.copy_(
                    arch_get(f"{pre}.{nm}.weight").float())
            scomp.norm.weight.copy_(
                arch_get(f"{pre}.norm.weight").float())

        if attn.compressor is not None:
            lc(attn.compressor, f"layers.{layer}.attn.compressor")
        if attn.indexer is not None:
            attn.indexer.wq_b.weight.copy_(
                dq_fp8_arch(layer, "indexer.wq_b"))
            attn.indexer.weights_proj.weight.copy_(arch_get(
                f"layers.{layer}.attn.indexer.weights_proj.weight"))
            lc(attn.indexer.compressor,
               f"layers.{layer}.attn.indexer.compressor")

    router = HashRouter(cfg, layer) if layer < 3 else Router(cfg)
    with torch.no_grad():
        router.weight.copy_(
            arch_get(f"layers.{layer}.ffn.gate.weight").float())
        if layer < 3:
            router.tid2eid = arch_get(
                f"layers.{layer}.ffn.gate.tid2eid").long().to(DEV)
        else:
            router.e_score_correction_bias.data = (
                arch_get(f"layers.{layer}.ffn.gate.bias").float().to(DEV))
    return attn, router


def ours_layer_forward(layer, x_hc, start_pos, input_ids, attn, router,
                       shared_w, rec=None):
    """rec: optional list — appends (layer, ids, weights) for the last
    token row (source-corpus comparison mode)."""
    hc_attn_fn = arch_get(f"layers.{layer}.hc_attn_fn").float().to(DEV)
    hc_attn_base = arch_get(f"layers.{layer}.hc_attn_base").float().to(DEV)
    hc_attn_scale = arch_get(f"layers.{layer}.hc_attn_scale").float().to(DEV)
    hc_ffn_fn = arch_get(f"layers.{layer}.hc_ffn_fn").float().to(DEV)
    hc_ffn_base = arch_get(f"layers.{layer}.hc_ffn_base").float().to(DEV)
    hc_ffn_scale = arch_get(f"layers.{layer}.hc_ffn_scale").float().to(DEV)
    attn_norm_w = arch_get(f"layers.{layer}.attn_norm.weight").float().to(DEV)
    ffn_norm_w = arch_get(f"layers.{layer}.ffn_norm.weight").float().to(DEV)

    def hc_pre(x, fn, sc, base):
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        mixes = F.linear(xf, fn) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(mixes, sc, base, 4, 20, 1e-6)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def rms(x, w):
        xf = x.float()
        return (w * xf * torch.rsqrt(
            xf.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)

    b, s = x_hc.shape[:2]
    residual = x_hc
    x, post, comb = hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    x = rms(x, attn_norm_w)
    with torch.no_grad():
        a = attn(x.reshape(-1, HIDDEN), start_pos).reshape(b, s, HIDDEN)
    x = hc_post(a, residual, post, comb)

    residual = x
    x, post, comb = hc_pre(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
    x = rms(x, ffn_norm_w)
    xf = x.reshape(-1, HIDDEN)
    if layer < 3:
        scores, idx = router(xf, input_ids.flatten())
    else:
        scores, idx = router(xf)
    if rec is not None:
        rec.append((layer, idx[-1].tolist(),
                    scores[-1].float().tolist()))

    # ---- routed experts: PRODUCTION MXIE class (FP8-act x FP4-weight,
    # exact integer arithmetic via the frozen MXINT kernels; source-exact
    # swiglu + route-weight placement + dynamic w2 requant in the
    # epilogue). Replaces the boundary-free dequant-bf16 reference path.
    uniq = idx.unique().tolist()
    remap = {e: j for j, e in enumerate(uniq)}
    idx_local = torch.zeros_like(idx)
    for e, j in remap.items():
        idx_local[idx == e] = j
    packed_g = torch.stack([
        snap_get(f"layers.{layer}.ffn.experts.{e}.w1.weight")
        for e in uniq]).to(DEV)
    scale_g = torch.stack([
        snap_get(f"layers.{layer}.ffn.experts.{e}.w1.scale")
        for e in uniq]).to(DEV)
    packed_u = torch.stack([
        snap_get(f"layers.{layer}.ffn.experts.{e}.w3.weight")
        for e in uniq]).to(DEV)
    scale_u = torch.stack([
        snap_get(f"layers.{layer}.ffn.experts.{e}.w3.scale")
        for e in uniq]).to(DEV)
    packed_d = torch.stack([
        snap_get(f"layers.{layer}.ffn.experts.{e}.w2.weight")
        for e in uniq]).to(DEV)
    scale_d = torch.stack([
        snap_get(f"layers.{layer}.ffn.experts.{e}.w2.scale")
        for e in uniq]).to(DEV)
    bank = prepack_expert_bank_mxint_packed4(
        packed_g, scale_g, packed_u, scale_u, packed_d, scale_d)
    routed = fused_moe_mxint(
        xf, idx_local, scores, bank=bank, weight_repr="packed4",
        swiglu_limit=10.0)
    del bank, packed_g, scale_g, packed_u, scale_u, packed_d, scale_d
    from mosaicvm.runtime.memory_budget import budget as _mb
    _mb().reset()  # simulated accounting: bank charges never auto-release
    y = routed.float()
    # shared expert: SOURCE FP8 CLASS — dynamic E4M3 act quant per-128
    # (pow2 E8M0 scale, act_quant_ref = the source act_quant) then GEMM
    # against the block-dequantized FP8 weights. Replaces the boundary-
    # free dequant-bf16 path (the last non-source-class primitive).
    from mosaicvm.kernels.fp4_source import act_quant_ref
    sh_w1, sh_w3, sh_w2 = shared_w

    def _q(x):
        xq, sa = act_quant_ref(x, 128)
        # dequantized activation = the source GEMM's effective input
        m, k = x.shape
        return (xq.float().view(m, k // 128, 128)
                * sa.unsqueeze(-1)).reshape(m, k)

    xq = _q(xf)
    g = (xq @ sh_w1.float().t()).clamp(max=10.0)
    u = (xq @ sh_w3.float().t()).clamp(-10.0, 10.0)
    z = (F.silu(g) * u).to(torch.bfloat16)
    zq = _q(z)
    y += zq @ sh_w2.float().t()
    y = y.to(xf.dtype).reshape(b, s, HIDDEN)
    x = hc_post(y, residual, post, comb)
    return x


def build_src_layer(layer):
    args = build_args()
    blk = SM.Block(layer, args).to(DEV)
    with torch.no_grad():
        for s_nm in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            getattr(blk.attn, s_nm).weight.copy_(dq_fp8_arch(layer, s_nm))
        for nm in ("q_norm", "kv_norm"):
            getattr(blk.attn, nm).weight.copy_(
                arch_get(f"layers.{layer}.attn.{nm}.weight").float())
        blk.attn.attn_sink.copy_(
            arch_get(f"layers.{layer}.attn.attn_sink").float())

        def lc2(scomp, pre):
            scomp.ape.copy_(arch_get(f"{pre}.ape").float())
            for nm in ("wkv", "wgate"):
                getattr(scomp, nm).weight.copy_(
                    arch_get(f"{pre}.{nm}.weight").float())
            scomp.norm.weight.copy_(
                arch_get(f"{pre}.norm.weight").float())

        if getattr(blk.attn, "compressor", None) is not None:
            lc2(blk.attn.compressor, f"layers.{layer}.attn.compressor")
        if getattr(blk.attn, "indexer", None) is not None:
            ix = blk.attn.indexer
            ix.wq_b.weight.copy_(dq_fp8_arch(layer, "indexer.wq_b"))
            ix.weights_proj.weight.copy_(arch_get(
                f"layers.{layer}.attn.indexer.weights_proj.weight"))
            lc2(ix.compressor, f"layers.{layer}.attn.indexer.compressor")
        blk.attn_norm.weight.copy_(
            arch_get(f"layers.{layer}.attn_norm.weight").float())
        blk.ffn_norm.weight.copy_(
            arch_get(f"layers.{layer}.ffn_norm.weight").float())
        for nm in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                   "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            getattr(blk, nm).copy_(arch_get(f"layers.{layer}.{nm}").float())
        blk.ffn.gate.weight.copy_(
            arch_get(f"layers.{layer}.ffn.gate.weight").float())
        if blk.ffn.gate.hash:
            blk.ffn.gate.tid2eid.copy_(
                arch_get(f"layers.{layer}.ffn.gate.tid2eid").long())
        else:
            blk.ffn.gate.bias.copy_(
                arch_get(f"layers.{layer}.ffn.gate.bias").float())
        for nm in ("w1", "w3", "w2"):
            lin = getattr(blk.ffn.shared_experts, nm)
            lin.weight.copy_(dq_fp8_shared(layer, nm))
            if hasattr(lin, "scale"):
                delattr(lin, "scale")
        for i in range(N_EXPERTS):
            ex = blk.ffn.experts[i]
            for slot, nm in enumerate(("w1", "w3", "w2")):
                setattr(ex, nm, LazyExpertLinear(layer, i, slot))
    return blk


def head_forward(h):
    """hc_head + norm + lm_head -> logits (greedy-ready)."""
    hc_head_fn = arch_get("hc_head_fn").float().to(DEV)
    hc_head_base = arch_get("hc_head_base").float().to(DEV)
    hc_head_scale = arch_get("hc_head_scale").float().to(DEV)
    norm_w = arch_get("norm.weight").float().to(DEV)
    head_w = arch_get("head.weight").float().to(DEV)
    shape, dtype = h.size(), h.dtype
    xf = h.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
    mixes = F.linear(xf, hc_head_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_head_scale + hc_head_base) + 1e-6
    y = torch.sum(pre.unsqueeze(-1) * h.view(shape), dim=2).to(dtype)
    yf = y.float()
    yn = (norm_w * yf * torch.rsqrt(
        yf.square().mean(-1, keepdim=True) + 1e-6)).to(dtype)
    return F.linear(yn.float(), head_w)


def run_one(prompt_entry, n_gen, rec=None, hid=None, top5=None):
    """Returns (generated_ids, expected_ids, exact_prefix_len).
    rec: optional list — per-step route records appended:
    rec[step] = [(layer, ids, weights) x 43].
    hid: optional list — per decode step dk: [(layer, h_mean_fp16)] x 43
    (hc-mean of the layer output at the last position; corpus-aligned).
    top5: optional list — per step (incl. prefill): (ids, vals) of my
    logits at the produced token's position."""
    expected = prompt_entry["expected_generated_ids"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SNAP), trust_remote_code=True)
    input_ids = tok(prompt_entry["prompt"],
                    return_tensors="pt").input_ids.to(DEV)
    S = input_ids.shape[1]
    embed_w = arch_get("embed.weight").to(DEV)
    h_ours = F.embedding(input_ids, embed_w).unsqueeze(2) \
        .repeat(1, 1, HC, 1)
    for st in STATES:
        st.kv_cache = None; st.comp_kv = None; st.comp_score = None
        st.ix_kv = None; st.ix_score = None
    generated = []
    if hid is not None:
        hid.append([])  # decode step 0 slot
    import time
    for step in range(n_gen + 1):
        t0 = time.time()
        start_pos = 0 if step == 0 else S + step - 1
        ids_step = input_ids if step == 0 else generated[-1].reshape(1, 1)
        step_rec = [] if rec is not None else None
        for L in range(N_LAYERS):
            shared_w = tuple(dq_fp8_shared(L, nm).to(DEV)
                             for nm in ("w1", "w3", "w2"))
            attn, router = build_ours_layer(L)
            if step > 0 and STATES[L].kv_cache is not None:
                STATES[L].restore(attn)
            with torch.no_grad():
                h_new = ours_layer_forward(L, h_ours, start_pos, ids_step,
                                           attn, router, shared_w,
                                           rec=step_rec)
            STATES[L].save(attn)
            h_ours = h_new
            if hid is not None and step > 0:
                hid[-1].append(
                    (L, h_ours[0, -1].float().mean(0).half().cpu()))
            del attn, router, shared_w
            _EXPERT_CACHE.clear()
            torch.cuda.empty_cache()
        if rec is not None:
            rec.append(step_rec)
        if hid is not None and step > 0:
            hid.append([])
        logits = head_forward(h_ours)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        generated.append(nxt)
        if top5 is not None:
            tv, ti = logits[0, -1].float().topk(5)
            top5.append((ti.tolist(), [round(v, 4) for v in tv.tolist()]))
        if step == 0:
            print(f"  prefill ({time.time()-t0:.0f}s) tok0 "
                  f"{nxt.item()} vs {expected[0]}")
        else:
            print(f"  tok {step}: {nxt.item()} vs "
                  f"{expected[step] if step < len(expected) else '?'} "
                  f"({time.time()-t0:.0f}s)")
        if step < n_gen:
            h_ours = F.embedding(nxt, embed_w).unsqueeze(2) \
                .repeat(1, 1, HC, 1)
    gen_ids = [t.item() for t in generated[:n_gen + 1]]
    exp_ids = expected[:n_gen + 1]
    prefix = 0
    for a, b in zip(gen_ids, exp_ids):
        if a != b:
            break
        prefix += 1
    return gen_ids, exp_ids, prefix


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "golden"
    corpus = json.loads(Path(
        "artifacts/routes_overlap_corpus_3060.json").read_text())

    if mode == "srcchain":
        # kit-source full 43-layer chain (local) — three-way comparison:
        # my runtime vs corpus (Blackwell transformers) vs kit source
        from transformers import AutoTokenizer as _AT
        tok = _AT.from_pretrained(str(SNAP), trust_remote_code=True)
        embed_w = arch_get("embed.weight").to(DEV)
        cats = sys.argv[2].split(",") if len(sys.argv) > 2 else \
            ["math_reasoning", "long_generation"]
        for pe in corpus["prompts"]:
            if pe["category"] not in cats:
                continue
            input_ids = tok(pe["prompt"],
                            return_tensors="pt").input_ids.to(DEV)
            S = input_ids.shape[1]
            h = F.embedding(input_ids, embed_w).unsqueeze(2) \
                .repeat(1, 1, HC, 1)
            for L in range(N_LAYERS):
                blk = build_src_layer(L)
                with torch.no_grad():
                    h = blk(h, 0, input_ids)
                del blk
                _EXPERT_CACHE.clear()
                torch.cuda.empty_cache()
            logits = head_forward(h)
            g0 = logits[0, -1].argmax().item()
            topv, topi = logits[0, -1].float().topk(5)
            print(f"[{pe['category']}] KIT-SRC greedy0={g0} "
                  f"corpus={pe['expected_generated_ids'][0]}")
            top5 = [(int(i), round(float(v), 4))
                    for i, v in zip(topi, topv)]
            print(f"  top5: {top5}")
        return

    if mode == "corpus":
        n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        results = []
        for pe in corpus["prompts"]:
            cat = pe["category"]
            print(f"\n=== [{cat}] ===")
            gen, exp, prefix = run_one(pe, n_gen)
            results.append({
                "category": cat,
                "generated_ids": gen,
                "expected_ids": exp,
                "exact_prefix": prefix,
                "token_exact_full": gen == exp,
            })
            print(f"[{cat}] exact prefix: {prefix}/{len(exp)}")
        out = {
            "classification": "RTX3060_DISPLAY_ATTACHED + "
                              "EXACT_CONSTRAINED_RUNTIME (sequential "
                              "reference path)",
            "numeric_class": "boundary-free dequant-bf16 experts "
                             "(NUMERIC_PARITY vs the Blackwell fp8-act x "
                             "fp4-weight corpus reference)",
            "archive": "a9089f25fa85bac9",
            "n_tokens": n_gen + 1,
            "results": results,
            "summary": {
                "mean_exact_prefix": sum(r["exact_prefix"]
                                         for r in results) / len(results),
                "full_exact_prompts": sum(
                    1 for r in results if r["token_exact_full"]),
            },
        }
        Path("artifacts/routes_overlap_3060_result.json").write_text(
            json.dumps(out, indent=1))
        print("\n=== CORPUS RUN COMPLETE ===")
        print(f"mean exact prefix: "
              f"{out['summary']['mean_exact_prefix']:.2f}")
        print(f"full-exact prompts: {out['summary']['full_exact_prompts']}"
              f"/{len(results)}")
        return

    if mode == "localize":
        # §41 FIRST-DIVERGENCE LOCALIZATION: per-layer hidden drift
        # between my runtime and the source-class corpus at every
        # matched decode step up to each prompt's first divergence.
        from collections import defaultdict
        prev = json.loads(Path(
            "artifacts/source_corpus_compare.json").read_text())
        first_div = {}
        for r in prev["results"]:
            d = r["divergences"]
            first_div[r["category"]] = d[0]["token"] if d else None
        out = []
        for pe in corpus["prompts"]:
            cat = pe["category"]
            fd = first_div.get(cat)
            if fd is None:
                print(f"[{cat}] fully exact — nothing to localize")
                continue
            f = Path(f"artifacts/source_corpus/{cat}.pt")
            if not f.exists():
                print(f"[{cat}] MISSING corpus file")
                continue
            scr = torch.load(f, map_location="cpu", weights_only=False)
            ch = {(h["step"], h["layer"]): h["h_mean"].float()
                  for h in scr["hiddens"]}
            print(f"\n=== [{cat}] localizing first divergence at "
                  f"token {fd} ===")
            rec, hid, top5 = [], [], []
            pe = dict(pe)
            pe["expected_generated_ids"] = scr["generated"]
            gen, _, _ = run_one(pe, fd, rec=rec, hid=hid, top5=top5)
            # decode step dk produces gen[dk+1]; matched producing
            # forwards are dk = 0..fd-1 (tokens gen[1..fd])
            steps = []
            for dk in range(0, fd):
                lay = []
                for (L, hm) in hid[dk]:
                    c = ch.get((dk, L))
                    if c is None:
                        continue
                    a = hm.float()
                    na, nb = a.norm(), c.norm()
                    cos = torch.nn.functional.cosine_similarity(
                        a, c, dim=0).item()
                    rel = ((a - c).norm() / nb.clamp_min(1e-6)).item()
                    lay.append({"layer": L, "cos": round(cos, 6),
                                "rel_l2": round(rel, 6),
                                "max_abs": round(
                                    (a - c).abs().max().item(), 6)})
                steps.append({"decode_step": dk,
                              "token_produced": gen[dk + 1],
                              "layers": lay})
            # logit near-tie evidence at the divergence position
            m = scr["top5_per_token"][fd]
            div_logit = {
                "token": fd, "ours": gen[fd], "corpus": scr["generated"][fd],
                "ours_top5_ids": top5[fd][0] if fd < len(top5) else None,
                "ours_top5_vals": top5[fd][1] if fd < len(top5) else None,
                "corpus_top5_ids": m["ids"],
                "corpus_top5_vals": [round(v, 4) for v in m["vals"]],
                "corpus_margin_top12": round(m["vals"][0] - m["vals"][1], 4),
            }
            out.append({
                "category": cat, "first_divergence_token": fd,
                "n_matched_decode_steps": fd,
                "steps": steps, "divergence_logits": div_logit,
            })
            # summary: worst layers at the last matched step
            last = steps[-1]["layers"]
            worst = sorted(last, key=lambda x: x["rel_l2"])[-5:]
            print(f"  matched decode steps: {fd}; "
                  f"drift at last matched step (top-5 worst rel_l2): "
                  f"{[(w['layer'], w['rel_l2']) for w in worst]}")
            first = steps[0]["layers"]
            w0 = sorted(first, key=lambda x: x["rel_l2"])[-3:]
            print(f"  drift at first decode step (worst 3): "
                  f"{[(w['layer'], w['rel_l2']) for w in w0]}")
        Path("artifacts/first_divergence_localization.json").write_text(
            json.dumps({
                "classification": "SECTION41_FIRST_DIVERGENCE_LOCALIZATION",
                "comparison": "RTX3060 source-class runtime vs "
                              "kit-code source corpus (Blackwell)",
                "prompts": out,
            }, indent=1))
        print("\n=== LOCALIZATION COMPLETE ===")
        return

    if mode == "certify":
        # G2 §38-unit certification, MosaicVM side: full 32-token runs
        # with complete recording (tokens, per-token top-5, per-step
        # routes, per-step per-layer hiddens). Records are saved raw;
        # the frozen spec is applied OFFLINE by g2_certify_eval.py.
        ref_dir = Path(sys.argv[2] if len(sys.argv) > 2
                       else "artifacts/g2_cert_ref_3060")
        n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 31
        out_dir = Path("artifacts/g2_cert_mosaicvm")
        out_dir.mkdir(exist_ok=True)
        for f in sorted(ref_dir.glob("*.pt")):
            ref = torch.load(f, map_location="cpu", weights_only=False)
            cat = ref["category"]
            print(f"\n=== [{cat}] (G2 certification, MosaicVM side) ===")
            pe = None
            for p in corpus["prompts"]:
                if p["category"] == cat:
                    pe = dict(p)
                    break
            if pe is None or pe["prompt"] != ref["prompt"]:
                # fall back to constructing from ref (same tokenizer)
                pe = {"category": cat, "prompt": ref["prompt"],
                      "expected_generated_ids": ref["generated"]}
            pe["expected_generated_ids"] = ref["generated"]
            rec, hid, top5 = [], [], []
            gen, _, _ = run_one(pe, n_gen, rec=rec, hid=hid, top5=top5)
            torch.save({
                "category": cat, "prompt": ref["prompt"],
                "generated": gen, "top5": top5,
                "routes": rec, "hiddens": hid,
            }, out_dir / f"{cat}.pt")
            print(f"[{cat}] recorded {len(gen)} tokens "
                  f"({len(hid)} hidden steps)")
        print("\n=== CERTIFY RECORDING COMPLETE ===")
        return

    if mode == "srcorpus":
        # G1 gate: compare against the regenerated SOURCE-CLASS corpus
        # (kit code + torch kernel shims, Blackwell). Tokens + Top-6
        # expert IDs + route weights per layer per generated token.
        n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        src_dir = Path("artifacts/source_corpus")
        cats = (sys.argv[3].split(",") if len(sys.argv) > 3
                else [p["category"] for p in corpus["prompts"]])
        results = []
        for pe in corpus["prompts"]:
            cat = pe["category"]
            if cat not in cats:
                continue
            f = src_dir / f"{cat}.pt"
            if not f.exists():
                print(f"[{cat}] MISSING corpus file {f}")
                continue
            scr = torch.load(f, map_location="cpu",
                             weights_only=False)
            exp = scr["generated"]
            pe = dict(pe)
            pe["expected_generated_ids"] = exp  # print vs SOURCE corpus
            print(f"\n=== [{cat}] (source-class corpus) ===")
            rec = []
            gen, _, _ = run_one(pe, min(n_gen, len(exp) - 1), rec=rec)

            # align corpus gate records: token k <- prefill last row if
            # k==0 else decode step k-1 (the forward that produced gen[k])
            def corpus_routes(k):
                if k == 0:
                    pref = [g for g in scr["gates"]
                            if g["phase"] == "prefill"]
                else:
                    pref = [g for g in scr["gates"]
                            if g["phase"] == "decode"
                            and g["step"] == k - 1]
                return {g["layer"]: (g["idx"][-1], g["w"][-1])
                        for g in pref}

            tok_match, id_match, id_tot = 0, 0, 0
            set_match = 0
            layer_detail = []
            w_diffs = []
            div_details = []
            for k in range(len(gen)):
                if gen[k] == exp[k]:
                    tok_match += 1
                else:
                    m = scr["top5_per_token"][k]
                    margin = (m["vals"][0] - m["vals"][1]
                              if len(m["vals"]) > 1 else None)
                    div_details.append({
                        "token": k, "ours": gen[k], "corpus": exp[k],
                        "corpus_top5_ids": m["ids"],
                        "corpus_top5_vals": [
                            round(v, 4) for v in m["vals"]],
                        "corpus_margin_top12": margin,
                    })
                if k < len(rec):
                    cres = corpus_routes(k)
                    for (L, ids, ws) in rec[k]:
                        id_tot += 1
                        c_ids, c_ws = cres.get(L, ([], []))
                        if ids == c_ids:
                            id_match += 1
                            set_match += 1
                        elif set(ids) == set(c_ids):
                            set_match += 1
                        if c_ws:
                            w_diffs.append(max(
                                abs(a - b) for a, b in zip(ws, c_ws)))
                        if ids != c_ids and k == 0:
                            layer_detail.append({
                                "token": k, "layer": L,
                                "ours_ids": ids, "corpus_ids": c_ids,
                                "ours_w": [round(x, 4) for x in ws],
                                "corpus_w": [round(x, 4) for x in c_ws],
                            })
            results.append({
                "category": cat,
                "generated_ids": gen,
                "source_corpus_ids": exp,
                "token_exact": tok_match,
                "n_tokens": len(gen),
                "token_exact_full": gen == exp[:len(gen)],
                "route_ids_exact": id_match,
                "route_ids_set_match": set_match,
                "route_ids_total": id_tot,
                "route_weight_max_abs_diff": (
                    round(max(w_diffs), 6) if w_diffs else None),
                "route_weight_mean_abs_diff": (
                    round(sum(w_diffs) / len(w_diffs), 8)
                    if w_diffs else None),
                "divergences": div_details,
                "token0_layer_route_divergences": layer_detail,
            })
            print(f"[{cat}] tokens {tok_match}/{len(gen)} "
                  f"routes {id_match}/{id_tot} "
                  f"(set {set_match}/{id_tot}) "
                  f"wmax={max(w_diffs) if w_diffs else 0:.4f}")
        out = {
            "classification": "G1 SOURCE-CLASS CORPUS COMPARISON — "
                              "RTX3060 source-class runtime vs kit-code "
              "torch-shim corpus (Blackwell)",
            "archive": "a9089f25fa85bac9",
            "n_tokens": n_gen + 1,
            "results": results,
            "summary": {
                "token0_exact": sum(
                    1 for r in results
                    if r["generated_ids"][0]
                    == r["source_corpus_ids"][0]),
                "full_exact_prompts": sum(
                    1 for r in results if r["token_exact_full"]),
                "route_ids_exact_total": sum(
                    r["route_ids_exact"] for r in results),
                "route_ids_total": sum(
                    r["route_ids_total"] for r in results),
            },
        }
        Path("artifacts/source_corpus_compare.json").write_text(
            json.dumps(out, indent=1))
        print("\n=== SOURCE-CORPUS COMPARISON COMPLETE ===")
        for k, v in out["summary"].items():
            print(f"  {k}: {v}")
        return

    prompt = corpus["prompts"][0]
    expected = prompt["expected_generated_ids"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SNAP), trust_remote_code=True)
    input_ids = tok(prompt["prompt"], return_tensors="pt").input_ids.to(DEV)
    S = input_ids.shape[1]
    print(f"mode={mode} prompt='{prompt['prompt'][:40]}' S={S} "
          f"expected[:8]={expected[:8]}")

    # embed -> hc expansion (source Model.forward semantics)
    embed_w = arch_get("embed.weight").to(DEV)  # bf16
    h_ours = F.embedding(input_ids, embed_w).unsqueeze(2) \
        .repeat(1, 1, HC, 1)
    h_src = h_ours.clone() if mode == "parity" else None

    # per-token loop (prefill = 1 pass at start_pos=0 with S tokens)
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    generated = []
    t_start = None
    import time
    for step in range(n_gen + 1):
        t0 = time.time()
        start_pos = 0 if step == 0 else S + step - 1
        ids_step = input_ids if step == 0 else \
            generated[-1].reshape(1, 1)
        cur = ids_step.shape[1]
        for L in range(N_LAYERS):
            shared_w = tuple(dq_fp8_shared(L, nm).to(DEV)
                             for nm in ("w1", "w3", "w2"))
            attn, router = build_ours_layer(L)
            if step > 0 and STATES[L].kv_cache is not None:
                STATES[L].restore(attn)
            # step 0 (prefill) starts from the module's zeroed state
            x_in = h_ours
            with torch.no_grad():
                h_new = ours_layer_forward(L, x_in, start_pos, ids_step,
                                           attn, router, shared_w)
            STATES[L].save(attn)  # save after EVERY pass incl. prefill
            h_ours = h_new
            del attn, router, shared_w
            _EXPERT_CACHE.clear()  # CPU RAM: ~2.3GB/layer if kept
            torch.cuda.empty_cache()

        logits = head_forward(h_ours[:, -cur:, :]) \
            if step == 0 else head_forward(h_ours)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        if step == 0:
            generated.append(nxt)
            print(f"prefill done ({time.time()-t0:.0f}s) "
                  f"first token: {nxt.item()} (expected {expected[0]}) "
                  f"{'MATCH' if nxt.item() == expected[0] else 'DIFF'}")
        else:
            generated.append(nxt)
            gen_ids = [t.item() for t in generated]
            match = gen_ids == expected[:len(gen_ids)]
            print(f"token {step}: {nxt.item()} "
                  f"(expected {expected[step]}) "
                  f"{'MATCH' if nxt.item() == expected[step] else 'DIFF'} "
                  f"| running exact: {match} "
                  f"({time.time()-t0:.0f}s)")
            if not match:
                print(f"  first divergence at token {step}; stopping")
                break
        # for the next decode step, feed the new token: h_ours currently
        # holds the layer outputs for the LAST position only (decode);
        # rebuild input for the next token from the embedding + hc
        if step < n_gen:
            h_ours = F.embedding(nxt, embed_w).unsqueeze(2) \
                .repeat(1, 1, HC, 1)

    print("\n=== EXACT-MODEL GOLDEN COMPLETE ===")
    gen_ids = [t.item() for t in generated]
    n_match = sum(1 for a, b in zip(gen_ids, expected) if a == b)
    print(f"generated: {gen_ids}")
    print(f"expected:  {expected[:len(gen_ids)]}")
    print(f"exact prefix: {n_match}/{len(gen_ids)}")


if __name__ == "__main__":
    main()
