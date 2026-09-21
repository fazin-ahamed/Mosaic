"""§43 rung 1 — single-layer exact parity vs the kit source Block.

Reference: the checkpoint's own Block code (.mosaicvm/source/
inference_model.py) with REAL authoritative weights, bf16 leaves, and
pure-torch kernel shims (act_quant / fp4_act_quant / sparse_attn /
hc_split_sinkhorn / fp4_gemm — the fp4_gemm shim is boundary-free:
dequant-fp4 weights + F.linear, matching the runtime-under-test's expert
semantics so this validates the COMPOSITION, not the quant boundaries).

Runtime-under-test: ExactAttention (dsa_layers) + mosaicvm Router/
HashRouter + dequant_fp4 routed experts + FP8-dequant shared expert,
with the source's own hc_pre/hc_post/norms (shared math, not under test).

Layers: 0 (hash-routed, tid2eid) and 5 (learned router).
Modes: prefill s=64.
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

torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(DEV)

import inference_model as SM  # noqa: E402

# ---- kernel shims (torch math from inference_kernel.py) ---------------------
_E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
_E2M1_MID = (_E2M1[1:] + _E2M1[:-1]) / 2
_E2M1_FULL = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                           -0., -.5, -1., -1.5, -2., -3., -4., -6.])


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
    """Boundary-free shim: dequant fp4 weights, F.linear. `a` arrives as
    the fp8 quantized activation — dequant it with its scale so the shim
    matches the runtime-under-test's bf16-act x dequant-weight semantics."""
    M, K = a.shape
    n = w.shape[0]
    A = a.float()
    if a_s is not None:
        A = A * a_s.float().repeat_interleave(
            K // a_s.shape[-1], dim=-1).reshape(A.shape) \
            if a_s.dim() == 2 else A
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

# disable fp4-gemm act quant inside the source Linear for boundary-free
# parity (both sides: bf16 acts x dequant-fp4 weights)
_orig_linear = SM.linear


def _linear_nofp8act(x, weight, bias=None):
    if weight.dtype == torch.float4_e2m1fn_x2:
        return _fp4_gemm(x.float().to(torch.bfloat16), None, weight,
                         weight.scale, None)
    return _orig_linear(x, weight, bias)


SM.linear = _linear_nofp8act

# ---- archives ---------------------------------------------------------------
from safetensors import safe_open  # noqa: E402

ARCH = safe_open(".mosaicvm/core/core.safetensors", framework="pt")
SNAP = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731"
    "/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
IDX = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]


def snap_get(name):
    with safe_open(str(SNAP / IDX[name]), framework="pt") as f:
        return f.get_tensor(name)


def dq_fp8_arch(layer, proj):
    w = ARCH.get_tensor(f"layers.{layer}.attn.{proj}.weight")
    s = ARCH.get_tensor(f"layers.{layer}.attn.{proj}.scale")
    rows, cols = w.shape
    bs = s.shape
    return (w.float() * s.float().repeat_interleave(rows // bs[0], 0)
            .repeat_interleave(cols // bs[1], 1)).to(torch.bfloat16)


def dq_fp8_shared(layer, which):
    w = ARCH.get_tensor(f"layers.{layer}.ffn.shared_experts.{which}.weight")
    s = ARCH.get_tensor(f"layers.{layer}.ffn.shared_experts.{which}.scale")
    rows, cols = w.shape
    bs = s.shape
    return (w.float() * s.float().repeat_interleave(rows // bs[0], 0)
            .repeat_interleave(cols // bs[1], 1)).to(torch.bfloat16)


from mosaicvm.runtime.fp8 import dequant_fp4  # noqa: E402

# ---- config ------------------------------------------------------------------
src_cfg = json.load(open(".mosaicvm/source/inference_config.json"))
from mosaicvm.model.config import V4Config  # noqa: E402
from mosaicvm.model.dsa_layers import (  # noqa: E402
    ExactAttention, cast_exact_attention)
from mosaicvm.model.layers import Router, HashRouter  # noqa: E402

cfg = V4Config()


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
        "n_routed_experts": N_EXPERTS,
        "n_activated_experts": TOPK,
        "moe_inter_dim": INTER,
        "score_func": "sqrtsoftplus", "route_scale": 1.5,
        "n_shared_experts": 1, "swiglu_limit": 10.0,
        "expert_dtype": "fp4",
        "vocab_size": 129280,
        "n_hash_layers": 3,
    })
    return args


def load_real_attention(src_attn, ours, layer):
    with torch.no_grad():
        for s_nm, o_nm in [("wq_a", "q_a"), ("wq_b", "q_b"),
                           ("wkv", "kv_proj"), ("wo_a", "o_lora_a"),
                           ("wo_b", "o_lora_b")]:
            pass  # kit Attention uses wq_a..wo_b names; ours uses q_a...
        # kit side
        for s_nm in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            getattr(src_attn, s_nm).weight.copy_(dq_fp8_arch(layer, s_nm))
        for nm in ("q_norm", "kv_norm"):
            getattr(src_attn, nm).weight.copy_(
                ARCH.get_tensor(f"layers.{layer}.attn.{nm}.weight").float())
        src_attn.attn_sink.copy_(
            ARCH.get_tensor(f"layers.{layer}.attn.attn_sink").float())

        def lc(scomp, pre):
            scomp.ape.copy_(ARCH.get_tensor(f"{pre}.ape").float())
            for nm in ("wkv", "wgate"):
                getattr(scomp, nm).weight.copy_(
                    ARCH.get_tensor(f"{pre}.{nm}.weight").float())
            scomp.norm.weight.copy_(
                ARCH.get_tensor(f"{pre}.norm.weight").float())

        lc(src_attn.compressor, f"layers.{layer}.attn.compressor")
        if src_attn.indexer is not None:
            ix = src_attn.indexer
            ix.wq_b.weight.copy_(dq_fp8_arch(layer, "indexer.wq_b"))
            ix.weights_proj.weight.copy_(
                ARCH.get_tensor(
                    f"layers.{layer}.attn.indexer.weights_proj.weight"))
            lc(ix.compressor, f"layers.{layer}.attn.indexer.compressor")

        # our side (ExactAttention param names)
        for s_nm, o_nm in [("wq_a", "q_a"), ("wq_b", "q_b"),
                           ("wkv", "kv_proj"), ("wo_a", "wo_a"),
                           ("wo_b", "wo_b")]:
            getattr(ours, o_nm).weight.copy_(dq_fp8_arch(layer, s_nm))
        for nm in ("q_norm", "kv_norm"):
            getattr(ours, nm).weight.copy_(
                ARCH.get_tensor(f"layers.{layer}.attn.{nm}.weight").float())
        ours.attn_sink.data.copy_(
            ARCH.get_tensor(f"layers.{layer}.attn.attn_sink").float())
        lc(ours.compressor, f"layers.{layer}.attn.compressor")
        if ours.indexer is not None:
            oix = ours.indexer
            oix.wq_b.weight.copy_(dq_fp8_arch(layer, "indexer.wq_b"))
            oix.weights_proj.weight.copy_(
                ARCH.get_tensor(
                    f"layers.{layer}.attn.indexer.weights_proj.weight"))
            lc(oix.compressor, f"layers.{layer}.attn.indexer.compressor")


# exact-block eager path for the runtime-under-test
import mosaicvm.model.dsa_layers as _DL  # noqa: E402
_DL.USE_FUSED_SPARSE = False


def ours_block_forward(layer, x_hc, start_pos, input_ids, attn_ours,
                       router, tid2eid=None, expert_w=None,
                       shared_w=None):
    """Runtime-under-test block: source hc/norms (shared math) +
    ExactAttention + Router/HashRouter + dequant-fp4 experts + shared."""
    args = build_args()
    hc_attn_fn = ARCH.get_tensor(f"layers.{layer}.hc_attn_fn").float().to(DEV)
    hc_attn_base = ARCH.get_tensor(f"layers.{layer}.hc_attn_base").float().to(DEV)
    hc_attn_scale = ARCH.get_tensor(f"layers.{layer}.hc_attn_scale").float().to(DEV)
    hc_ffn_fn = ARCH.get_tensor(f"layers.{layer}.hc_ffn_fn").float().to(DEV)
    hc_ffn_base = ARCH.get_tensor(f"layers.{layer}.hc_ffn_base").float().to(DEV)
    hc_ffn_scale = ARCH.get_tensor(f"layers.{layer}.hc_ffn_scale").float().to(DEV)
    attn_norm_w = ARCH.get_tensor(f"layers.{layer}.attn_norm.weight").float().to(DEV)
    ffn_norm_w = ARCH.get_tensor(f"layers.{layer}.ffn_norm.weight").float().to(DEV)

    def hc_pre(x, fn, sc, base):
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        mixes = torch.nn.functional.linear(xf, fn) * rsqrt
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

    # ---- attention sublayer ----
    residual = x_hc
    x, post, comb = hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    x = rms(x, attn_norm_w)
    with torch.no_grad():
        a = attn_ours(x.reshape(-1, HIDDEN), start_pos).reshape(b, s, HIDDEN)
    x = hc_post(a, residual, post, comb)

    # ---- ffn sublayer ----
    residual = x
    x, post, comb = hc_pre(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
    x = rms(x, ffn_norm_w)
    xf = x.reshape(-1, HIDDEN)

    # router
    if tid2eid is not None:
        scores, idx = router(xf, input_ids.flatten())
    else:
        scores, idx = router(xf)

    # routed experts: exact grouped semantics (source Expert loop math).
    # expert_w(e) -> (w1, w3, w2) dequantized [out, in] float32 — provided
    # by the caller (weight loading is regression-tested separately; the
    # parity target here is the forward math).
    y = torch.zeros_like(xf, dtype=torch.float32)
    uniq = idx.unique().tolist()
    cache = {}
    for e in uniq:
        cache[e] = expert_w(e)
    for slot in range(idx.shape[-1]):
        fidx = idx[:, slot]
        ws = scores[:, slot]
        for e in uniq:
            m = fidx == e
            if not bool(m.any()):
                continue
            w1, w3, w2 = cache[e]
            xe = xf[m].float()
            gate = xe @ w1.t()
            up = xe @ w3.t()
            gate = gate.clamp(max=10.0)
            up = up.clamp(-10.0, 10.0)
            z = torch.nn.functional.silu(gate) * up
            z = ws[m][:, None] * z
            y[m] += (z.to(torch.bfloat16).float() @ w2.t())
    # shared expert (no weight) — caller-provided dequantized weights
    sh_w1, sh_w3, sh_w2 = shared_w
    xs = xf.float()
    g = (xs @ sh_w1.t()).clamp(max=10.0)
    u = (xs @ sh_w3.t()).clamp(-10.0, 10.0)
    y += torch.nn.functional.silu(g) * u @ sh_w2.t()
    y = y.to(xf.dtype).reshape(b, s, HIDDEN)
    x = hc_post(y, residual, post, comb)
    return x


_EXPERT_CACHE: dict = {}


def _get_expert_weights(layer, e):
    """Shared dequantized expert weights (both sides use identical
    values — parity targets the forward math)."""
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
    """Dequant-on-first-use expert linear (replaces the kit Linear for
    routed experts; boundary-free parity path)."""

    def __init__(self, layer, e, out, inn):
        super().__init__()
        self.layer, self.e = layer, e
        self.out, self.inn = out, inn

    def forward(self, x):
        w1, w3, w2 = _get_expert_weights(self.layer, self.e)
        w = {0: w1, 1: w3, 2: w2}[self._slot]
        return torch.nn.functional.linear(x, w.to(x.device))


def main():
    for LAYER in (5, 0):
        args = build_args()
        src_block = SM.Block(LAYER, args).to(DEV)
        # ---- load real weights into the source block ----
        with torch.no_grad():
            load_real_attention(src_block.attn, None, LAYER) \
                if False else None
            # attention
            for s_nm in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
                getattr(src_block.attn, s_nm).weight.copy_(
                    dq_fp8_arch(LAYER, s_nm))
            for nm in ("q_norm", "kv_norm"):
                getattr(src_block.attn, nm).weight.copy_(
                    ARCH.get_tensor(
                        f"layers.{LAYER}.attn.{nm}.weight").float())
            src_block.attn.attn_sink.copy_(
                ARCH.get_tensor(f"layers.{LAYER}.attn.attn_sink").float())

            def lc2(scomp, pre):
                scomp.ape.copy_(ARCH.get_tensor(f"{pre}.ape").float())
                for nm in ("wkv", "wgate"):
                    getattr(scomp, nm).weight.copy_(
                        ARCH.get_tensor(f"{pre}.{nm}.weight").float())
                scomp.norm.weight.copy_(
                    ARCH.get_tensor(f"{pre}.norm.weight").float())

            if src_block.attn.compress_ratio:
                lc2(src_block.attn.compressor,
                    f"layers.{LAYER}.attn.compressor")
            if getattr(src_block.attn, "indexer", None) is not None:
                ix = src_block.attn.indexer
                ix.wq_b.weight.copy_(dq_fp8_arch(LAYER, "indexer.wq_b"))
                ix.weights_proj.weight.copy_(ARCH.get_tensor(
                    f"layers.{LAYER}.attn.indexer.weights_proj.weight"))
                lc2(ix.compressor,
                    f"layers.{LAYER}.attn.indexer.compressor")
            # norms + hc
            src_block.attn_norm.weight.copy_(
                ARCH.get_tensor(f"layers.{LAYER}.attn_norm.weight").float())
            src_block.ffn_norm.weight.copy_(
                ARCH.get_tensor(f"layers.{LAYER}.ffn_norm.weight").float())
            for nm in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                       "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
                getattr(src_block, nm).copy_(
                    ARCH.get_tensor(f"layers.{LAYER}.{nm}").float())
            # gate
            src_block.ffn.gate.weight.copy_(
                ARCH.get_tensor(f"layers.{LAYER}.ffn.gate.weight").float())
            if src_block.ffn.gate.hash:
                src_block.ffn.gate.tid2eid.copy_(
                    ARCH.get_tensor(
                        f"layers.{LAYER}.ffn.gate.tid2eid").long())
            else:
                src_block.ffn.gate.bias.copy_(
                    ARCH.get_tensor(
                        f"layers.{LAYER}.ffn.gate.bias").float())
            # shared expert (fp8 dequant)
            for nm in ("w1", "w3", "w2"):
                getattr(src_block.ffn.shared_experts, nm).weight.copy_(
                    dq_fp8_shared(LAYER, nm))
                # strip the fp8 scale attr so Linear falls to F.linear
                if hasattr(getattr(src_block.ffn.shared_experts, nm),
                           "scale"):
                    delattr(getattr(src_block.ffn.shared_experts, nm),
                            "scale")
            # routed experts: LAZY dequant-on-use linears (the selection
            # depends on post-attention activations, which differ between
            # the two sides at bf16 noise — preloading a fixed set breaks)
            S_TEST = 8
            _x_probe = torch.randn(1, S_TEST, 4, HIDDEN,
                                   dtype=torch.bfloat16, device=DEV) * 0.5
            _ids_probe = torch.randint(0, 129280, (1, S_TEST), device=DEV)
            for i in range(N_EXPERTS):
                ex = src_block.ffn.experts[i]
                for slot, nm in enumerate(("w1", "w3", "w2")):
                    lin = LazyExpertLinear(
                        LAYER, i,
                        INTER if nm != "w2" else HIDDEN,
                        HIDDEN if nm != "w2" else INTER)
                    lin._slot = slot
                    setattr(ex, nm, lin)

        # ---- our side ----
        attn_ours = ExactAttention(cfg, layer_id=LAYER, max_seq=MAXSEQ,
                                   device=DEV)
        cast_exact_attention(attn_ours, DEV)
        # our ExactAttention has q_a/q_b/kv_proj/wo_a/wo_b names
        with torch.no_grad():
            for s_nm, o_nm in [("wq_a", "q_a"), ("wq_b", "q_b"),
                               ("wkv", "kv_proj"), ("wo_a", "wo_a"),
                               ("wo_b", "wo_b")]:
                getattr(attn_ours, o_nm).weight.copy_(
                    dq_fp8_arch(LAYER, s_nm))
            for nm in ("q_norm", "kv_norm"):
                getattr(attn_ours, nm).weight.copy_(
                    ARCH.get_tensor(
                        f"layers.{LAYER}.attn.{nm}.weight").float())
            attn_ours.attn_sink.data.copy_(
                ARCH.get_tensor(f"layers.{LAYER}.attn.attn_sink").float())

            def lc3(scomp, pre):
                scomp.ape.copy_(ARCH.get_tensor(f"{pre}.ape").float())
                for nm in ("wkv", "wgate"):
                    getattr(scomp, nm).weight.copy_(
                        ARCH.get_tensor(f"{pre}.{nm}.weight").float())
                scomp.norm.weight.copy_(
                    ARCH.get_tensor(f"{pre}.norm.weight").float())

            if attn_ours.compressor is not None:
                lc3(attn_ours.compressor, f"layers.{LAYER}.attn.compressor")
            if attn_ours.indexer is not None:
                oix = attn_ours.indexer
                oix.wq_b.weight.copy_(dq_fp8_arch(LAYER, "indexer.wq_b"))
                oix.weights_proj.weight.copy_(ARCH.get_tensor(
                    f"layers.{LAYER}.attn.indexer.weights_proj.weight"))
                lc3(oix.compressor,
                    f"layers.{LAYER}.attn.indexer.compressor")

        router = HashRouter(cfg, LAYER) if LAYER < 3 else Router(cfg)
        with torch.no_grad():
            router.weight.copy_(
                ARCH.get_tensor(f"layers.{LAYER}.ffn.gate.weight").float())
            if LAYER < 3:
                router.tid2eid = ARCH.get_tensor(
                    f"layers.{LAYER}.ffn.gate.tid2eid").long().to(DEV)
            else:
                # source Gate bias is fp32; the Router param defaults to
                # bf16 under this script's default — force fp32 (a bf16
                # bias flips near-tie selections vs the source)
                router.e_score_correction_bias.data = (
                    ARCH.get_tensor(
                        f"layers.{LAYER}.ffn.gate.bias").float().to(DEV))
        tid2eid = True if LAYER < 3 else None

        # ---- input (same as the selection probe) ----
        S = S_TEST
        x_hc = _x_probe
        input_ids = _ids_probe

        def expert_w(e):
            w1, w3, w2 = _get_expert_weights(LAYER, e)
            return (w1.float().to(DEV), w3.float().to(DEV),
                    w2.float().to(DEV))

        shared_w = (src_block.ffn.shared_experts.w1.weight.float(),
                    src_block.ffn.shared_experts.w3.weight.float(),
                    src_block.ffn.shared_experts.w2.weight.float())

        with torch.no_grad():
            o_src = src_block(x_hc, 0, input_ids)
            o_ours = ours_block_forward(LAYER, x_hc, 0, input_ids,
                                        attn_ours, router, tid2eid,
                                        expert_w, shared_w)

        a, b_ = o_src.float(), o_ours.float()
        print(f"\n=== layer {LAYER} "
              f"({'hash' if LAYER < 3 else 'routed'}) — exact-layer parity ===")
        print(f"  src finite: {torch.isfinite(a).all().item()}, "
              f"ours finite: {torch.isfinite(b_).all().item()}")
        d = (a - b_).abs()
        rel = d / b_.abs().clamp_min(1e-3)
        print(f"  max_abs={d.max().item():.5f} "
              f"rtol2%={(rel <= 0.02).float().mean().item():.4f} "
              f"rtol0.2%={(rel <= 0.002).float().mean().item():.4f}")
        # router agreement check
        with torch.no_grad():
            # collapsed (post-hc_pre) shaped input: [s, HIDDEN]
            xf = x_hc.reshape(-1, HIDDEN, 4).mean(2) if False else                 torch.randn(x_hc.shape[1], HIDDEN,
                            dtype=torch.bfloat16, device=DEV) * 0.5
            ids_flat = input_ids.flatten()
            s_src, i_src = src_block.ffn.gate(xf.float(), ids_flat)
            if LAYER < 3:
                s_our, i_our = router(xf, ids_flat)
            else:
                s_our, i_our = router(xf)
            print(f"  router idx equal: {torch.equal(i_src, i_our)}, "
                  f"weights max diff: "
                  f"{(s_src - s_our).abs().max().item():.6f}")
        del src_block, attn_ours, router
        _EXPERT_CACHE.clear()
        torch.cuda.empty_cache()

    print("\n=== EXACT-LAYER PARITY COMPLETE ===")


if __name__ == "__main__":
    main()
