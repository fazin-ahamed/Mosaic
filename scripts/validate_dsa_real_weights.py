"""§24 IMMEDIATE NEXT — real-weight DSA numeric validation (authoritative bytes).

Same side-by-side structure as validate_dsa_source.py, but weights are the
REAL authoritative checkpoint tensors (dequantized fp8×e8m0 → bf16 for the
projections; fp32 for norms/compressor/indexer-compressor; bf16 for
weights_proj). This is the validation that was BLOCKED by the stale-CDN
archive (mutual-finite overlap 0%); on authoritative bytes it must pass.

Both sides receive IDENTICAL dequantized weights, so this validates the DSA
port mechanics under production value ranges with production quantization
semantics (ue8m0 act-quant sims on, bf16 leaves, F.linear).

Covers layers 0 (ratio 0), 2 (ratio 4 + indexer), 3 (ratio 128) at prefill,
decode, and chunked widths.
"""
import json
import sys
import types

import torch

sys.path.insert(0, ".")
sys.path.insert(0, ".mosaicvm/source")

import mosaicvm.model.dsa as mdsa  # noqa: E402

fht = types.ModuleType("fast_hadamard_transform")
fht.hadamard_transform = lambda x, scale=None: mdsa.hadamard(x)
sys.modules["fast_hadamard_transform"] = fht

import inference_kernel  # noqa: E402
sys.modules["kernel"] = inference_kernel

# SM86 shared-memory limit: the fused Triton kernel (BLOCK_T=64 x D=512,
# stages=2 = ~114KB) exceeds the 3060's 101KB at full attention width.
# Use the eager BITWISE_MOSAIC_REFERENCE path (dsa_layers.sparse_attn_ref)
# for this validation; the fused kernel is separately parity-tested
# (tests/test_dsa_sparse_fused.py). Production dtype stays bf16.
import mosaicvm.model.dsa_layers as _DL  # noqa: E402
_DL.USE_FUSED_SPARSE = False

DEV = "cuda"
MAXSEQ = 1024

torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(DEV)

import inference_model as SM  # noqa: E402

# ---- leaf kernel shims (identical to validate_dsa_source.py) ----------------
_E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
_E2M1_MID = (_E2M1[1:] + _E2M1[:-1]) / 2


def _act_quant(x, block_size=128, scale_fmt=None,
               scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous().float()
    xb = z.view(*z.shape[:-1], N // block_size, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    if scale_fmt is not None:
        s = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    else:
        s = amax / 448.0
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


SM.act_quant = _act_quant
SM.fp4_act_quant = _fp4_act_quant
SM.sparse_attn = _sparse_attn
SM.hc_split_sinkhorn = _hc_split_sinkhorn

SM.default_dtype = torch.bfloat16
SM.scale_fmt = "ue8m0"
SM.scale_dtype = torch.float8_e8m0fnu

# ---- authoritative archive accessor ----------------------------------------
from safetensors import safe_open  # noqa: E402

ARCH = safe_open(".mosaicvm/core/core.safetensors", framework="pt")


def dq_fp8(layer: int, proj: str):
    """Dequantize layers.L.attn.<proj>.{weight,scale} -> bf16 [out, in]."""
    w = ARCH.get_tensor(f"layers.{layer}.attn.{proj}.weight")
    s = ARCH.get_tensor(f"layers.{layer}.attn.{proj}.scale")
    rows, cols = w.shape
    bs = s.shape
    sf = s.float()
    W = w.float() * sf.repeat_interleave(rows // bs[0], 0) \
        .repeat_interleave(cols // bs[1], 1)
    return W.to(torch.bfloat16)


def get32(layer: int, name: str):
    return ARCH.get_tensor(f"layers.{layer}.attn.{name}").float()


# ---- shared config -----------------------------------------------------------
src_cfg = json.load(open(".mosaicvm/source/inference_config.json"))
from mosaicvm.model.config import V4Config  # noqa: E402
from mosaicvm.model.dsa_layers import (  # noqa: E402
    ExactAttention, cast_exact_attention)

cfg = V4Config()


def build_args():
    args = SM.ModelArgs()
    args.__dict__.update({
        "dim": src_cfg["dim"], "n_heads": src_cfg["n_heads"],
        "head_dim": src_cfg["head_dim"],
        "rope_head_dim": src_cfg["rope_head_dim"],
        "q_lora_rank": src_cfg["q_lora_rank"],
        "o_groups": src_cfg["o_groups"],
        "o_lora_rank": src_cfg["o_lora_rank"],
        "window_size": src_cfg["window_size"],
        "original_seq_len": src_cfg["original_seq_len"],
        "rope_theta": src_cfg["rope_theta"],
        "rope_factor": src_cfg["rope_factor"],
        "beta_fast": src_cfg["beta_fast"],
        "beta_slow": src_cfg["beta_slow"],
        "index_n_heads": src_cfg["index_n_heads"],
        "index_head_dim": src_cfg["index_head_dim"],
        "index_topk": src_cfg["index_topk"], "norm_eps": 1e-6,
        "max_batch_size": 1, "max_seq_len": MAXSEQ,
        "compress_rope_theta": src_cfg["compress_rope_theta"],
        "compress_ratios": src_cfg["compress_ratios"],
    })
    return args


def load_real(src, ours, layer: int):
    with torch.no_grad():
        for s_nm, o_nm in [("wq_a", "q_a"), ("wq_b", "q_b"),
                           ("wkv", "kv_proj"), ("wo_a", "wo_a"),
                           ("wo_b", "wo_b")]:
            w = dq_fp8(layer, s_nm)
            getattr(src, s_nm).weight.copy_(w)
            getattr(ours, o_nm).weight.copy_(w)
        for nm in ("q_norm", "kv_norm"):
            v = get32(layer, f"{nm}.weight")
            getattr(src, nm).weight.copy_(v)
            getattr(ours, nm).weight.copy_(v)
        sink = get32(layer, "attn_sink")
        src.attn_sink.copy_(sink)
        ours.attn_sink.data.copy_(sink)

        def lc(scomp, ocomp, pre):
            scomp.ape.copy_(ARCH.get_tensor(f"{pre}.ape").float())
            ocomp.ape.copy_(ARCH.get_tensor(f"{pre}.ape").float())
            for nm in ("wkv", "wgate"):
                t = ARCH.get_tensor(f"{pre}.{nm}.weight").float()
                getattr(scomp, nm).weight.copy_(t)
                getattr(ocomp, nm).weight.copy_(t)
            t = ARCH.get_tensor(f"{pre}.norm.weight").float()
            scomp.norm.weight.copy_(t)
            ocomp.norm.weight.copy_(t)

        if getattr(src, "compressor", None) is not None:
            lc(src.compressor, ours.compressor,
               f"layers.{layer}.attn.compressor")
            if getattr(src, "indexer", None) is not None \
                    and ours.indexer is not None:
                ix, oix = src.indexer, ours.indexer
                w = dq_fp8(layer, "indexer.wq_b")
                ix.wq_b.weight.copy_(w)
                oix.wq_b.weight.copy_(w)
                wp = ARCH.get_tensor(
                    f"layers.{layer}.attn.indexer.weights_proj.weight") \
                    .to(torch.bfloat16)
                ix.weights_proj.weight.copy_(wp)
                oix.weights_proj.weight.copy_(wp)
                lc(ix.compressor, oix.compressor,
                   f"layers.{layer}.attn.indexer.compressor")


def compare(a, b, label):
    a, b = a.float(), b.float()
    fa = torch.isfinite(a).float().mean().item()
    fb = torch.isfinite(b).float().mean().item()
    fin = torch.isfinite(a) & torch.isfinite(b)
    line = f"  {label}: fin(src)={fa:.4f} fin(ours)={fb:.4f}"
    if fin.any():
        d = (a - b).abs()[fin]
        rel = d / b[fin].abs().clamp_min(1e-3)
        line += (f" max_abs={d.max().item():.5f}"
                 f" rtol2%={(rel <= 0.02).float().mean().item():.4f}"
                 f" rtol0.2%={(rel <= 0.002).float().mean().item():.4f}")
    print(line)
    return fa, fb


for LAYER, label in [(0, "ratio0"), (2, "ratio4+indexer"),
                     (3, "ratio128")]:
    args = build_args()
    src_attn = SM.Attention(LAYER, args).to(DEV)
    ours = ExactAttention(cfg, layer_id=LAYER, max_seq=MAXSEQ, device=DEV)
    cast_exact_attention(ours, DEV)
    load_real(src_attn, ours, LAYER)

    print(f"\n=== layer {LAYER} ({label}) — REAL authoritative weights ===")
    x = torch.randn(1, 256, args.dim, dtype=torch.bfloat16, device=DEV)

    with torch.no_grad():
        o_src = src_attn(x, 0)
        o_ours = ours(x.squeeze(0)).unsqueeze(0)
        compare(o_src, o_ours, "prefill s=256")

        xd = torch.randn(1, 1, args.dim, dtype=torch.bfloat16, device=DEV)
        o_src = src_attn(xd, 256)
        o_ours = ours(xd.squeeze(0), 256).unsqueeze(0)
        compare(o_src, o_ours, "decode p=256")

        # chunked: 130 then 34 (remainder-lead path), fresh state
        src_attn2 = SM.Attention(LAYER, args).to(DEV)
        ours2 = ExactAttention(cfg, layer_id=LAYER, max_seq=MAXSEQ,
                               device=DEV)
        cast_exact_attention(ours2, DEV)
        load_real(src_attn2, ours2, LAYER)
        xc1 = torch.randn(1, 130, args.dim, dtype=torch.bfloat16, device=DEV)
        xc2 = torch.randn(1, 34, args.dim, dtype=torch.bfloat16, device=DEV)
        # reference: source FULL-prefix prefill (the source has no chunked
        # path); compare its rows 130: against our chunked output
        x_full = torch.cat([xc1, xc2], dim=1)
        ref_full = src_attn2(x_full, 0)
        _ = ours2(xc1.squeeze(0), 0)
        o_ours = ours2(xc2.squeeze(0), 130).unsqueeze(0)
        compare(ref_full[:, 130:], o_ours, "chunked 130+34")

print("\n=== REAL-WEIGHT DSA VALIDATION COMPLETE ===")
