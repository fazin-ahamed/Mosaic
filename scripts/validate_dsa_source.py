"""§60-1: side-by-side validation of mosaicvm ExactAttention vs the
checkpoint's own reference implementation (.mosaicvm/source/inference_model.py).

Runs the SOURCE model code (not a re-translation) on the local 3060 —
semantics comparison only, never a P-level timing claim.

Leaf configuration (per §60-1 review):
  * SM.default_dtype = bf16 -> Linear() creates bf16 weights and the kit's
    linear() dispatches to F.linear (no fp8 GEMM anywhere).
  * SM.scale_fmt/scale_dtype keep the production ue8m0 values so the
    act_quant / fp4_act_quant calls inside the model code run the exact
    production quantization semantics (pow2 scales, e4m3/e2m1 round-trip).
  * Weights: IDENTICAL seeded-random bf16 tensors copied into BOTH sides —
    the public checkpoint's attention scale tensors are integrity-broken
    (artifacts/deviation_checkpoint_integrity.md), so real weights cannot
    validate the port; random weights validate every code path except the
    (shimmed) tilelang kernels.
  * __init__ buffers are left alone: the source Attention allocates
    kv_cache [b, window + max_seq//ratio, head_dim] and freqs_cis
    (compress_rope_theta=160000 with YaRN when ratio != 0) itself.

Kernel shims (tilelang JIT is blocked on this host by the gcc16/cu13 CCCL
incompatibility): act_quant / fp4_act_quant / sparse_attn /
hc_split_sinkhorn are re-derived from inference_kernel.py source.
"""
import sys
import types

import torch

sys.path.insert(0, ".")
sys.path.insert(0, ".mosaicvm/source")

import mosaicvm.model.dsa as mdsa  # noqa: E402  (hadamard shim)

fht = types.ModuleType("fast_hadamard_transform")
fht.hadamard_transform = lambda x, scale=None: mdsa.hadamard(x)
sys.modules["fast_hadamard_transform"] = fht

import inference_kernel  # noqa: E402
sys.modules["kernel"] = inference_kernel

DEV = "cuda"
LAYER = 2          # ratio 4 + indexer — the fullest stack
S = 256            # multiple of window 128 and ratio 4
MAXSEQ = 1024

torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)   # kit default serving dtype
torch.set_default_device(DEV)             # kit assumes a device context

import inference_model as SM  # noqa: E402

# ---- leaf kernel shims (from inference_kernel.py math) --------------------
_E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
_E2M1_MID = (_E2M1[1:] + _E2M1[:-1]) / 2


def _act_quant(x, block_size=128, scale_fmt=None,
               scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous().float()
    xb = z.view(*z.shape[:-1], N // block_size, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    if scale_fmt is not None:  # pow2 scale (fast_round_scale)
        s = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    else:
        s = amax / 448.0
    qf = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        return x.copy_((qf.float() * s).to(x.dtype).reshape(*x.shape))
    return qf.reshape(*x.shape), s.reshape(*x.shape[:-1], N // block_size)


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
    return dq.to(torch.float8_e4m3fn).reshape(*x.shape), s


def _sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    # online-softmax gather attention with denominator-only sink (kernel math;
    # p is cast to bf16 before the value gemm, as the kernel does)
    b, s, h, d = q.size()
    valid = topk_idxs >= 0
    idxc = topk_idxs.clamp(min=0)
    g = kv.unsqueeze(1).expand(b, s, *kv.shape[1:]) \
        .gather(2, idxc.unsqueeze(-1).expand(-1, -1, -1, d))
    scores = torch.einsum("bshd,bstd->bsht", q.float(), g.float()) * softmax_scale
    scores = torch.where(valid.unsqueeze(2), scores, torch.full_like(scores, float("-inf")))
    m = scores.max(-1, keepdim=True).values
    p = torch.exp(scores - m)
    p_bf = p.to(torch.bfloat16).float()
    denom = p_bf.sum(-1) + torch.exp(attn_sink.view(1, 1, h).float() - m.squeeze(-1))
    o = torch.einsum("bsht,bstd->bshd", p_bf, g.float()) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre_ = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].view(*mixes.shape[:-1], hc, hc) * hc_scale[2] \
        + hc_base[2 * hc:].view(hc, hc)
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

# ---- bf16 leaves: F.linear everywhere; production quant semantics ----------
SM.default_dtype = torch.bfloat16
SM.scale_fmt = "ue8m0"
SM.scale_dtype = torch.float8_e8m0fnu

# ---- source side (buffers from __init__, untouched) ------------------------
import json  # noqa: E402
src_cfg = json.load(open(".mosaicvm/source/inference_config.json"))
args = SM.ModelArgs()
args.__dict__.update({
    "dim": src_cfg["dim"], "n_heads": src_cfg["n_heads"],
    "head_dim": src_cfg["head_dim"], "rope_head_dim": src_cfg["rope_head_dim"],
    "q_lora_rank": src_cfg["q_lora_rank"], "o_groups": src_cfg["o_groups"],
    "o_lora_rank": src_cfg["o_lora_rank"], "window_size": src_cfg["window_size"],
    "original_seq_len": src_cfg["original_seq_len"],
    "rope_theta": src_cfg["rope_theta"], "rope_factor": src_cfg["rope_factor"],
    "beta_fast": src_cfg["beta_fast"], "beta_slow": src_cfg["beta_slow"],
    "index_n_heads": src_cfg["index_n_heads"],
    "index_head_dim": src_cfg["index_head_dim"],
    "index_topk": src_cfg["index_topk"], "norm_eps": 1e-6,
    "max_batch_size": 1, "max_seq_len": MAXSEQ,
    "compress_rope_theta": src_cfg["compress_rope_theta"],
    "compress_ratios": src_cfg["compress_ratios"],
})
src_attn = SM.Attention(LAYER, args).to(DEV)
# .to() replaces Parameter objects and drops the kit's `weight.scale`
# attribute; re-attach where present (bf16 linears have none)
assert src_attn.compress_ratio == 4
# buffers stay exactly as __init__ built them: kv_cache [1, 128+256, 512]
# bf16 and freqs_cis [1024, 32] complex (theta 160000, YaRN 65536)

# ---- identical seeded-random weights into BOTH sides -----------------------
from mosaicvm.model.config import V4Config  # noqa: E402
from mosaicvm.model.dsa_layers import (  # noqa: E402
    ExactAttention, cast_exact_attention)

cfg = V4Config()
ours = ExactAttention(cfg, layer_id=LAYER, max_seq=MAXSEQ, device=DEV)
cast_exact_attention(ours, DEV)


def gen(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, dtype=dtype, device=DEV) * (0.02 if dtype.is_floating_point and dtype != torch.float32 else 1.0)


with torch.no_grad():
    # attention GEMMs (both bf16)
    w = {}
    w["wq_a"] = gen(args.dim, args.q_lora_rank).T.contiguous()
    w["wq_b"] = gen(args.q_lora_rank, args.n_heads * args.head_dim).T.contiguous()
    w["wkv"] = gen(args.dim, args.head_dim).T.contiguous()
    w["wo_b"] = gen(args.o_groups * args.o_lora_rank, args.dim).T.contiguous()
    src_attn.wq_a.weight.copy_(w["wq_a"]); ours.q_a.weight.copy_(w["wq_a"])
    src_attn.wq_b.weight.copy_(w["wq_b"]); ours.q_b.weight.copy_(w["wq_b"])
    src_attn.wkv.weight.copy_(w["wkv"]); ours.kv_proj.weight.copy_(w["wkv"])
    src_attn.wo_b.weight.copy_(w["wo_b"]); ours.wo_b.weight.copy_(w["wo_b"])
    # wo_a: source [o_groups*rank, group_span] == ours
    wo_a = gen(src_attn.wo_a.weight.shape).contiguous()
    src_attn.wo_a.weight.copy_(wo_a); ours.wo_a.weight.copy_(wo_a)
    # norms + sink (fp32 both)
    for nm, ours_nm in [("q_norm", "q_norm"), ("kv_norm", "kv_norm")]:
        v = torch.rand(getattr(src_attn, nm).weight.shape, dtype=torch.float32, device=DEV) + 0.5
        getattr(src_attn, nm).weight.copy_(v); getattr(ours, ours_nm).weight.copy_(v)
    sink = torch.randn(args.n_heads, dtype=torch.float32, device=DEV)
    src_attn.attn_sink.copy_(sink); ours.attn_sink.data.copy_(sink)

    def load_comp(scomp, ocomp, head_dim):
        ape = torch.randn(scomp.ape.shape, dtype=torch.float32, device=DEV) * 0.1
        scomp.ape.copy_(ape); ocomp.ape.copy_(ape)
        cwkv = torch.randn(scomp.wkv.weight.shape, dtype=torch.float32, device=DEV) * 0.02
        scomp.wkv.weight.copy_(cwkv); ocomp.wkv.weight.copy_(cwkv)
        cwg = torch.randn(scomp.wgate.weight.shape, dtype=torch.float32, device=DEV) * 0.02
        scomp.wgate.weight.copy_(cwg); ocomp.wgate.weight.copy_(cwg)
        cn = torch.rand(scomp.norm.weight.shape, dtype=torch.float32, device=DEV) + 0.5
        scomp.norm.weight.copy_(cn); ocomp.norm.weight.copy_(cn)

    load_comp(src_attn.compressor, ours.compressor, args.head_dim)
    ix, oix = src_attn.indexer, ours.indexer
    wq_ix = gen(args.q_lora_rank, ix.n_heads * ix.head_dim).T.contiguous()
    ix.wq_b.weight.copy_(wq_ix); oix.wq_b.weight.copy_(wq_ix)
    wp = gen(args.dim, ix.n_heads).T.contiguous()
    ix.weights_proj.weight.copy_(wp); oix.weights_proj.weight.copy_(wp)
    load_comp(ix.compressor, oix.compressor, ix.head_dim)

# ---- same input, run both ---------------------------------------------------
x = (torch.randn(1, S, args.dim, dtype=torch.bfloat16, device=DEV) * 0.5)

with torch.no_grad():
    out_src = src_attn(x, 0)
    out_ours = ours(x.squeeze(0)).unsqueeze(0)

a, b = out_src.float(), out_ours.float()
print(f"layer {LAYER} (ratio 4, indexer) s={S}, random bf16 weights both sides")
print(f"  source finite_frac = {torch.isfinite(a).float().mean().item():.4f}")
print(f"  ours   finite_frac = {torch.isfinite(b).float().mean().item():.4f}")
fin = torch.isfinite(a) & torch.isfinite(b)
if fin.any():
    d = (a - b).abs()[fin]
    rel = d / b[fin].abs().clamp_min(1e-3)
    print(f"  finite frac        = {fin.float().mean().item():.4f}")
    print(f"  max_abs_err        = {d.max().item():.6f}")
    print(f"  max_rel_err        = {rel.max().item():.6f}")
    print(f"  rtol0.2% match     = {(rel <= 0.002).float().mean().item():.4f}")
    print(f"  rtol2%   match     = {(rel <= 0.02).float().mean().item():.4f}")
