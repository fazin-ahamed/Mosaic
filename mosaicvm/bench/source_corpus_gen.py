"""G1 source-class reference corpus generator (roadmap §28-31).

Runs the CHECKPOINT'S OWN inference/model.py (from the HF snapshot) with
torch kernel shims replacing the tilelang kernels:

  act_quant         exact source math (per-128 amax -> pow2 E8M0 -> E4M3)
  fp8_gemm         kit arithmetic class: per-128-block code accumulation
                   then block scales (fp32; matches kernel block structure)
  fp4_gemm         kit arithmetic class: per-32-block code accumulation
                   (exact in fp32 for expert shapes) then act(128) x
                   weight(32) block scales
  sparse_attn      exact torch math (bf16 p cast, sink denominator)
  hc_split_sinkhorn exact torch math
  fp4_act_quant    exact torch math (DSpark only; unused here)

Experts: native packed FP4 + E8M0 scales streamed from the 48 shards
(GPU LRU). Core resident on GPU. Greedy temperature=0.

Records per prompt: greedy tokens, top-5 logits + margins, per-layer
Top-6 expert IDs + route weights (all 43 layers, hash + learned), and
per-layer hidden checksums (hc-mean) for every decode token.

Provenance: revision, core-archive cross-ref, script hash, hardware,
arithmetic-class documentation.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

import os
OUT_ROOT = Path(os.environ.get(
    "SCG_OUT",
    "/marimo/MosaicVM/mosaicvm/bench/rfix_out/source_corpus"))
N_GEN = int(os.environ.get("SCG_NGEN", "32"))
CATS_ENV = os.environ.get("SCG_CATS")  # comma filter for dry-runs
MEMDBG = os.environ.get("SCG_MEMDBG") == "1"


def memdbg(tag):
    if MEMDBG:
        torch.cuda.synchronize()
        print(f"    [mem] {tag}: alloc="
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
PROMPTS_ENV = os.environ.get("SCG_PROMPTS")  # alternate prompt file
CALIB = os.environ.get("SCG_CALIB") == "1"  # record router top-10
CPU_HEAD_EMBED = os.environ.get("SCG_CPU_HEAD_EMBED") == "1"
LAYERS_ENV = int(os.environ.get("SCG_LAYERS", "0"))    # dry-run only
EXPERTS_ENV = int(os.environ.get("SCG_EXPERTS", "0"))  # dry-run only
VOCAB_ENV = int(os.environ.get("SCG_VOCAB", "0"))      # dry-run only
EXPERT_CACHE_CAP = int(os.environ.get("SCG_CACHE_CAP", "3000"))

FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- shims ----
def build_kernel_module():
    k = type(sys)("kernel")

    def act_quant(x, block_size=128, scale_fmt=None,
                  scale_dtype=torch.float32, inplace=False):
        N = x.size(-1)
        z = x.contiguous().float()
        xb = z.view(*z.shape[:-1], N // block_size, block_size)
        amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        if scale_fmt is not None:  # ue8m0 pow2 (source)
            s = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
        else:
            s = amax / 448.0
        qf = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        if inplace:
            return x.copy_(
                (qf.float() * s).to(x.dtype).reshape(*x.shape))
        return qf.reshape(*z.shape), \
            s.reshape(*z.shape[:-1], N // block_size)

    def fp4_act_quant(x, block_size=32, inplace=False):
        N = x.size(-1)
        z = x.contiguous().float()
        xb = z.view(*z.shape[:-1], N // block_size, block_size)
        amax = xb.abs().amax(-1, keepdim=True).clamp_min(6 * 2 ** -126)
        s = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
        q = (xb / s).clamp(-6.0, 6.0)
        mid = (FP4_TABLE.to(q.device)[1:]
               + FP4_TABLE.to(q.device)[:-1]) / 2
        idx = torch.bucketize(q.abs(), mid)
        dq = q.sign() * FP4_TABLE.to(q.device)[idx] * s
        if inplace:
            return x.copy_(dq.to(x.dtype).reshape(*x.shape))
        return dq.to(torch.float8_e4m3fn).reshape(*z.shape), s

    def fp8_gemm(a, a_s, w, w_s, scale_dtype=None):
        """Kit fp8_gemm class: per-128-block code accumulation, then
        act x weight block scales (fp32)."""
        lead = a.shape[:-1]
        a = a.reshape(-1, a.shape[-1])
        a_s = a_s.reshape(a.shape[0], -1) if a_s.dim() > 2 else a_s
        M, K = a.shape
        N = w.shape[0]
        Npad = (N + 127) // 128 * 128
        wf = w.float()
        if Npad != N:  # partial out-block (not hit by real dims)
            wf = torch.nn.functional.pad(wf, (0, 0, 0, Npad - N))
        a_b = a.float().view(M, K // 128, 128)
        w_b = wf.view(Npad, K // 128, 128)
        P = torch.einsum('mgk,ngk->mng', a_b, w_b)  # [M, Npad, K//128]
        w_sE = w_s.float().repeat_interleave(128, dim=0)  # [>=N, K//128]
        y = (P * a_s.float()[:, None, :] * w_sE[None, :, :]).sum(2)
        return y[:, :N].reshape(*lead, N).to(torch.bfloat16)

    def fp4_gemm(a, a_s, w, w_s, scale_dtype=None):
        """Kit fp4_gemm class: FP8 act codes x FP4 weight codes, per-32
        block accumulation (exact in fp32 at expert shapes), then
        act(128) x weight(32) scales."""
        lead = a.shape[:-1]
        a = a.reshape(-1, a.shape[-1])
        a_s = a_s.reshape(a.shape[0], -1)
        M, K = a.shape
        N = w.shape[0]
        lut = FP4_TABLE.to(w.device)
        wb = w.view(torch.uint8).long()
        lo, hi = wb & 0x0F, wb >> 4
        w_c = torch.stack((lut[lo], lut[hi]), dim=-1) \
            .flatten(-2).float()                      # [N, K]
        a_b = a.float().view(M, K // 32, 32)
        w_b = w_c.view(N, K // 32, 32)
        P = torch.einsum('mgk,ngk->mng', a_b, w_b)   # [M, N, K//32]
        sA = a_s.float().repeat_interleave(4, dim=-1)  # [M, K//32]
        sW = w_s.float()                               # [N, K//32]
        y = (P * sA[:, None, :] * sW[None, :, :]).sum(2)
        return y.reshape(*lead, N).to(torch.bfloat16)

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        b, s, h, d = q.size()
        if not topk_idxs.is_cuda:
            topk_idxs = topk_idxs.to(q.device)
        valid = topk_idxs >= 0
        idxc = topk_idxs.clamp(min=0)
        g = kv.unsqueeze(1).expand(b, s, *kv.shape[1:]) \
            .gather(2, idxc.unsqueeze(-1).expand(-1, -1, -1, d))
        scores = torch.einsum("bshd,bstd->bsht", q.float(), g.float()) \
            * softmax_scale
        scores = torch.where(valid.unsqueeze(2), scores,
                             torch.full_like(scores, float("-inf")))
        m = scores.max(-1, keepdim=True).values
        p = torch.exp(scores - m)
        p_bf = p.to(torch.bfloat16).float()
        denom = p_bf.sum(-1) + torch.exp(
            attn_sink.view(1, 1, h).float() - m.squeeze(-1))
        o = torch.einsum("bsht,bstd->bshd", p_bf, g.float()) \
            / denom.unsqueeze(-1)
        return o.to(q.dtype)

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4,
                          sinkhorn_iters=20, eps=1e-6):
        hc = hc_mult
        pre_ = torch.sigmoid(
            mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
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

    k.act_quant = act_quant
    k.fp4_act_quant = fp4_act_quant
    k.fp8_gemm = fp8_gemm
    k.fp4_gemm = fp4_gemm
    k.sparse_attn = sparse_attn
    k.hc_split_sinkhorn = hc_split_sinkhorn
    return k


# ------------------------------------------------------- expert streaming ---
class StreamLinear(torch.nn.Module):
    """Routed-expert linear: packed FP4 + E8M0 scale read from the
    snapshot shards on demand, cached on GPU (class-level LRU)."""

    _cache: dict = {}
    _snap = None
    _index = None

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def _materialize(self):
        w = StreamLinear._cache.get(self.prefix)
        if w is None:
            from safetensors import safe_open
            shard = StreamLinear._index[self.prefix + ".weight"]
            with safe_open(str(StreamLinear._snap / shard),
                          framework="pt") as f:
                t = f.get_tensor(self.prefix + ".weight")
                s = f.get_tensor(self.prefix + ".scale")
            w = t.view(torch.float4_e2m1fn_x2).cuda()
            s = s.cuda()
            w.scale = s
            if len(StreamLinear._cache) >= EXPERT_CACHE_CAP:
                StreamLinear._cache.pop(next(iter(
                    StreamLinear._cache)))
            StreamLinear._cache[self.prefix] = w
        return w

    def forward(self, x):
        if os.environ.get("SCG_SKIP_EXPERTS") == "1":
            n_out = 4096 if self.prefix.endswith("w2") else 2048
            return torch.zeros(
                *x.shape[:-1], n_out, dtype=x.dtype, device=x.device)
        import model as KM
        return KM.linear(x, self._materialize(), None)


def main():
    t00 = time.time()
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(
        "deepseek-ai/DeepSeek-V4-Flash-0731"))
    log(f"snapshot: {snap}")
    sys.path.insert(0, str(snap / "inference"))
    sys.modules["kernel"] = build_kernel_module()
    # fast_hadamard_transform shim (Sylvester WHT, d^-0.5 — matches the
    # package the kit imports; proven in the 43-layer srcchain runs)
    import math as _math
    _fht = type(sys)("fast_hadamard_transform")
    _HAD = {}

    def _hadamard(x):
        d = x.shape[-1]
        assert d & (d - 1) == 0
        H = _HAD.get(d)
        if H is None or H.device != x.device:
            h = torch.tensor([[1.0]], device=x.device,
                             dtype=torch.float32)
            while h.shape[0] < d:
                h = torch.cat([torch.cat([h, h], 1),
                               torch.cat([h, -h], 1)])
            _HAD[d] = H = h / _math.sqrt(d)
        return (x.float() @ H).to(x.dtype)

    _fht.hadamard_transform = lambda x, scale=None: _hadamard(x)
    sys.modules["fast_hadamard_transform"] = _fht
    import model as KM  # the checkpoint's own code

    torch.set_default_dtype(torch.bfloat16)
    cfg = json.loads(
        (snap / "inference" / "config.json").read_text())
    snap_cfg = json.loads((snap / "config.json").read_text())
    mismatch = {k: (v, snap_cfg[k]) for k, v in cfg.items()
                if k in snap_cfg and snap_cfg[k] != v}
    assert not mismatch, f"config mismatch: {mismatch}"
    fields = {f.name for f in dataclasses.fields(KM.ModelArgs)}
    kwargs = {k: v for k, v in cfg.items() if k in fields}
    torch.manual_seed(33377335)

    args = KM.ModelArgs(**kwargs)
    args.temperature = 0.0
    args.max_batch_size = 1
    args.max_seq_len = 512
    args.dspark_block_size = 0
    args.n_mtp_layers = 0
    if LAYERS_ENV:
        args.n_layers = LAYERS_ENV
    if EXPERTS_ENV:
        args.n_routed_experts = EXPERTS_ENV
    if VOCAB_ENV:
        args.vocab_size = VOCAB_ENV
    # routed experts are STREAMED — keep their fp4 weights on meta
    # during build, drop them at replacement time
    _orig_empty = torch.empty

    def _empty_meta_q(*a, **k):
        if k.get("dtype") == torch.float4_e2m1fn_x2:
            k["device"] = "meta"
        return _orig_empty(*a, **k)
    # Build on CPU with routed-expert fp4 weights on meta (monkey-
    # patching torch.empty poisons `with torch.device(...)`/default-
    # device machinery on some torch builds, so the model is built on
    # CPU and moved after the experts are swapped out).
    torch.empty = _empty_meta_q
    model = KM.Transformer(args)
    torch.empty = _orig_empty

    # ---- replace routed experts with StreamLinear (drops the meta
    # fp4 weights + their e8m0 scales), then move the core to cuda ----
    for L, layer in enumerate(model.layers):
        moe = layer.ffn
        for e_idx, ex in enumerate(moe.experts):
            if ex is None:
                continue
            for nm in ("w1", "w3", "w2"):
                setattr(ex, nm, StreamLinear(
                    f"layers.{L}.ffn.experts.{e_idx}.{nm}"))
    for name, p in model.named_parameters():
        assert not p.is_meta, f"unreplaced meta param: {name}"
    import gc
    gc.collect()
    if CPU_HEAD_EMBED:
        # selective move: embed (1.1 GB bf16) + LM head (2.1 GB fp32)
        # stay on CPU (patched forward) so the core fits 11.6 GB
        for n, p in model.named_parameters():
            if n in ("embed.weight", "head.weight"):
                p.data = p.data.cpu()
            else:
                p.data = p.data.cuda()
        for b in model.buffers():
            b.data = b.data.cuda()
    else:
        model.to("cuda")
    model.eval()
    memdbg("after_move")
    log("built on cpu; experts streamed; core moved to cuda"
        + (" (embed+head on CPU)" if CPU_HEAD_EMBED else ""))

    # ---- load core weights (skip routed experts: streamed) ----
    index = json.loads(
        (snap / "model.safetensors.index.json").read_text())["weight_map"]
    StreamLinear._snap = snap
    StreamLinear._index = index
    params = dict(model.named_parameters())
    from safetensors import safe_open

    def dequant_wo_a(w, s):
        """convert.py serving transform: fp8 + 128x128 scale -> bf16."""
        rows, cols = w.shape
        bs = s.shape
        wf = w.float() * s.float() \
            .repeat_interleave(rows // bs[0], 0) \
            .repeat_interleave(cols // bs[1], 1)
        return wf.to(torch.bfloat16)

    index_shapes = {}
    for _sh in sorted(set(index.values())):
        with safe_open(str(snap / _sh), framework="pt") as _f:
            for _k in _f.keys():
                index_shapes[_k] = tuple(_f.get_slice(_k).get_shape())
    log(f"index shapes: {len(index_shapes)} tensors")
    loaded = skipped = 0
    t0 = time.time()
    for name, shard in index.items():
        if ".ffn.experts." in name and ".shared_experts." not in name:
            skipped += 1
            continue
        if name not in params:
            skipped += 1
            continue
        if tuple(params[name].shape) != tuple(
                index_shapes.get(name, params[name].shape)):
            # dry-run overrides only (vocab/layers/experts shrink):
            # sane-init the mismatched param so the mechanical path
            # still runs (never hit with the full config)
            p = params[name]
            with torch.no_grad():
                if p.dtype == torch.float8_e4m3fn:
                    b = torch.randint(0, 120, p.shape,
                                      dtype=torch.uint8, device=p.device)
                    p.copy_(b.view(torch.float8_e4m3fn))
                elif not p.is_floating_point():
                    p.copy_(torch.randint(
                        0, 8, p.shape, dtype=p.dtype, device=p.device))
                else:
                    p.copy_(torch.randn(
                        p.shape, device=p.device).to(p.dtype) * 0.02)
            skipped += 1
            continue
        with safe_open(str(snap / shard), framework="pt") as f:
            t = f.get_tensor(name)
            if name.endswith("wo_a.weight"):
                s = f.get_tensor(name.replace(".weight", ".scale"))
                t = dequant_wo_a(t, s)
        with torch.no_grad():
            params[name].data.copy_(t)
        loaded += 1
    memdbg("after_load")
    log(f"core loaded: {loaded} tensors ({skipped} skipped/streamed) "
        f"({time.time()-t0:.0f}s)")

    # sanity: all params finite
    n_bad = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.is_floating_point() and p.numel():
                if not torch.isfinite(p.float()).all():
                    n_bad += 1
                    log(f"  NON-FINITE: {name}")
    assert n_bad == 0, "non-finite params — fail closed"
    memdbg("after_finite")
    log("all params finite")
    # kit generate.py sets the default device to cuda for generation
    # (host-side arange/masks inside the kit land on cuda)
    torch.set_default_device("cuda")

    if CPU_HEAD_EMBED:
        # forward patches: embed/head compute on CPU, return cuda
        import types

        def _embed_fwd(self, x):
            return torch.nn.functional.embedding(
                x.cpu(), self.weight).to(x.device)

        def _head_fwd(self, x, full_logits=False):
            if not full_logits:
                x = x[:, -1]
            return torch.nn.functional.linear(
                x.cpu().float(), self.weight).to("cuda")

        model.embed.forward = types.MethodType(_embed_fwd, model.embed)
        model.head.forward = types.MethodType(_head_fwd, model.head)
        log("embed + head offloaded to CPU (display-GPU fit)")

    # ---- recording hooks ----
    REC = {"phase": "prefill", "step": -1}
    gate_recs = []
    hidden_recs = []

    def gate_hook(layer_idx):
        def h(mod, args, output):
            w, idx = output
            rec = {
                "layer": layer_idx, "phase": REC["phase"],
                "step": REC["step"],
                "idx": idx.detach().cpu().tolist(),
                "w": w.detach().float().cpu().tolist(),
            }
            if CALIB and not mod.hash:
                with torch.no_grad():
                    sc = KM.linear(args[0].float(), mod.weight.float())
                    sc = torch.nn.functional.softplus(sc).sqrt()
                    sc = sc + mod.bias.float()
                    tv, ti = sc[-1].topk(10)
                    rec["bias_top10_ids"] = ti.tolist()
                    rec["bias_top10_vals"] = [
                        round(v, 5) for v in tv.tolist()]
            gate_recs.append(rec)
        return h

    def layer_hook(layer_idx):
        def h(mod, args, output):
            if REC["phase"] == "decode":
                t = output[0] if isinstance(output, tuple) else output
                hidden_recs.append({
                    "layer": layer_idx, "step": REC["step"],
                    "h_mean": t[0, -1].float().mean(0).half().cpu(),
                })
        return h

    memdbg("after_hooks")
    handles = []
    for i, layer in enumerate(model.layers):
        handles.append(layer.ffn.gate.register_forward_hook(
            gate_hook(i)))
        handles.append(layer.register_forward_hook(layer_hook(i)))

    # ---- tokenizer + corpus prompts ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=True)
    corpus = json.loads(Path(PROMPTS_ENV or
                             "artifacts/routes_overlap_corpus_3060.json"
                             ).read_text())

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "revision": snap.name,
        "core_archive_sha_prefix": "a9089f25fa85bac9",
        "generator": "source_corpus_gen.py (kit inference/model.py + "
                     "torch kernel shims; fp4_gemm/fp8_gemm block-"
                     "factored code accumulation; act_quant exact)",
        "arithmetic_class": "source: E4M3 act per-128 E8M0 x "
                            "E2M1 packed per-32 E8M0 (experts), "
                            "E4M3 weights per-128 (core), bf16 wo_a",
        "hardware": torch.cuda.get_device_name(0),
        "temperature": 0.0,
        "n_gen": N_GEN,
        "prompts": [],
    }
    script_sha = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()[:16]
    manifest["script_sha"] = script_sha

    cat_count = {}
    for pe in corpus["prompts"]:
        cat = pe["category"]
        if CATS_ENV and cat not in CATS_ENV.split(","):
            continue
        gate_recs.clear()
        hidden_recs.clear()
        ids = tok(pe["prompt"], return_tensors="pt").input_ids.cuda()
        ids = ids.clamp(0, model.embed.weight.shape[0] - 1)
        S = ids.shape[1]
        t0 = time.time()

        memdbg("before_prefill")
        # prefill (kit returns out_ids [b], logits [b, vocab]:
        # head computes last-position logits only)
        REC["phase"] = "prefill"
        REC["step"] = -1
        with torch.no_grad():
            out_ids, logits, _ = model(ids, 0)
        topv, topi = logits[0].float().topk(5)
        top5s = [{"ids": topi.tolist(), "vals": topv.tolist()}]
        gen = [int(out_ids[0])]

        # decode
        for step in range(N_GEN - 1):
            REC["phase"] = "decode"
            REC["step"] = step
            with torch.no_grad():
                out_ids, logits, _ = model(
                    torch.tensor([[gen[-1]]], device="cuda"), S + step)
            topv, topi = logits[0].float().topk(5)
            top5s.append({"ids": topi.tolist(), "vals": topv.tolist()})
            gen.append(int(out_ids[0]))

        rec = {
            "category": cat,
            "prompt": pe["prompt"],
            "input_ids": ids[0].cpu().tolist(),
            "S": S,
            "generated": gen,
            "top5_per_token": top5s,
            "gates": gate_recs,
            "hiddens": hidden_recs,
        }
        n = cat_count.get(cat, 0)
        cat_count[cat] = n + 1
        fname = f"{cat}.pt" if CATS_ENV or not PROMPTS_ENV else \
            f"{cat}_{n}.pt"
        rec["file"] = fname
        torch.save(rec, OUT_ROOT / fname)
        manifest["prompts"].append({
            "category": cat, "S": S, "generated": gen,
            "file": fname})
        log(f"[{cat}] S={S} gen[:8]={gen[:8]} "
            f"gates={len(gate_recs)} hiddens={len(hidden_recs)} "
            f"({time.time()-t0:.0f}s)")

    for h in handles:
        h.remove()
    manifest["done"] = True
    (OUT_ROOT / "manifest.json").write_text(json.dumps(
        manifest, indent=1))
    log(f"=== SOURCE CORPUS DONE ({(time.time()-t00)/60:.0f} min) ===")
    log(f"generated: {[p['generated'][:4] for p in manifest['prompts']]}")


if __name__ == "__main__":
    main()
