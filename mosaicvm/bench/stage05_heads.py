"""Track S Stage-0.5 (roadmap §47-53): trained deep-exit heads.

Phase 1 (record): run the kit model on the Stage-0.5 prompts
(train/val/held-out, 7 domains each), capture FULL last-position
states [4, 4096] fp16 at exits L36/L38/L40/L42 + the actual next
token per position.

Phase 2 (train): per exit, train the lightweight hc_head-form mixing
(params: fn [4, 16384] + base [4] + scale [1] ≈ 65K) against the
FROZEN lm_head (bf16, GPU) with cross-entropy on the next token.
Baseline = the source hc_head mixing (untrained) on the same data.

Phase 3 (eval): held-out top-1/4/8/16 and d2/d4 path survival,
trained vs raw.

Design note: single-token heads are a representation PROBE. If they
improve strongly, the next step is multi-token heads from one state
(§50-51), NOT a sequential L40 drafter.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))

from source_corpus_gen import build_kernel_module, StreamLinear

PROMPTS = Path("artifacts/stage05_prompts.json")
OUT = Path("artifacts/stage05_heads_result.json")
DATA = Path("/tmp/stage05_data")
EXITS = (36, 38, 40, 42)
N_GEN = 32
HIDDEN = 4096
HC = 4


def build_model():
    import dataclasses
    snap = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--"
        "DeepSeek-V4-Flash-0731/snapshots/"
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    sys.path.insert(0, str(snap / "inference"))
    sys.modules["kernel"] = build_kernel_module()
    import math as _math
    _fht = type(sys)("fast_hadamard_transform")
    _HAD = {}

    def _had(x):
        d = x.shape[-1]
        H = _HAD.get(d)
        if H is None or H.device != x.device:
            h = torch.tensor([[1.0]], device=x.device,
                             dtype=torch.float32)
            while h.shape[0] < d:
                h = torch.cat([torch.cat([h, h], 1),
                               torch.cat([h, -h], 1)])
            _HAD[d] = H = h / _math.sqrt(d)
        return (x.float() @ H).to(x.dtype)

    _fht.hadamard_transform = lambda x, scale=None: _had(x)
    sys.modules["fast_hadamard_transform"] = _fht
    import model as KM
    torch.set_default_dtype(torch.bfloat16)
    cfg = json.loads((snap / "inference" / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(KM.ModelArgs)}
    args = KM.ModelArgs(**{k: v for k, v in cfg.items() if k in fields})
    args.temperature = 0.0
    args.max_batch_size = 1
    args.max_seq_len = 512
    args.dspark_block_size = 0
    args.n_mtp_layers = 0
    _orig = torch.empty

    def _patch(*a, **k):
        if k.get("dtype") == torch.float4_e2m1fn_x2:
            k["device"] = "meta"
        return _orig(*a, **k)

    torch.empty = _patch
    model = KM.Transformer(args)
    torch.empty = _orig
    for L, layer in enumerate(model.layers):
        for e_idx, ex in enumerate(layer.ffn.experts):
            if ex is None:
                continue
            for nm in ("w1", "w3", "w2"):
                setattr(ex, nm, StreamLinear(
                    f"layers.{L}.ffn.experts.{e_idx}.{nm}"))
    import gc
    gc.collect()
    for n, p in model.named_parameters():
        p.data = p.data.cpu() if n in ("embed.weight", "head.weight") \
            else p.data.cuda()
    for b in model.buffers():
        b.data = b.data.cuda()
    model.eval()
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

    from safetensors import safe_open
    index = json.loads((snap / "model.safetensors.index.json")
                       .read_text())["weight_map"]
    StreamLinear._snap = snap
    StreamLinear._index = index
    params = dict(model.named_parameters())

    def dequant_wo_a(w, s):
        rows, cols = w.shape
        return (w.float() * s.float()
                .repeat_interleave(rows // s.shape[0], 0)
                .repeat_interleave(cols // s.shape[1], 1)
                ).to(torch.bfloat16)

    for name, shard in index.items():
        if ".ffn.experts." in name and ".shared_experts." not in name:
            continue
        if name not in params:
            continue
        with safe_open(str(snap / shard), framework="pt") as f:
            t = f.get_tensor(name)
            if name.endswith("wo_a.weight"):
                s = f.get_tensor(name.replace(".weight", ".scale"))
                t = dequant_wo_a(t, s)
        with torch.no_grad():
            params[name].data.copy_(t)
    print("core loaded", flush=True)
    return model, snap


def record():
    model, snap = build_model()
    N_LAYERS = len(model.layers)
    REC = {"cur": []}
    keep = set(EXITS)

    def hook(L):
        def h(mod, args_, output):
            if L in keep:
                t = output[0] if isinstance(output, tuple) else output
                REC["cur"].append((L, t[0, -1].detach().half().cpu()))
        return h

    handles = [model.layers[L].register_forward_hook(hook(L))
               for L in EXITS]
    torch.set_default_device("cuda")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=True)
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    DATA.mkdir(exist_ok=True)
    for pe in prompts:
        cat = pe["category"]
        t0 = time.time()
        ids = tok(pe["prompt"], return_tensors="pt").input_ids.cuda()
        S = ids.shape[1]
        REC["cur"] = []
        with torch.no_grad():
            out_ids, _, _ = model(ids, 0)
        states = dict(REC["cur"])            # {L: [4,4096] fp16}
        gen = [int(out_ids[0])]
        for step in range(N_GEN - 1):
            REC["cur"] = []
            with torch.no_grad():
                out_ids, _, _ = model(
                    torch.tensor([[gen[-1]]], device="cuda"), S + step)
            for L, st in REC["cur"]:
                # accumulate position axis
                if L in states:
                    states[L] = torch.cat([states[L].float(),
                                           st.float()]).half()
                else:
                    states[L] = st
            gen.append(int(out_ids[0]))
        # positions axis: states[L] is [positions, 4, 4096]
        torch.save({"gen": gen, "states": states,
                    "prompt": pe["prompt"]}, DATA / f"{cat}.pt")
        print(f"[{cat}] recorded {len(gen)} positions "
              f"({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()


def load_split(cats=None):
    prompts = json.loads(PROMPTS.read_text())
    splits = prompts["splits"]
    if cats is None:
        cats = splits["train"]

    def load(cats):
        xs = {L: [] for L in EXITS}
        ys = []
        for cat in cats:
            d = torch.load(DATA / f"{cat}.pt", weights_only=False)
            g = d["gen"]
            for L in EXITS:
                # recorded as [4*pos, 4096] (pos-major rows);
                # reshape to [pos, 4, 4096]. position p predicts g[p]
                st = d["states"][L].float().view(-1, 4, 4096)
                xs[L].append(st[:len(g)])
            ys += g
        return ({L: torch.cat(v) for L, v in xs.items()},
                torch.tensor(ys))

    tr = load(splits["train"])
    va = load(splits["val"])
    ho = load(splits["held_out"])
    return tr, va, ho


def eval_head(fn, base, scale, X, head_w, topk=(1, 4, 8, 16)):
    """X: [N, 4, 4096] fp32 (GPU). Returns top-k accuracy dict."""
    with torch.no_grad():
        xf = X.reshape(X.shape[0], -1)
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        mixes = torch.nn.functional.linear(xf, fn) * rsqrt
        pre = torch.sigmoid(mixes * scale + base) + 1e-6
        y = torch.sum(pre.unsqueeze(-1) * X, dim=1)
        logits = torch.nn.functional.linear(y.to(torch.bfloat16), head_w)
        tv, ti = logits.float().topk(16, dim=-1)
    return tv, ti


def train_exit(L, tr, va, ho, head_w, src_hc):
    dev = "cuda"
    Xtr, ytr = tr[0][L].to(dev), tr[1].to(dev)
    Xva, yva = va[0][L].to(dev), va[1].to(dev)
    Xho, yho = ho[0][L].to(dev), ho[1].to(dev)

    # baseline: the frozen source hc_head mixing
    fn0 = src_hc["fn"].clone().to(dev)
    base0 = src_hc["base"].clone().to(dev)
    scale0 = src_hc["scale"].clone().to(dev)

    def accs(fn, base, scale, X, y):
        tv, ti = eval_head(fn, base, scale, X, head_w)
        pred1 = ti[:, 0]
        out = {"top1": (pred1 == y).float().mean().item()}
        for k in (4, 8, 16):
            out[f"top{k}"] = (ti[:, :k] == y.unsqueeze(1)).any(
                dim=1).float().mean().item()
        return out

    base_acc = accs(fn0, base0, scale0, Xho, yho)

    # train
    fn = torch.nn.Parameter(fn0.detach().clone())
    base = torch.nn.Parameter(base0.detach().clone())
    scale = torch.nn.Parameter(scale0.detach().clone())
    opt = torch.optim.Adam([fn, base, scale], lr=3e-3)
    N = Xtr.shape[0]
    bs = 64
    best_va, best = -1, None
    for step in range(400):
        idx = torch.randint(0, N, (bs,), device=dev)
        Xb, yb = Xtr[idx], ytr[idx]
        xf = Xb.reshape(bs, -1)
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        mixes = torch.nn.functional.linear(xf, fn) * rsqrt
        pre = torch.sigmoid(mixes * scale + base) + 1e-6
        y = torch.sum(pre.unsqueeze(-1) * Xb, dim=1)
        logits = torch.nn.functional.linear(y.to(torch.bfloat16),
                                            head_w).float()
        loss = torch.nn.functional.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 24:
            with torch.no_grad():
                va_acc = accs(fn.detach(), base.detach(), scale.detach(),
                              Xva, yva)["top1"]
            if va_acc > best_va:
                best_va = va_acc
                best = (fn.detach().clone(), base.detach().clone(),
                        scale.detach().clone())
    if best is None:
        best = (fn.detach(), base.detach(), scale.detach())
    trained_acc = accs(*best, Xho, yho)
    # d2/d4 survival on held-out (consecutive positions per prompt)
    # approximate: within-prompt consecutive position chains
    d2 = d4 = trials = 0
    ho_cats = json.loads(PROMPTS.read_text())["splits"]["held_out"]
    for cat in ho_cats:
        d = torch.load(DATA / f"{cat}.pt", weights_only=False)
        g = d["gen"]
        st = d["states"][L].float().view(-1, 4, 4096).to(dev)
        pos = len(g)
        tv, ti = eval_head(*best, st, head_w)
        for p in range(pos - 4):
            trials += 1
            if all(ti[p + j, 0].item() == g[p + j] for j in range(2)):
                d2 += 1
            if all(ti[p + j, 0].item() == g[p + j] for j in range(4)):
                d4 += 1
    return {
        "exit": L,
        "raw": base_acc,
        "trained": trained_acc,
        "val_top1": round(best_va, 4),
        "d2_trained": round(d2 / trials, 4),
        "d4_trained": round(d4 / trials, 4),
        "trials": trials,
        "params": int(fn.numel() + base.numel() + scale.numel()),
    }


def train():
    model, snap = build_model()
    src_hc = {"fn": model.hc_head_fn.float().cpu(),
              "base": model.hc_head_base.float().cpu(),
              "scale": model.hc_head_scale.float().cpu()}
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    from safetensors.torch import safe_open
    import os, glob
    snap_p = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--"
        "DeepSeek-V4-Flash-0731/snapshots/*"))[0]
    idx = json.load(open(os.path.join(
        snap_p, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(snap_p, idx["head.weight"]),
                   framework="pt") as f:
        head_w = f.get_tensor("head.weight").to("cuda")  # bf16 1.06GB
    tr, va, ho = load_split()
    results = []
    for L in (38, 40):
        r = train_exit(L, tr, va, ho, head_w, src_hc)
        results.append(r)
        print(json.dumps(r, indent=1), flush=True)
    out = {
        "classification": "STAGE05_TRAINED_EXIT_HEADS",
        "architecture": "hc_head-form mixing (65,541 params) + frozen "
                        "lm_head (bf16 GPU); CE on next token; Adam "
                        "3e-3, 400 steps, batch 64; best-of-val",
        "splits": "7 train / 7 val / 7 held-out prompts (disjoint)",
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record()
    elif sys.argv[1] == "train":
        train()
