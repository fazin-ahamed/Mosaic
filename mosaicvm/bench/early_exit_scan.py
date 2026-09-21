"""Track S Stage-0 (roadmap §65-67): early-exit layer scan.

Zero-training oracle: run the kit model (source-class shims) on the G1
corpus prompts and, at every generated position, evaluate the REAL head
(hc_head + norm + lm_head) on every intermediate layer output.

Per (prompt, position, layer) record: exit top-16 token ids + logits.
Offline metrics per exit depth L:

  top1_acc     P(exit top-1 == actual next token)
  top4/8/16    coverage of the actual next token
  head_agree   P(exit top-1 == FINAL-layer top-1)
  multi-step   path survival from consecutive-position exit chains

Success metric is NOT top-1 alone: the E[A]/(T_draft+T_remaining) view
needs the cheap-head latency + reusable compute (recorded per exit as
head-GEMM time at batch 43; the drafter cost model comes later).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))

from source_corpus_gen import (build_kernel_module, StreamLinear,
                               FP4_TABLE)

OUT = Path("artifacts/early_exit_scan.json")
CORPUS = Path("artifacts/routes_overlap_corpus_3060.json")
N_GEN = 32


def main():
    snap = Path.home() / (
        ".cache/huggingface/hub/models--deepseek-ai--"
        "DeepSeek-V4-Flash-0731/snapshots/"
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    sys.path.insert(0, str(snap / "inference"))
    sys.modules["kernel"] = build_kernel_module()
    # fast_hadamard_transform shim (proven)
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
    import dataclasses
    cfg = json.loads((snap / "inference" / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(KM.ModelArgs)}
    args = KM.ModelArgs(**{k: v for k, v in cfg.items() if k in fields})
    args.temperature = 0.0
    args.max_batch_size = 1
    args.max_seq_len = 512
    args.dspark_block_size = 0
    args.n_mtp_layers = 0

    _orig_empty = torch.empty

    def _patch(*a, **k):
        if k.get("dtype") == torch.float4_e2m1fn_x2:
            k["device"] = "meta"
        return _orig_empty(*a, **k)

    torch.empty = _patch
    model = KM.Transformer(args)
    torch.empty = _orig_empty
    for L, layer in enumerate(model.layers):
        for e_idx, ex in enumerate(layer.ffn.experts):
            if ex is None:
                continue
            for nm in ("w1", "w3", "w2"):
                setattr(ex, nm, StreamLinear(
                    f"layers.{L}.ffn.experts.{e_idx}.{nm}"))
    import gc
    gc.collect()
    # selective move: embed + head on CPU
    for n, p in model.named_parameters():
        p.data = p.data.cpu() if n in ("embed.weight", "head.weight") \
            else p.data.cuda()
    for b in model.buffers():
        b.data = b.data.cuda()
    model.eval()

    # CPU embed/head forward patches (same as kit calibration runs)
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

    # ---- load core ----
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

    # ---- capture full per-layer last-position states ----
    N_LAYERS = len(model.layers)
    REC = {"positions": []}  # per forward: [43 x [4, 4096] cpu bf16]

    def layer_hook(mod, args_, output):
        h = output[0] if isinstance(output, tuple) else output
        REC["positions"].append(h[0, -1].detach().cpu())

    handles = [layer.register_forward_hook(layer_hook)
               for layer in model.layers]
    torch.set_default_device("cuda")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=True)
    corpus = json.loads(CORPUS.read_text())

    hc_head_fn = model.hc_head_fn            # fp32 CPU? -> to cpu once
    hc_head_base = model.hc_head_base
    hc_head_scale = model.hc_head_scale
    norm = model.norm
    head = model.head
    last_block = model.layers[-1]
    results = []
    t_head_total = 0.0

    for pe in corpus["prompts"]:
        cat = pe["category"]
        t0 = time.time()
        ids = tok(pe["prompt"], return_tensors="pt").input_ids.cuda()
        S = ids.shape[1]
        # prefill
        REC["positions"] = []
        with torch.no_grad():
            out_ids, logits, _ = model(ids, 0)
        states = list(REC["positions"])
        gen = [int(out_ids[0])]

        # decode
        for step in range(N_GEN - 1):
            REC["positions"] = []
            with torch.no_grad():
                out_ids, logits, _ = model(
                    torch.tensor([[gen[-1]]], device="cuda"), S + step)
            states += REC["positions"]
            gen.append(int(out_ids[0]))
        # reference: exit_top[p][N_LAYERS-1] (layer 42 through the same
        # head) IS the model's final logits — head(h) runs after the
        # last layer in Transformer.forward.

        # per-position: exit logits for all 43 layers (batched head)
        positions = len(gen)           # prefill + N-1 decodes
        assert len(states) == positions * N_LAYERS, \
            f"{len(states)} != {positions * N_LAYERS}"
        t_h0 = time.time()
        exit_top = []                   # [positions][43] top-16
        for p in range(positions):
            stack = torch.stack(
                [states[p * N_LAYERS + L].float() for L in
                 range(N_LAYERS)]).to("cpu")      # [43, 4, 4096] f32
            # hc_head: [43, 4, 4096] -> [43, 4096]
            xf = stack.reshape(43, -1)
            rsqrt = torch.rsqrt(
                xf.square().mean(-1, keepdim=True) + model.norm_eps)
            mixes = torch.nn.functional.linear(xf, hc_head_fn.cpu()) \
                * rsqrt
            pre = torch.sigmoid(
                mixes * hc_head_scale.cpu() + hc_head_base.cpu()) \
                + model.hc_eps
            y = torch.sum(
                pre.unsqueeze(-1) * stack.view(43, 4, 4096), dim=1)
            # norm + head (CPU, batched 43)
            yn = torch.nn.functional.rms_norm(
                y, (4096,), norm.weight.cpu())
            lg = torch.nn.functional.linear(yn, head.weight.cpu())
            tv, ti = lg.topk(16, dim=-1)
            exit_top.append((ti, tv))
        t_head_total += time.time() - t_h0

        # final-layer reference per position: exit_top[p][-1] IS the
        # final logits (layer 42 through the same head) — verify once:
        # actual next token at position p is gen[p+1] (p < positions-1)
        results.append({
            "category": cat, "S": S, "generated": gen,
            "exit_top16": [[[int(x) for x in
                             exit_top[p][0][L].tolist()]
                            for L in range(N_LAYERS)]
                           for p in range(positions)],
            "exit_top16_vals": [[[round(v, 3) for v in
                                  exit_top[p][1][L].tolist()]
                                 for L in range(N_LAYERS)]
                                for p in range(positions)],
        })
        # free-form sanity print
        acc = {}
        for L in range(0, N_LAYERS, 6):
            hits = sum(
                1 for p in range(positions)
                if exit_top[p][0][L][0].item() == gen[p])
            acc[L] = hits / positions
        print(f"[{cat}] {positions} pos; per-layer top1 acc "
              f"(L:acc): {acc} ({time.time()-t0:.0f}s)", flush=True)
        del states, exit_top, stack
        gc.collect()

    for h in handles:
        h.remove()

    # ---- offline metrics ----
    per_layer = {}
    for L in range(N_LAYERS):
        top1 = top4 = top8 = top16 = head_agree = 0
        n = 0
        for r in results:
            positions = len(r["generated"])
            for p in range(positions):
                ids16 = r["exit_top16"][p][L]
                nxt = r["generated"][p]
                final_ids = r["exit_top16"][p][N_LAYERS - 1]
                top1 += ids16[0] == nxt
                top4 += nxt in ids16[:4]
                top8 += nxt in ids16[:8]
                top16 += nxt in ids16[:16]
                head_agree += ids16[0] == final_ids[0]
                n += 1
        per_layer[L] = {"top1": round(top1 / n, 4),
                        "top4": round(top4 / n, 4),
                        "top8": round(top8 / n, 4),
                        "top16": round(top16 / n, 4),
                        "head_agree": round(head_agree / n, 4),
                        "n": n}

    # multi-step survival from exit chains (per layer, d=1..4)
    survival = {}
    for L in range(N_LAYERS):
        surv = {d: 0 for d in (1, 2, 3, 4)}
        trials = 0
        for r in results:
            positions = len(r["generated"])
            for p in range(positions - 4):
                trials += 1
                for d in (1, 2, 3, 4):
                    ok = all(
                        r["exit_top16"][p + j][L][0]
                        == r["generated"][p + j]
                        for j in range(d))
                    if ok:
                        surv[d] += 1
        survival[L] = {"trials": trials,
                       **{f"d{d}": round(surv[d] / trials, 4)
                          for d in (1, 2, 3, 4)}}

    out = {
        "classification": "EARLY_EXIT_STAGE0_SCAN (zero-training "
                          "oracle: real head on intermediate layers)",
        "model": "kit inference/model.py + torch shims, RTX 3060 "
                 "(CPU head/embed)",
        "corpus": "G1 certification prompts, 32 positions/prompt",
        "head_eval_time_s": round(t_head_total, 1),
        "per_layer": per_layer,
        "survival": survival,
    }
    OUT.write_text(json.dumps(out, indent=1))
    # summary table
    print("\n=== EARLY-EXIT STAGE-0 (real head on layer-L state) ===")
    print(f"{'L':>3}{'top1':>7}{'top4':>7}{'top8':>7}{'top16':>7}"
          f"{'agree':>7}{'d2':>7}{'d4':>7}")
    for L in range(N_LAYERS):
        pl = per_layer[L]
        sv = survival[L]
        print(f"{L:>3}{pl['top1']:>7.3f}{pl['top4']:>7.3f}"
              f"{pl['top8']:>7.3f}{pl['top16']:>7.3f}"
              f"{pl['head_agree']:>7.3f}{sv['d2']:>7.3f}{sv['d4']:>7.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
