"""Track D1-B: L41/L42 tail robustness to approximate h40 trajectories.

THE decisive proposer measurement: how much target-token quality
survives when the h40 trajectory fed to the real source tail is
approximate?

Method (per prompt):
  Phase 1: run the full kit model; capture the TRUE h40 trajectory
  (all positions), true h42, true L41/L42 routes (gate hooks), and
  the generated token sequence (greedy).
  Phase 2: for each error family x level:
    - perturb h40 at SPECULATIVE positions only (S..n_pos-1);
      committed prefix (0..S-1) stays exact (production semantics)
    - reset L41/L42 KV/DSA state, replay the whole trajectory
      through the real source tail (prefill batch + sequential)
    - fp32 head on replayed h42 -> logits
    - measure: Top-k vs true next token, route overlap, h42 cosine,
      acceptance survival S_k and E[A]
  Root position S-1 is never perturbed -> root-token reuse check.

Error families:
  gauss      isotropic Gaussian, per-position exact rel-L2
  direction  norm-preserving rotation (orthogonal perturbation)
  scale      pure norm error, direction preserved (1+level)
  ar1        temporally correlated drift (rho=0.9), trajectory-level

Levels (relative L2): 1e-4, 1e-3, 5e-3, 2e-2, 1e-1, 3e-1 (+ exact)

Note: the tail interface is bf16 — perturbations below bf16
resolution are partially absorbed by the cast. We record both
pre-cast and post-cast rel-L2.
"""
from __future__ import annotations

import json
import math
import os
import hashlib
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

N_TEST = int(os.environ.get("D1B_N_TEST", "21"))
HORIZONS = tuple(int(k) for k in os.environ.get("D1B_HORIZONS", "8,16,32").split(","))
N_STEPS = int(os.environ.get("D1B_N_STEPS", str(max(HORIZONS))))
if N_TEST < 1 or min(HORIZONS) < 1 or N_STEPS < max(HORIZONS):
    raise ValueError("Need positive prompt count/horizons and N_STEPS >= max horizon")
LEVELS = [float(x) for x in os.environ.get(
    "D1B_LEVELS", "1e-4,1e-3,5e-3,2e-2,1e-1,3e-1").split(",")]
FAMILIES = os.environ.get(
    "D1B_FAMILIES", "gauss,direction,scale,ar1").split(",")
OUT = Path(os.environ.get(
    "D1B_OUT", "artifacts/d1b32_tail_robustness_v2.json"))
TOPKS = (1, 2, 4, 8, 16)
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prefix_metrics(correct, horizon):
    """K uncertain draft decisions, excluding exact root; prefix+1 emission.

    Input must come from ONE exact committed boundary. Sliding crops of
    an already perturbed replay are not independent production cycles.
    """
    if horizon < 1 or len(correct) < horizon:
        raise ValueError("Need at least K speculative decisions")
    depth = 0
    for hit in correct[:horizon]:
        if not hit:
            break
        depth += 1
    return {"K": horizon, "accepted_draft": depth, "E_A": depth + 1,
            "Sk": [int(depth >= k) for k in range(1, horizon + 1)]}


def prompt_mean_ci(values):
    """Percentile bootstrap over independent prompts, never sliding windows."""
    if len(values) < 2:
        return None
    rng = random.Random(104729)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                   for _ in range(2000))
    return [means[49], means[1949]]


def reset_decode_state(module):
    """Reset all nested source attention AND indexer compressor buffers."""
    for name, buf in module.named_buffers():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("kv_cache", "kv_state", "score_state"):
            buf.fill_(float("-inf") if leaf == "score_state" else 0)


def main():
    from source_corpus_gen import build_kernel_module, StreamLinear
    if os.environ.get("SCG_SKIP_EXPERTS") == "1":
        raise ValueError("D1-B requires all routed source experts")
    t0 = time.time()
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
    import dataclasses
    torch.set_default_dtype(torch.bfloat16)
    cfg = json.loads((snap / "inference" / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(KM.ModelArgs)}
    args = KM.ModelArgs(**{k: v for k, v in cfg.items() if k in fields})
    args.temperature = 0.0
    args.max_batch_size = 1
    args.max_seq_len = 256
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
    import types

    for n, p in model.named_parameters():
        p.data = p.data.cpu() if n in ("embed.weight",
                                        "head.weight") else p.data.cuda()
    for b in model.buffers():
        b.data = b.data.cuda()
    model.eval()

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
    print(f"model loaded {time.time()-t0:.0f}s", flush=True)
    torch.set_default_device("cuda")

    # Reuse the source HC head and norm, including their BF16 boundaries.
    # Only the final FP32 vocabulary projection runs on the CPU.
    def head_logits(h42):
        with torch.no_grad():
            h = model.layers[42].hc_head(
                h42.to(device="cuda", dtype=torch.bfloat16).unsqueeze(0),
                model.hc_head_fn, model.hc_head_scale, model.hc_head_base)
            h = model.norm(h).squeeze(0).cpu().float()
            return torch.nn.functional.linear(h, model.head.weight)

    # ---- capture machinery ----
    CAP = {"h40": [], "h42": []}
    ROUTE = {"on": False, "l41": [], "l42": []}

    def make_state_hook(key):
        def h(mod, args_, output):
            t = output[0] if isinstance(output, tuple) else output
            CAP[key].append(t.detach().float().cpu())
        return h

    def make_gate_hook(key):
        def h(mod, args_, output):
            if ROUTE["on"]:
                ROUTE[key].append(output[1].detach().cpu())
        return h

    hs = [
        model.layers[40].register_forward_hook(make_state_hook("h40")),
        model.layers[42].register_forward_hook(make_state_hook("h42")),
        model.layers[41].ffn.gate.register_forward_hook(
            make_gate_hook("l41")),
        model.layers[42].ffn.gate.register_forward_hook(
            make_gate_hook("l42")),
    ]

    blk41, blk42 = model.layers[41], model.layers[42]

    def replay_tail(h40_bf16, input_ids_all, S):
        """Replay [n_pos,4,4096] bf16 trajectory; returns h42 [n_pos,4,4096]."""
        reset_decode_state(blk41)
        reset_decode_state(blk42)
        n_pos = h40_bf16.shape[0]
        h40_dev = h40_bf16.cuda()
        outs = []
        with torch.no_grad():
            pre = h40_dev[:S].unsqueeze(0)
            h41_out = blk41(pre, 0, input_ids_all[:, :S])
            outs.append(blk42(h41_out, 0, input_ids_all[:, :S]))
            for p in range(S, n_pos):
                h41_p = blk41(h40_dev[p:p + 1].unsqueeze(0), p,
                              input_ids_all[:, p:p + 1])
                outs.append(blk42(h41_p, p, input_ids_all[:, p:p + 1]))
        return torch.cat(outs, dim=1).squeeze(0)

    # ---- prompts ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(snap),
                                        trust_remote_code=True)
    spec = json.loads(Path(
        "artifacts/stage1_prompts.json").read_text())
    test_cats = (os.environ["D1B_CATS"].split(",") if os.environ.get("D1B_CATS")
                 else spec["splits"]["test"][:N_TEST])
    if len(set(test_cats)) != len(test_cats) or not set(test_cats) <= set(spec["splits"]["test"]):
        raise ValueError("D1B_CATS must contain distinct held-out test categories")

    def perturb(h40, S, family, level, seed):
        """h40: [n_pos,4,4096] fp32 (true). Perturb positions >= S only."""
        g = torch.Generator().manual_seed(seed)
        out = h40.clone()
        n_dec = h40.shape[0] - S
        if n_dec <= 0 or level == 0.0:
            return out
        dec = h40[S:]                                  # [n_dec,4,4096]
        if family == "gauss":
            noise = torch.randn(dec.shape, generator=g, device="cpu", dtype=torch.float32)
            nrm = noise.flatten(1).norm(dim=1, keepdim=True)
            tgt = dec.flatten(1).norm(dim=1, keepdim=True)
            noise = noise * (level * tgt / (nrm + 1e-12)).view(-1, 1, 1)
        elif family == "scale":
            noise = dec * level                        # direction kept
        elif family == "direction":
            gv = torch.randn(dec.shape, generator=g, device="cpu", dtype=torch.float32)
            dv = dec.flatten(1)
            gn = gv.flatten(1)
            # orthogonal component
            proj = (gn * dv).sum(-1, keepdim=True) / \
                (dv.square().sum(-1, keepdim=True) + 1e-12)
            orth = (gn - proj * dv)
            orth = orth / (orth.norm(dim=1, keepdim=True) + 1e-12)
            theta = 2.0 * math.asin(min(level, 2.0) / 2.0)
            newv = math.cos(theta) * dv + math.sin(theta) * orth * \
                dv.norm(dim=1, keepdim=True)
            noise = (newv.view_as(dec) - dec)
        elif family == "ar1":
            rho = 0.9
            noise = torch.randn(dec.shape, generator=g, device="cpu", dtype=torch.float32)
            for t in range(1, noise.shape[0]):
                noise[t] = rho * noise[t - 1] + \
                    math.sqrt(1 - rho * rho) * \
                    torch.randn(dec.shape[1:], generator=g, device="cpu", dtype=torch.float32)
            # trajectory-level scaling: mean per-position rel L2 = level
            rel = noise.flatten(1).norm(dim=1) / \
                (dec.flatten(1).norm(dim=1) + 1e-12)
            noise = noise * (level / (rel.mean() + 1e-12))
        else:
            raise ValueError(family)
        out[S:] = dec + noise
        return out

    all_rows = []
    for cat in test_cats:
        pi = spec["splits"]["test"].index(cat)  # seed stable across subsets/order
        pe = next(p for p in spec["prompts"]
                  if p["category"] == cat)
        ids = tok(pe["prompt"], return_tensors="pt").input_ids.cuda()
        S = ids.shape[1]
        if S + N_STEPS > args.max_seq_len:
            raise ValueError(f"{cat}: prefix + horizon exceeds max_seq_len")
        reset_decode_state(model)
        print(f"\n=== [{cat}] S={S} ===", flush=True)

        # ---- Phase 1: true trajectory ----
        for key in CAP:
            CAP[key].clear()
        ROUTE["on"] = True
        ROUTE["l41"].clear(); ROUTE["l42"].clear()
        h40_dec, h42_dec, r41d, r42d = [], [], [], []
        with torch.no_grad():
            out_ids, _, _ = model(ids, 0)
            gen = [int(out_ids[0])]
            h40_pref = CAP["h40"][0].clone()
            h42_pref = CAP["h42"][0].clone()
            r41 = [ROUTE["l41"][0].clone()]
            r42 = [ROUTE["l42"][0].clone()]
            for step in range(N_STEPS):
                for key in CAP:
                    CAP[key].clear()
                ROUTE["l41"].clear(); ROUTE["l42"].clear()
                cur = out_ids.reshape(1, 1) if out_ids.dim() == 1 \
                    else out_ids[:, -1:]
                out_ids, _, _ = model(cur, S + step)
                gen.append(int(out_ids[0]))
                if (step + 1) % 8 == 0:
                    print(f"reference {cat}: {step + 1}/{N_STEPS}", flush=True)
                h40_dec.append(CAP["h40"][0].clone())
                h42_dec.append(CAP["h42"][0].clone())
                r41d.append(ROUTE["l41"][0].clone())
                r42d.append(ROUTE["l42"][0].clone())
        ROUTE["on"] = False
        h40_all = torch.cat([h40_pref] + h40_dec, 1).squeeze(0)
        h42_true = torch.cat([h42_pref] + h42_dec, 1).squeeze(0)
        route41_true = torch.cat(r41 + r41d, 0)   # [n_pos, 6]
        route42_true = torch.cat(r42 + r42d, 0)
        all_tokens = ids[0].tolist() + gen
        n_pos = h40_all.shape[0]
        print(f"phase1 done n_pos={n_pos} "
              f"({time.time()-t0:.0f}s)", flush=True)

        # ground-truth correctness per position (vs true next token)
        # true next token is greedy-generated for p >= S-1
        input_ids_all = torch.tensor(
            [all_tokens[:n_pos]], device="cuda")
        h40_bf16_true = h40_all.to(torch.bfloat16)

        # gen contains the next token for the final captured state too.
        score_ps = list(range(S - 1, n_pos))
        spec_ps = list(range(S, n_pos))        # perturbed only

        def evaluate(h42_rep, routes41, routes42, tag):
            """Return metric dict for one replay."""
            lg = head_logits(h42_rep)             # [n_pos, V] cpu fp32
            top_vals, top_idx = lg.topk(32, dim=-1)
            pred_tok = top_idx[:, 0]
            correct = [bool(pred_tok[p].item() ==
                            all_tokens[p + 1])
                       for p in range(n_pos)]
            topk_hit = {k: 0 for k in TOPKS}
            for p in score_ps:
                tk = top_idx[p].tolist()
                if all_tokens[p + 1] in tk[:16]:
                    for k in TOPKS:
                        if all_tokens[p + 1] in tk[:k]:
                            topk_hit[k] += 1
            # spec-only curve (exclude root)
            topk_spec = {k: 0 for k in TOPKS}
            for p in spec_ps:
                tk = top_idx[p].tolist()
                for k in TOPKS:
                    if all_tokens[p + 1] in tk[:k]:
                        topk_spec[k] += 1
            # route overlap at decode positions
            def route_metrics(rt_true, rt_rep):
                ex, jac = [], []
                for p in range(S, n_pos):
                    a = set(rt_true[p].tolist())
                    b = set(rt_rep[p].tolist())
                    ex.append(a == b)
                    jac.append(len(a & b) / len(a | b))
                return (sum(ex) / len(ex), sum(jac) / len(jac))
            r41e, r41j = route_metrics(route41_true, routes41)
            r42e, r42j = route_metrics(route42_true, routes42)
            # h42 cosine at decode positions
            cos = []
            for p in range(S, n_pos):
                a = h42_true[p].flatten()
                b = h42_rep[p].flatten().float().cpu()
                cos.append(torch.nn.functional.cosine_similarity(
                    a.unsqueeze(0), b.unsqueeze(0)).clamp(-1, 1).item())
            horizons = {}
            for k in HORIZONS:
                metrics = prefix_metrics([correct[p] for p in spec_ps], k)
                metrics["topk"] = {
                    str(tk): sum(all_tokens[p + 1] in top_idx[p, :tk].tolist()
                                 for p in spec_ps[:k]) / k for tk in (1, 4, 16)}
                metrics["h42_final_rel_l2"] = float(
                    (h42_rep[S + k - 1].float().cpu() - h42_true[S + k - 1]).norm()
                    / h42_true[S + k - 1].norm().clamp_min(1e-12))
                horizons[str(k)] = metrics
            return {
                "root_top1": 1.0 if correct[S - 1] else 0.0,
                "spec_topk": {str(k): round(topk_spec[k] /
                                            max(len(spec_ps), 1), 4)
                              for k in TOPKS},
                "all_topk": {str(k): round(topk_hit[k] /
                                           max(len(score_ps), 1), 4)
                             for k in TOPKS},
                "route_exact_l41": round(r41e, 4),
                "route_jac_l41": round(r41j, 4),
                "route_exact_l42": round(r42e, 4),
                "route_jac_l42": round(r42j, 4),
                "h42_cos_p50": round(sorted(cos)[len(cos) // 2], 6),
                "h42_cos_min": round(min(cos), 6),
                "horizons": horizons,
                "spec_correct": [correct[p] for p in spec_ps],
            }

        # ---- level 0: exact replay sanity ----
        ROUTE["on"] = True
        ROUTE["l41"].clear(); ROUTE["l42"].clear()
        h42_rep = replay_tail(h40_bf16_true, input_ids_all, S)
        rt41 = torch.cat(ROUTE["l41"], 0)
        rt42 = torch.cat(ROUTE["l42"], 0)
        ROUTE["on"] = False
        m0 = evaluate(h42_rep, rt41, rt42, "exact")
        bit_exact = torch.equal(h42_rep.float().cpu(), h42_true)
        m0["h42_bit_exact"] = bit_exact
        ok0 = (bit_exact and m0["all_topk"]["1"] == 1.0 and
               m0["route_exact_l41"] == 1.0 and
               m0["route_exact_l42"] == 1.0)
        print(f"exact sanity: top1={m0['all_topk']['1']} "
              f"route41={m0['route_exact_l41']} "
              f"route42={m0['route_exact_l42']} "
              f"{'OK' if ok0 else 'BROKEN'}", flush=True)
        if not ok0:
            raise RuntimeError("Exact source tail/head replay failed sanity")

        all_rows.append({"prompt": cat, "family": "exact", "level": 0.0, **m0})
        capture_dir = os.environ.get("D1B_CAPTURE_DIR")
        if capture_dir:
            dest = Path(capture_dir)
            dest.mkdir(parents=True, exist_ok=True)
            torch.save({"category": cat, "split": "test", "prompt": pe["prompt"],
                        "S": S, "snapshot_revision": snap.name,
                        "h40": h40_bf16_true.cpu(), "h42": h42_true.to(torch.bfloat16),
                        "tokens": all_tokens, "routes41": route41_true,
                        "routes42": route42_true, "h42_bit_exact": True},
                       dest / f"{cat}.pt")

        # ---- perturbed replays ----
        for fam in FAMILIES:
            for lev in LEVELS:
                seed = 1000 * pi + 100 * FAMILIES.index(fam)
                h40_p = perturb(h40_all, S, fam, lev, seed)
                # pre/post-cast rel L2 at decode positions
                def rel_l2(a, b):
                    return ((a.flatten(1) - b.flatten(1))
                            .norm(dim=1) /
                            b.flatten(1).norm(dim=1))
                pre = rel_l2(h40_p[S:], h40_all[S:]).mean().item()
                h40_p_bf = h40_p.to(torch.bfloat16)
                post = rel_l2(h40_p_bf[S:].float(),
                              h40_all[S:]).mean().item()
                ROUTE["on"] = True
                ROUTE["l41"].clear(); ROUTE["l42"].clear()
                h42_rep = replay_tail(h40_p_bf, input_ids_all, S)
                rt41 = torch.cat(ROUTE["l41"], 0)
                rt42 = torch.cat(ROUTE["l42"], 0)
                ROUTE["on"] = False
                m = evaluate(h42_rep, rt41, rt42, f"{fam}@{lev}")
                if m["root_top1"] != 1.0:
                    raise RuntimeError("Exact committed root changed during perturbation")
                row = {"prompt": cat, "family": fam, "level": lev,
                       "rel_l2_pre": round(pre, 6),
                       "rel_l2_post": round(post, 6), **m}
                all_rows.append(row)
                print(f"{fam}@{lev:.0e}: pre={pre:.4f} "
                      f"post={post:.4f} "
                      f"top1={m['spec_topk']['1']} "
                      f"top4={m['spec_topk']['4']} "
                      f"top16={m['spec_topk']['16']} "
                      f"r41={m['route_exact_l41']} "
                      f"EA={[(k, v['E_A']) for k, v in m['horizons'].items()]} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # Keep completed boundaries recoverable during long source-reference runs.
        partial = OUT.with_suffix(".partial.json")
        partial.parent.mkdir(parents=True, exist_ok=True)
        temp = partial.with_suffix(".tmp")
        temp.write_text(json.dumps({"complete": False, "script_sha256": SCRIPT_SHA256,
                                   "prompts_requested": test_cats, "horizons": HORIZONS,
                                   "families": FAMILIES, "levels": LEVELS,
                                   "per_prompt": all_rows}, indent=1))
        temp.replace(partial)

    for h in hs:
        h.remove()

    # ---- aggregate across prompts (prompt-weighted mean; one committed boundary per prompt) ----
    def agg(family, level):
        rows = [r for r in all_rows
                if r["family"] == family and r["level"] == level]
        if not rows:
            return None
        out = {"n": len(rows)}
        for k in ("root_top1", "route_exact_l41", "route_jac_l41",
                  "route_exact_l42", "route_jac_l42",
                  "h42_cos_p50", "h42_cos_min"):
            out[k] = round(sum(r[k] for r in rows) / len(rows), 4)
        for tk in ("spec_topk", "all_topk"):
            out[tk] = {k2: round(sum(r[tk][k2] for r in rows) /
                                 len(rows), 4)
                       for k2 in rows[0][tk]}
        out["horizons"] = {}
        for k in HORIZONS:
            hr = [r["horizons"][str(k)] for r in rows]
            out["horizons"][str(k)] = {
                "n_boundaries": len(hr),
                "E_A": sum(r["E_A"] for r in hr) / len(hr),
                "Sk": [sum(r["Sk"][i] for r in hr) / len(hr) for i in range(k)],
                "topk": {tk: sum(r["topk"][tk] for r in hr) / len(hr)
                         for tk in ("1", "4", "16")},
                "per_prompt_E_A": [r["E_A"] for r in hr],
                "E_A_ci95_prompt_bootstrap": prompt_mean_ci([r["E_A"] for r in hr]),
            }
        if "rel_l2_pre" in rows[0]:
            out["rel_l2_pre"] = round(
                sum(r["rel_l2_pre"] for r in rows) / len(rows), 6)
            out["rel_l2_post"] = round(
                sum(r["rel_l2_post"] for r in rows) / len(rows), 6)
        return out

    curve = []
    curve.append({"family": "exact", "level": 0.0,
                  **agg("exact", 0.0)})
    for fam in FAMILIES:
        for lev in LEVELS:
            a = agg(fam, lev)
            if a:
                curve.append({"family": fam, "level": lev, **a})

    out = {
        "classification": "D1B32_SOURCE_TAIL_SYNTHETIC_BOUNDARY_PROBE",
        "schema_version": 2,
        "production_certified": False,
        "acceptance_semantics": "K speculative decisions exclude exact root; prefix+1",
        "conditioning": "true future h40 plus synthetic error; not real student acceptance",
        "snapshot_revision": snap.name,
        "perturbation_seed_policy": "1000*prompt_index + 100*family_index; matched across levels",
        "families": FAMILIES,
        "levels": LEVELS,
        "prompts": test_cats,
        "expert_cache_cap": os.environ.get("SCG_CACHE_CAP", "3000"),
        "script_sha256": SCRIPT_SHA256,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "method": "perturbed h40 trajectory (spec positions only) "
                  "-> real L41/L42 replay -> fp32 head -> Top-k / "
                  "routes / E[A]",
        "horizons": HORIZONS,
        "n_steps": N_STEPS,
        "per_prompt": all_rows,
        "curve": curve,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    OUT.with_suffix(".partial.json").unlink(missing_ok=True)
    print(f"\nwrote {OUT} ({time.time()-t0:.0f}s total)")

    for c in curve:
        print(c["family"], c["level"],
              {k: v["E_A"] for k, v in c["horizons"].items()})


if __name__ == "__main__":
    main()
