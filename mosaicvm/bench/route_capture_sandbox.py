"""Real >=256-position route capture on the Blackwell sandbox (BATCHED).

All 7 prompts decode simultaneously as one batch — CPU-offload cost
amortizes, total time ~= single-prompt decode.

Loads DeepSeek-V4-Flash-0731 via transformers (device_map=auto,
GPU+CPU split), hooks DeepseekV4TopKRouter and DeepseekV4HashRouter on
all 43 layers, runs batched greedy decode with left-padding + KV cache,
records per-position/per-layer routing.

Writes to /marimo/MosaicVM/artifacts/:
  routes_real_256_raw.json       — full raw records + metadata
  routes_real_256_frontier.json  — per-width sliding-window analysis
  routes_real_256_frontier.md    — human summary
  routes_overlap_corpus_3060.json — small corpus for 3060 equivalence check

All timing/memory numbers are BLACKWELL_SANDBOX_ONLY (model-property
capture, NOT RTX 3060 performance evidence).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time

OUT_DIR = "/marimo/MosaicVM/artifacts"
N_POSITIONS = 260          # generate 260 -> >=259 routed decode positions
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
WIDTHS = [64, 80, 96, 112, 124, 128, 144, 160]
N_LAYERS = 43
N_EXPERTS = 256
TOPK = 6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download("deepseek-ai/DeepSeek-V4-Flash-0731")
    revision = os.path.basename(model_dir)
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }
    try:
        env["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"], capture_output=True,
            text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    log(f"env: {env}")
    log(f"model revision: {revision}")

    log("loading tokenizer + model (device_map=auto)...")
    tok = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True,
        padding_side="left")          # left-pad for batched decode
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    os.makedirs("/tmp/offload", exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=cfg, torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "88GiB", "cpu": "100GiB"},
        offload_folder="/tmp/offload",
        trust_remote_code=True)
    model.eval()
    from collections import Counter
    devs = Counter(str(p.device) for p in model.parameters())
    log(f"LOADED in {time.time()-t_start:.0f}s, param devices: {dict(devs)}")

    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4HashRouter, DeepseekV4TopKRouter)

    # ---- router hooks (batch-aware) ----
    # state carries per-row metadata for the CURRENT forward:
    #   batch_captures[i]  capture_id for row i (None = padding, skip)
    #   batch_positions[i] absolute position for row i
    #   batch_tokens[i]    token id for row i
    #   phase              "prefill" | "decode"
    state = {"phase": "prefill", "batch_captures": [],
             "batch_positions": [], "batch_tokens": [], "records": []}
    handles = []
    layer_modules = model.model.layers
    for li, layer in enumerate(layer_modules):
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is None or not isinstance(
                gate, (DeepseekV4HashRouter, DeepseekV4TopKRouter)):
            continue

        def make_hook(layer_idx):
            def hook(mod, args, kwargs, output):
                if state["phase"] != "decode":
                    return                # only decode records
                logits, weights, indices = output
                n_tok = weights.shape[0]
                for row in range(n_tok):
                    cid = (state["batch_captures"][row]
                           if row < len(state["batch_captures"]) else None)
                    if cid is None:
                        continue          # padding row — skip
                    state["records"].append({
                        "capture_id": cid,
                        "pos": state["batch_positions"][row],
                        "token": state["batch_tokens"][row],
                        "layer": layer_idx,
                        "router_class": ("hash"
                                         if isinstance(mod, DeepseekV4HashRouter)
                                         else "topk"),
                        "expert_ids": indices[row].tolist(),
                        "weights": weights[row].tolist(),
                        "phase": state["phase"],
                    })
            return hook

        handles.append(gate.register_forward_hook(
            make_hook(li), with_kwargs=True))
    log(f"hooked {len(handles)} routers")

    # ---- tokenize + left-pad the prompt batch ----
    texts = [p for _, p in PROMPTS]
    cats = [c for c, _ in PROMPTS]
    enc = tok(texts, return_tensors="pt", padding=True)
    input_ids = enc.input_ids.to(model.device)
    attn_mask = enc.attention_mask.to(model.device)
    n_batch, n_pad_len = input_ids.shape
    prompt_lens = attn_mask.sum(dim=1).tolist()
    log(f"batch {n_batch}x{n_pad_len} (prompt lens {prompt_lens})")

    # ---- batched prefill ----
    t0 = time.time()
    with torch.no_grad():
        state["phase"] = "prefill"
        # per-row metadata: position = index within the padded row
        # (absolute positions per capture restart at 0; padding masked out)
        state["batch_captures"] = list(cats)
        state["batch_positions"] = list(range(n_pad_len))
        state["batch_tokens"] = input_ids[0].tolist()   # placeholder
        out = model(input_ids, attention_mask=attn_mask, use_cache=True)
        past = out.past_key_values
        next_toks = out.logits[:, -1, :].argmax(dim=-1)   # [n_batch]

    # prefill routing deliberately NOT captured — the frontier uses the
    # >=256 consecutive decode positions only
    log(f"prefill done in {time.time()-t0:.0f}s")

    # ---- batched greedy decode ----
    t0 = time.time()
    # generated[c][step] is the token routed at position n_pad_len+step;
    # index 0 = first token from prefill logits
    generated = {c: [next_toks[i].item()] for i, c in enumerate(cats)}
    state["phase"] = "decode"
    for step in range(N_POSITIONS):
        state["batch_captures"] = list(cats)
        state["batch_positions"] = [n_pad_len + step] * n_batch
        state["batch_tokens"] = next_toks.tolist()
        with torch.no_grad():
            out = model(next_toks.unsqueeze(1),
                        attention_mask=None,
                        past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_toks = out.logits[:, -1, :].argmax(dim=-1)
        for i, c in enumerate(cats):
            generated[c].append(next_toks[i].item())
        if step % 50 == 0:
            rate = (step + 1) / (time.time() - t0)
            log(f"  step {step}/{N_POSITIONS} ({rate:.2f} tok/s)")
    log(f"decode done in {time.time()-t0:.0f}s")

    all_records = state["records"]
    log(f"total records: {len(all_records)}")

    # ---- sanity gates ----
    log("validating...")
    errors = []
    layers_seen = {r["layer"] for r in all_records}
    if len(layers_seen) != N_LAYERS:
        errors.append(f"layer count {len(layers_seen)} != {N_LAYERS}")
    for cid in cats:
        recs = [r for r in all_records if r["capture_id"] == cid]
        poss = sorted(r["pos"] for r in recs)
        if poss and poss != list(range(poss[0], poss[-1] + 1)):
            errors.append(f"{cid}: non-consecutive positions "
                          f"({len(poss)} recs)")
        if len(recs) < 256:
            errors.append(f"{cid}: only {len(recs)} decode records (<256)")
        # every position must have all 43 layers
        per_pos_layers = {}
        for r in recs:
            per_pos_layers.setdefault(r["pos"], set()).add(r["layer"])
        bad = [p for p, ls in per_pos_layers.items() if len(ls) != N_LAYERS]
        if bad:
            errors.append(f"{cid}: {len(bad)} positions missing layers")
    for r in all_records:
        if len(r["expert_ids"]) != TOPK:
            errors.append(f"pos {r['pos']} L{r['layer']}: "
                          f"{len(r['expert_ids'])} experts != {TOPK}")
            break
        if any(e < 0 or e >= N_EXPERTS for e in r["expert_ids"]):
            errors.append(f"pos {r['pos']} L{r['layer']}: expert out of range")
            break
        if any(w != w or abs(w) == float("inf") for w in r["weights"]):
            errors.append(f"pos {r['pos']} L{r['layer']}: non-finite weight")
            break
    unique_sets = {tuple(sorted(r["expert_ids"])) for r in all_records}
    if len(unique_sets) <= 1:
        errors.append("routing collapse")
    validation = {
        "ok": len(errors) == 0, "errors": errors[:20],
        "n_records": len(all_records),
        "n_layers": len(layers_seen),
        "n_decode_records": len(all_records),
    }
    log(f"validation: ok={validation['ok']}, "
        f"{validation['n_records']} records")
    if errors:
        log(f"ERRORS: {errors}")

    # ---- frontier analysis (sliding windows over decode positions) ----
    log("frontier analysis...")
    by_cappos = {}
    for r in all_records:
        by_cappos.setdefault(r["capture_id"], {}).setdefault(
            r["pos"], []).append(r)

    def expert_set(rec):
        return set(rec["expert_ids"])

    frontier = {}
    for w in WIDTHS:
        unions = []
        windows_detail = []
        for cid in sorted(by_cappos):
            positions = sorted(by_cappos[cid])
            if len(positions) < w:
                continue
            for start in range(0, len(positions) - w + 1):
                window = positions[start:start + w]
                w_experts = set()
                per_layer = {}
                for p in window:
                    for rec in by_cappos[cid][p]:
                        es = expert_set(rec)
                        w_experts |= es
                        per_layer.setdefault(rec["layer"], set()).update(es)
                unions.append(len(w_experts))
                windows_detail.append({
                    "capture_id": cid, "start": window[0],
                    "end": window[-1], "union": len(w_experts),
                    "per_layer_union": {str(k): len(v)
                                        for k, v in sorted(per_layer.items())},
                })
        unions.sort()
        n = len(unions)
        stats = {
            "min": unions[0], "p50": unions[n // 2],
            "p90": unions[int(n * 0.9)] if n > 1 else unions[-1],
            "p95": unions[int(n * 0.95)] if n > 1 else unions[-1],
            "max": unions[-1],
            "mean": round(sum(unions) / n, 1),
        } if n else None
        frontier[str(w)] = {"n_windows": n, "windows": windows_detail,
                            "union_stats": stats}
        log(f"  N{w}: {n} windows, "
            + (f"p50={stats['p50']} p95={stats['p95']} max={stats['max']}"
               if stats else "no windows"))

    # expert frequency
    from collections import Counter
    freq = Counter()
    for r in all_records:
        for e in r["expert_ids"]:
            freq[e] += 1
    total = sum(freq.values())
    top10 = freq.most_common(10)
    top20 = freq.most_common(20)
    freq_analysis = {
        "total_assignments": total,
        "unique_experts_used": len(freq),
        "top10": [{"expert": e, "count": c,
                   "share": round(c / total, 4)} for e, c in top10],
        "top10_share": round(sum(c for _, c in top10) / total, 4),
        "top20_share": round(sum(c for _, c in top20) / total, 4),
    }

    # neighboring overlap
    overlaps = []
    for cid in by_cappos:
        by_layer = {}
        for p in by_cappos[cid]:
            for rec in by_cappos[cid][p]:
                by_layer.setdefault(rec["layer"], []).append(rec)
        for layer, recs in by_layer.items():
            recs.sort(key=lambda r: r["pos"])
            for i in range(len(recs) - 1):
                a, b = expert_set(recs[i]), expert_set(recs[i + 1])
                if a | b:
                    overlaps.append(len(a & b) / len(a | b))
    overlap_analysis = {
        "mean_jaccard": round(sum(overlaps) / len(overlaps), 4)
        if overlaps else 0,
        "n_comparisons": len(overlaps),
    }

    # ---- write artifacts ----
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata = {
        "timestamp": timestamp,
        "model": {
            "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": revision,
            "dtype": "bf16 activations, source FP8 weights "
                     "(HF finegrained FP8)",
        },
        "environment": env,
        "numerical_path": {
            "dtype": "bfloat16",
            "quantization": "source FP8 (transformers finegrained_fp8)",
            "attention": "transformers deepseek_v4 (source)",
            "router": "transformers DeepseekV4HashRouter/TopKRouter (source)",
            "deterministic": "greedy argmax, no sampling, batch decode",
            "classification": "BLACKWELL_SANDBOX_ONLY (model-property "
                              "capture; NOT RTX 3060 performance evidence)",
        },
        "test_baseline": "201/201 local (179 pre-sandbox + 22 route-capture)",
        "thermal": "UNKNOWN",
        "n_prompts": len(PROMPTS),
        "n_positions_generated": N_POSITIONS,
        "batch": {"size": n_batch, "padded_len": n_pad_len,
                  "prompt_lens": prompt_lens},
        "device_split": dict(devs),
        "validation": validation,
    }

    captures_meta = []
    for c, p in PROMPTS:
        gen_ids = generated[c]
        # decode records for this capture: positions n_pad_len..n_pad_len+259
        recs = [r for r in all_records if r["capture_id"] == c]
        # token for position n_pad_len+step is gen_ids[step] EXCEPT the last
        tokens_by_pos = {}
        for r in recs:
            tokens_by_pos[r["pos"]] = r["token"]
        captures_meta.append({
            "capture_id": c, "category": c, "prompt": p,
            "prompt_sha256": hashlib.sha256(p.encode()).hexdigest()[:16],
            "input_ids": input_ids[cats.index(c)].tolist(),
            "generated_ids": gen_ids,
            "n_prompt_tokens": prompt_lens[cats.index(c)],
            "n_generated": len(gen_ids),
            "n_decode_records": len(recs),
            "seed": "greedy (deterministic argmax)",
            "generation": {"method": "greedy argmax batched",
                           "do_sample": False,
                           "max_new_tokens": N_POSITIONS},
        })

    raw = {"metadata": metadata, "captures": captures_meta,
           "records": all_records}
    raw_path = os.path.join(OUT_DIR, "routes_real_256_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw, f)
    log(f"wrote {raw_path} ({os.path.getsize(raw_path)/1e6:.1f} MB)")

    frontier_out = {
        "metadata": metadata,
        "frequency": freq_analysis,
        "neighboring_overlap": overlap_analysis,
        "widths": frontier,
    }
    fpath = os.path.join(OUT_DIR, "routes_real_256_frontier.json")
    with open(fpath, "w") as f:
        json.dump(frontier_out, f)
    log(f"wrote {fpath}")

    lines = [
        "# Real ≥256-Position Route Capture Frontier (Blackwell sandbox)",
        "",
        f"Generated: {timestamp}",
        f"Model: deepseek-ai/DeepSeek-V4-Flash-0731 @ {revision}",
        f"GPU: {env['gpu']} ({env['vram_total_gib']} GiB)",
        "Classification: BLACKWELL_SANDBOX_ONLY — model-property capture,",
        "NOT RTX 3060 performance evidence.",
        "",
        f"Captures: {len(captures_meta)} prompts, "
        f"{N_POSITIONS} positions each (batched decode)",
        f"Total records: {validation['n_records']}",
        f"Validation: {'PASS' if validation['ok'] else 'FAIL'}",
        "",
        "## Union Size (U) by Width — sliding windows, decode positions only",
        "",
        "| Width | Windows | Min | p50 | p90 | p95 | Max | Mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for w in WIDTHS:
        s = frontier[str(w)]["union_stats"]
        if s:
            lines.append(f"| {w} | {frontier[str(w)]['n_windows']} | "
                         f"{s['min']} | {s['p50']} | {s['p90']} | "
                         f"{s['p95']} | {s['max']} | {s['mean']} |")
    lines += [
        "",
        "## Expert Frequency",
        "",
        f"- Total assignments: {freq_analysis['total_assignments']}",
        f"- Unique experts used: {freq_analysis['unique_experts_used']} "
        f"/ {N_EXPERTS}",
        f"- Top-10 share: {freq_analysis['top10_share']:.1%}",
        f"- Top-20 share: {freq_analysis['top20_share']:.1%}",
        "",
        "## Neighboring-Position Overlap",
        "",
        f"- Mean Jaccard (consecutive positions, same layer): "
        f"{overlap_analysis['mean_jaccard']}",
        "",
        "## Notes",
        "",
        "- Genuine consecutive positions; no corpus wrapping",
        "- Greedy deterministic batched decode (left-padded prompts)",
        "- Raw records preserved for recomputation",
        "- BranchPack conclusions downstream; raw routes remain "
        "scheduler-agnostic",
        "- Model-property evidence only; timing = BLACKWELL_SANDBOX_ONLY",
    ]
    mpath = os.path.join(OUT_DIR, "routes_real_256_frontier.md")
    with open(mpath, "w") as f:
        f.write("\n".join(lines))
    log(f"wrote {mpath}")

    overlap_corpus = {
        "purpose": "rerun these on RTX 3060 and compare tokens/routes/"
                   "weights for numerical-equivalence validation",
        "model_revision": revision,
        "prompts": [{"category": c, "prompt": p,
                     "expected_generated_ids": generated[c][:32]}
                    for c, p in PROMPTS],
        "compare": ["generated token ids", "expert_ids per position/layer",
                    "route weights per position/layer"],
    }
    opath = os.path.join(OUT_DIR, "routes_overlap_corpus_3060.json")
    with open(opath, "w") as f:
        json.dump(overlap_corpus, f, indent=1)
    log(f"wrote {opath}")

    log(f"=== DONE in {(time.time()-t_start)/60:.1f} min ===")
    log(f"validation: {'PASS' if validation['ok'] else 'FAIL'}")
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
