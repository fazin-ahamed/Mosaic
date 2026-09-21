"""Re-validate + re-analyze the capture from raw records (no model needed).

Fixes:
  1. consecutive-position gate: compare the SET of positions, not the
     43x-duplicated list
  2. per-layer union stats: the whole-window union saturates at 256
     (16k assignments always touch every expert); the informative
     metric — comparable to historical U_med 86/91 — is the per-LAYER
     union within a window

Reads  routes_real_256_raw.json, rewrites frontier + md artifacts.
"""
import json
import os
import re
import sys
from collections import Counter

OUT = "/marimo/MosaicVM/artifacts"
WIDTHS = [64, 80, 96, 112, 124, 128, 144, 160]
N_LAYERS = 43
N_EXPERTS = 256


def log(m):
    print(f"[{__import__('time').strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    raw = json.load(open(os.path.join(OUT, "routes_real_256_raw.json")))
    records = raw["records"]
    meta = raw["metadata"]
    captures = raw["captures"]
    cats = [c["capture_id"] for c in captures]
    log(f"loaded {len(records)} records, {len(cats)} captures")

    # ---- corrected validation ----
    errors = []
    layers_seen = {r["layer"] for r in records}
    if len(layers_seen) != N_LAYERS:
        errors.append(f"layer count {len(layers_seen)} != {N_LAYERS}")
    for cid in cats:
        recs = [r for r in records if r["capture_id"] == cid]
        pos_set = sorted({r["pos"] for r in recs})
        if pos_set != list(range(pos_set[0], pos_set[-1] + 1)):
            errors.append(f"{cid}: non-consecutive positions")
        if len(pos_set) < 256:
            errors.append(f"{cid}: only {len(pos_set)} positions (<256)")
        per_pos_layers = {}
        for r in recs:
            per_pos_layers.setdefault(r["pos"], set()).add(r["layer"])
        bad = [p for p, ls in per_pos_layers.items() if len(ls) != N_LAYERS]
        if bad:
            errors.append(f"{cid}: {len(bad)} positions missing layers")
    for r in records[:2000] + records[-2000:]:
        if len(r["expert_ids"]) != 6:
            errors.append("expert count != 6")
            break
        if any(e < 0 or e >= N_EXPERTS for e in r["expert_ids"]):
            errors.append("expert id out of range")
            break
        if any(w != w or abs(w) == float("inf") for w in r["weights"]):
            errors.append("non-finite weight")
            break
    unique_sets = {tuple(sorted(r["expert_ids"])) for r in records}
    if len(unique_sets) <= 1:
        errors.append("routing collapse")
    validation = {
        "ok": len(errors) == 0, "errors": errors[:20],
        "n_records": len(records),
        "n_layers": len(layers_seen),
        "n_positions_per_capture": min(
            len({r["pos"] for r in records if r["capture_id"] == c})
            for c in cats),
        "unique_expert_sets": len(unique_sets),
    }
    log(f"validation ok={validation['ok']} "
        f"({validation.get('unique_expert_sets')} unique expert sets)")
    if errors:
        log(f"errors: {errors}")

    # ---- per-layer union analysis ----
    by_cappos = {}
    for r in records:
        by_cappos.setdefault(r["capture_id"], {}).setdefault(
            r["pos"], []).append(r)

    def es(rec):
        return set(rec["expert_ids"])

    layer_stats = {}      # width -> {layer -> union distribution}
    all_layers = {}       # width -> aggregated union distribution
    for w in WIDTHS:
        per_layer_unions = {}   # layer -> list of union sizes
        window_unions = []
        for cid in sorted(by_cappos):
            positions = sorted(by_cappos[cid])
            if len(positions) < w:
                continue
            for start in range(0, len(positions) - w + 1):
                window = positions[start:start + w]
                w_experts = set()
                pl = {}
                for p in window:
                    for rec in by_cappos[cid][p]:
                        e = es(rec)
                        w_experts |= e
                        pl.setdefault(rec["layer"], set()).update(e)
                window_unions.append(len(w_experts))
                for layer, exp in pl.items():
                    per_layer_unions.setdefault(layer, []).append(len(exp))
        # aggregate stats
        agg = []
        for layer, unions in sorted(per_layer_unions.items()):
            u = sorted(unions)
            n = len(u)
            agg.append({
                "layer": layer, "n_windows": n,
                "min": u[0], "p50": u[n // 2],
                "p90": u[int(n * 0.9)] if n > 1 else u[-1],
                "max": u[-1],
                "mean": round(sum(u) / n, 1),
            })
        # median across layers of the p50
        p50s = sorted(a["p50"] for a in agg)
        p90s = sorted(a["p90"] for a in agg)
        all_layers[str(w)] = {
            "per_layer": agg,
            "p50_of_p50": p50s[len(p50s) // 2] if p50s else None,
            "p50_of_p90": p90s[len(p90s) // 2] if p90s else None,
            "window_union_stats": {
                "min": min(window_unions) if window_unions else None,
                "max": max(window_unions) if window_unions else None,
            },
        }
        log(f"N{w}: per-layer p50 (median across layers) = "
            f"{all_layers[str(w)]['p50_of_p50']}, "
            f"p50_of_p90={all_layers[str(w)]['p50_of_p90']}")

    # ---- expert frequency (per layer: top-10 share) ----
    freq_total = Counter()
    per_layer_freq = {}
    for r in records:
        for e in r["expert_ids"]:
            freq_total[e] += 1
        per_layer_freq.setdefault(r["layer"], Counter()).update(r["expert_ids"])
    total = sum(freq_total.values())
    top10 = freq_total.most_common(10)
    # per-layer concentration
    layer_gini = {}
    for layer, cnt in per_layer_freq.items():
        vals = sorted(cnt.values())
        n = len(vals)
        if n and sum(vals):
            cum = sum((i + 1) * v for i, v in enumerate(vals))
            layer_gini[layer] = round(
                2 * cum / (n * sum(vals)) - (n + 1) / n, 4)
    gini_sorted = sorted(layer_gini.values())
    freq_analysis = {
        "total_assignments": total,
        "unique_experts_used": len(freq_total),
        "top10": [{"expert": e, "count": c,
                   "share": round(c / total, 4)} for e, c in top10],
        "top10_share": round(sum(c for _, c in top10) / total, 4),
        "top20_share": round(
            sum(c for _, c in freq_total.most_common(20)) / total, 4),
        "per_layer_gini_median": gini_sorted[len(gini_sorted) // 2]
        if gini_sorted else None,
        "per_layer_gini_min": gini_sorted[0] if gini_sorted else None,
        "per_layer_gini_max": gini_sorted[-1] if gini_sorted else None,
    }
    log(f"frequency: {freq_analysis['unique_experts_used']} unique experts, "
        f"top10 share {freq_analysis['top10_share']}, "
        f"median layer gini {freq_analysis['per_layer_gini_median']}")

    # ---- neighboring overlap ----
    overlaps = []
    for cid in by_cappos:
        by_layer = {}
        for p in by_cappos[cid]:
            for rec in by_cappos[cid][p]:
                by_layer.setdefault(rec["layer"], []).append(rec)
        for layer, recs in by_layer.items():
            recs.sort(key=lambda r: r["pos"])
            for i in range(len(recs) - 1):
                a, b = es(recs[i]), es(recs[i + 1])
                if a | b:
                    overlaps.append(len(a & b) / len(a | b))
    overlap_analysis = {
        "mean_jaccard": round(sum(overlaps) / len(overlaps), 4),
        "n_comparisons": len(overlaps),
    }
    log(f"neighboring overlap: mean jaccard "
        f"{overlap_analysis['mean_jaccard']}")

    # ---- rewrite artifacts ----
    meta["validation"] = validation
    frontier = json.load(open(os.path.join(OUT,
                                            "routes_real_256_frontier.json")))
    frontier["metadata"] = meta
    frontier["frequency"] = freq_analysis
    frontier["neighboring_overlap"] = overlap_analysis
    frontier["per_layer_union"] = all_layers
    fpath = os.path.join(OUT, "routes_real_256_frontier.json")
    with open(fpath, "w") as f:
        json.dump(frontier, f)
    # also update raw metadata
    raw["metadata"] = meta
    with open(os.path.join(OUT, "routes_real_256_raw.json"), "w") as f:
        json.dump(raw, f)

    # human summary
    lines = [
        "# Real ≥256-Position Route Capture Frontier (Blackwell sandbox)",
        "",
        f"Generated: {meta['timestamp']}",
        f"Model: deepseek-ai/DeepSeek-V4-Flash-0731 @ "
        f"{meta['model']['revision']}",
        f"GPU: {meta['environment']['gpu']} "
        f"({meta['environment']['vram_total_gib']} GiB)",
        "",
        "Classification: **BLACKWELL_SANDBOX_ONLY** — model-property "
        "capture,",
        "NOT RTX 3060 performance evidence.",
        "",
        f"Captures: {len(cats)} prompts × 260 positions (batched greedy "
        "decode)",
        f"Total records: {validation['n_records']} "
        f"({validation['n_positions_per_capture']} consecutive positions "
        "per capture × {n} layers)".replace("{n}", str(N_LAYERS)),
        f"Validation: **{'PASS' if validation['ok'] else 'FAIL'}**",
        "",
        "## Per-Layer Union Size (U) by Width — the frontier metric",
        "",
        "Window-union saturates at 256 (16k assignments always touch every",
        "expert). The meaningful metric is per-LAYER unique experts within",
        "a window — comparable to historical U_med 86 (K56) / 91 (K63).",
        "",
        "| Width | median-of-layer p50 | median-of-layer p90 |",
        "|---:|---:|---:|",
    ]
    for w in WIDTHS:
        s = all_layers[str(w)]
        lines.append(f"| {w} | {s['p50_of_p50']} | {s['p50_of_p90']} |")
    lines += [
        "",
        "## Per-Layer Union Detail (p50 per layer at N160)",
        "",
        "| Layer | p50 U | p90 U | mean U |",
        "|---:|---:|---:|---:|",
    ]
    for a in all_layers["160"]["per_layer"]:
        lines.append(f"| {a['layer']} | {a['p50']} | {a['p90']} | "
                     f"{a['mean']} |")
    lines += [
        "",
        "## Expert Frequency",
        "",
        f"- Total assignments: {freq_analysis['total_assignments']}",
        f"- Unique experts used: {freq_analysis['unique_experts_used']} / "
        f"{N_EXPERTS}",
        f"- Top-10 share: {freq_analysis['top10_share']:.1%}",
        f"- Top-20 share: {freq_analysis['top20_share']:.1%}",
        f"- Per-layer Gini: median {freq_analysis['per_layer_gini_median']}"
        f" (min {freq_analysis['per_layer_gini_min']}, "
        f"max {freq_analysis['per_layer_gini_max']})",
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
        "- Raw records preserved: routes_real_256_raw.json (19.4 MB)",
        "- BranchPack conclusions downstream; raw routes remain "
        "scheduler-agnostic",
        "- Overlap corpus for 3060 numerical-equivalence check: "
        "routes_overlap_corpus_3060.json",
        "- Model-property evidence only; all timing = BLACKWELL_SANDBOX_ONLY",
    ]
    with open(os.path.join(OUT, "routes_real_256_frontier.md"), "w") as f:
        f.write("\n".join(lines))
    log("rewrote frontier.json + frontier.md")
    log(f"=== RE-VALIDATION {'PASS' if validation['ok'] else 'FAIL'} ===")
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
