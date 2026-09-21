"""Track P3: hot-bank sizing study from golden route traces.

Golden corpus: 301 records = 7 prompts x 43 layers, each with per-token
Top-6 expert ids. A verification block of width N at prompt offset o
touches union over 43 layers of expert_ids[layer][o:o+N].

Evaluates:
  - unique experts per block (N16/32/56/112)
  - frequency-policy hot-bank hit rate vs bank size (64-256)
  - H2D movement cost per block at 12.9 GB/s steady

Feeds the §50 hot-bank sizing race. CPU-only.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

GOLDEN = Path("artifacts/golden_routes.json")
OUT = Path("artifacts/hotbank_study.json")
N_LAYERS = 43
H2D_GB_S = 12.9
EXPERT_BYTES = 2048 * 2048 * 2 + 2048 * 128 * 2 + 4096 * 1024 + \
    4096 * 64          # packed4 w1+w3+w2 + scales ≈ 13.4 MB
BANK_SIZES = (48, 64, 96, 128, 160, 192, 256)


def main():
    g = json.loads(GOLDEN.read_text())
    recs = g["windows"]
    by_prompt = defaultdict(dict)
    for r in recs:
        by_prompt[r["prompt_idx"]][r["layer_name"]] = r["expert_ids"]
    # token length from the first layer record
    tok_len = {p: len(next(iter(layers.values())))
               for p, layers in by_prompt.items()}
    print(f"prompts: {len(by_prompt)}, token lengths: "
          f"{sorted(tok_len.values())}")

    # global frequency (all uses across all prompts/layers/tokens)
    freq = Counter()
    for layers in by_prompt.values():
        for ids in layers.values():
            for row in ids:
                freq.update(row)
    total_uses = sum(freq.values())
    top = [e for e, _ in freq.most_common()]

    rows = {}
    for N in (16, 32, 56, 112):
        blocks = []
        for p, layers in by_prompt.items():
            L = tok_len[p]
            for o in range(0, L - N + 1, max(1, N // 2)):
                u = set()
                for ids in layers.values():
                    for row in ids[o:o + N]:
                        u.update(row)
                blocks.append(u)
        if not blocks:
            continue  # capture length < N
        uniq = [len(b) for b in blocks]
        mean_uniq = sum(uniq) / len(uniq)
        row = {"n_blocks": len(blocks),
               "mean_unique_experts": round(mean_uniq, 1),
               "p50": sorted(uniq)[len(uniq) // 2],
               "max": max(uniq)}
        for bs in BANK_SIZES:
            hot = set(top[:bs])
            hits = sum(len(b & hot) for b in blocks)
            uses = sum(len(b) for b in blocks)
            hit_rate = hits / uses
            miss_experts = [len(b - hot) for b in blocks]
            mean_miss = sum(miss_experts) / len(miss_experts)
            move_ms = mean_miss * EXPERT_BYTES / 1e9 / H2D_GB_S * 1e3
            row[f"bank{bs}"] = {
                "use_hit_rate": round(hit_rate, 3),
                "mean_miss_experts": round(mean_miss, 1),
                "h2d_ms_per_block": round(move_ms, 1)}
        rows[f"N{N}"] = row
    out = {
        "classification": "HOTBANK_SIZING_STUDY",
        "source": "golden routes (real Blackwell routing, 7 prompts)",
        "expert_packed4_mb": round(EXPERT_BYTES / 1e6, 1),
        "h2d_steady_gbps": H2D_GB_S,
        "rows": rows,
        "note": "frequency policy from global usage; movement = "
                "miss_experts x 13.4MB / 12.9GB/s; LRU/per-layer "
                "policies and route-aware prefetch are the next race",
    }
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in rows.items():
        print(f"{k}: uniq={v['mean_unique_experts']} "
              f"bank128 hit={v['bank128']['use_hit_rate']} "
              f"miss={v['bank128']['mean_miss_experts']} "
              f"h2d={v['bank128']['h2d_ms_per_block']}ms")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
