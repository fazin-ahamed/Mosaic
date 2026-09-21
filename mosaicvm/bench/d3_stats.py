"""Canonical proposer robust-statistics harness (roadmap S17).

One single implementation of every reported statistic. No
hand-coded alternatives.

Definitions (frozen):
  mean            arithmetic mean of per-prompt E[A]
  median          median
  trim20          20% trimmed mean: drop floor(0.2*n) samples
                  from EACH tail (n=21 -> drop 4 each side)
  loo_min/max     leave-one-out mean after removing the
                  best/worst single prompt
  boot_ci95       percentile bootstrap over prompts (seeded)
  S_k             survival probability P(>= k speculative
                  tokens accepted), k = 1..K — INDIVIDUAL values
  M_deep          deep-survival MASS sum_{k=4}^{K} S_k
                  (NEVER labeled S4/S4+)
  accepted        prompts with E[A] > 1
  deepest         max accepted speculative tokens
  per-domain      all of the above grouped by domain

Inputs: any d3_*_eval.json with rows[mode]['E_A'] and
rows[mode]['Sk'], or an A2/B/C-style _cl.json with
rows[i][mode]['E_A']/['Sk'] (mode key per file family).
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

K = 32


def row_vals(rows, mode):
    """Extract per-prompt E[A] and Sk rows for a mode, handling
    both the bridge-eval schema (rows[mode]) and the
    d3_v1_eval schema (rows[i][mode])."""
    out = []
    for r in rows:
        d = r[mode] if mode in r else r
        out.append({"cat": r["category"],
                    "E_A": d["E_A"], "Sk": d["Sk"]})
    return out


def trim20(vals):
    n = len(vals)
    k = int(0.2 * n)
    s = sorted(vals)
    return statistics.mean(s[k:n - k]) if n > 2 * k else None


def boot_ci(vals, seed=104729, reps=2000):
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n
                   for _ in range(reps))
    return [means[49], means[1949]]


def stats(rows, mode, k=K):
    rs = row_vals(rows, mode)
    vals = [r["E_A"] for r in rs]
    n = len(vals)
    loo_min = (sum(vals) - max(vals)) / (n - 1)
    loo_max = (sum(vals) - min(vals)) / (n - 1)
    # individual survival probabilities S_1..S_K
    Sk = [sum(1 for r in rs if len(r["Sk"]) > i and r["Sk"][i])
          / n for i in range(k)]
    dom = {}
    for r in rs:
        dom.setdefault(r["cat"].rsplit("_", 1)[0],
                       []).append(r["E_A"])
    return {
        "n": n,
        "mean": round(statistics.mean(vals), 4),
        "median": statistics.median(vals),
        "trim20": round(trim20(vals), 4),
        "loo_min": round(loo_min, 4),
        "loo_max": round(loo_max, 4),
        "boot_ci95": [round(x, 4) for x in boot_ci(vals)],
        "S1": round(Sk[0], 4), "S2": round(Sk[1], 4),
        "S3": round(Sk[2], 4), "S4": round(Sk[3], 4),
        "S8": round(Sk[7], 4) if k > 7 else None,
        "S16": round(Sk[15], 4) if k > 15 else None,
        "M_deep": round(sum(Sk[3:]), 4),
        "accepted": sum(1 for v in vals if v > 1),
        "deepest": max(int(v) - 1 for v in vals),
        "per_domain_mean": {d: round(statistics.mean(v), 3)
                            for d, v in sorted(dom.items())},
        "per_prompt": {r["cat"]: r["E_A"] for r in rs},
    }


def main():
    files = sys.argv[1:]
    for f in files:
        d = json.load(open(f))
        rows = d["rows"]
        # detect schema
        if "aggregate" in d and "cl_loop" in d.get("aggregate", {}):
            modes = [m for m in ("one_step", "cl_loop", "tf_loop")
                     if "cl_loop" in d["aggregate"] or True]
        mode = "cl_loop"
        print(f"\n=== {f} ({mode}) ===")
        print(json.dumps(stats(rows, mode), indent=1))


if __name__ == "__main__":
    main()
