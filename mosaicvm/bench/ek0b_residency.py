"""EK0b — cross-cycle temporal residency: LRU + static-hot + Belady oracle.

Corrected oracle (EK0b-R): the request stream is the ORDERED sequence of
(layer,expert) accesses across cycles; caching is simulated on that flat
stream so Belady is provably optimal and Belady <= LRU/static always holds.
Per-cycle miss counts are recovered by summing stream misses within each
cycle's access span. Includes a brute-force optimality assertion on small
traces.

Cache key = (layer, expert). Capacity in GB of packed-FP4 experts (13.37MB).
Movement-only tok/s upper bound = 3.757 / (core 17.5ms + misses*13.37MB/12.9GBs).
Absolute gate: <=103 cold experts/cycle (budget 125.2 - core 17.5 -> 107.7ms H2D).
"""
from __future__ import annotations

import json
import glob
from collections import defaultdict, OrderedDict, Counter
from pathlib import Path

import torch

BYTES = 13369344
H2D = 12.90e9
CORE_MS = 17.5
BUDGET_MS = 125.2
N_LAYERS = 43
W = 6
STRIDE = 4
CAPS_GB = [2, 4, 6, 8, 10]


def load_cycles():
    out = {}
    for f in sorted(glob.glob("artifacts/source_corpus/*.pt")):
        cat = Path(f).stem
        if cat == "manifest":
            continue
        c = torch.load(f, map_location="cpu", weights_only=False)
        pos = defaultdict(dict)
        for r in c["gates"]:
            if r["phase"] != "decode":
                continue
            idx = r["idx"]
            if idx and isinstance(idx[0], list):
                idx = idx[0]
            pos[r["step"]][r["layer"]] = list(idx)
        steps = [pos[s] for s in sorted(pos) if len(pos[s]) == N_LAYERS]
        cycles = []
        for start in range(0, len(steps) - W + 1, STRIDE):
            U = set()
            for pm in steps[start:start + W]:
                for L in range(N_LAYERS):
                    for e in pm.get(L, []):
                        U.add((L, e))
            cycles.append(sorted(U))
        out[cat] = cycles
    return out


def cycles_to_stream(cycles):
    """Flat ordered access stream + per-cycle (start,end) spans.
    Within a cycle, each distinct expert is accessed once (dedup union)."""
    stream = []
    spans = []
    for U in cycles:
        s = len(stream)
        stream.extend(U)
        spans.append((s, len(stream)))
    return stream, spans


def lru_stream(stream, cap):
    cache = OrderedDict(); miss = [False] * len(stream)
    for i, k in enumerate(stream):
        if k in cache:
            cache.move_to_end(k)
        else:
            miss[i] = True; cache[k] = True
            if len(cache) > cap:
                cache.popitem(last=False)
    return miss


def belady_stream(stream, cap):
    """Optimal demand-paging (always admit): on miss at capacity, evict the
    resident key whose NEXT use is furthest in the future."""
    n = len(stream)
    nxt = [n] * n
    last = {}
    for i in range(n - 1, -1, -1):
        k = stream[i]
        nxt[i] = last.get(k, n)
        last[k] = i
    cache = {}
    miss = [False] * n
    for i, k in enumerate(stream):
        if k in cache:
            cache[k] = nxt[i]
            continue
        miss[i] = True
        if len(cache) < cap:
            cache[k] = nxt[i]
        else:
            victim = max(cache, key=lambda kk: cache[kk])
            del cache[victim]
            cache[k] = nxt[i]
    return miss


def static_hot_stream(stream, cap):
    freq = Counter(stream)
    hot = set(k for k, _ in freq.most_common(cap))
    return [k not in hot for k in stream]


def per_cycle_miss(miss, spans):
    return [sum(miss[s:e]) for (s, e) in spans]


def brute_optimal(stream, cap):
    """Exhaustive optimal miss count for tiny traces (test only)."""
    from functools import lru_cache
    n = len(stream)

    @lru_cache(maxsize=None)
    def rec(i, cache):
        if i == n:
            return 0
        k = stream[i]
        cset = set(cache)
        if k in cset:
            return rec(i + 1, cache)
        if len(cset) < cap:
            return 1 + rec(i + 1, tuple(sorted(cset | {k})))
        best = None
        for v in cset:
            nc = tuple(sorted((cset - {v}) | {k}))
            r = 1 + rec(i + 1, nc)
            best = r if best is None else min(best, r)
        return best
    return rec(0, ())


def summ(pcm):
    import statistics as st
    ms = [CORE_MS + m * BYTES / H2D * 1e3 for m in pcm]
    tput = [3.757 / (t / 1e3) for t in ms]
    return {"mean_miss": round(st.mean(pcm), 1),
            "p90_miss": sorted(pcm)[int(0.9 * (len(pcm) - 1))],
            "max_miss": max(pcm),
            "mean_cold_GB": round(st.mean([m * BYTES / 1e9 for m in pcm]), 3),
            "mean_move_ms": round(st.mean(ms), 1),
            "mean_tput_UB": round(st.mean(tput), 2),
            "min_tput_UB": round(min(tput), 2)}


def test_belady():
    import random
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 9); cap = rng.randint(1, 4)
        stream = [(0, rng.randint(0, 5)) for _ in range(n)]
        b = sum(belady_stream(stream, cap))
        opt = brute_optimal(stream, cap)
        assert b == opt, f"Belady {b} != optimal {opt} on {stream} cap{cap}"
    print("Belady optimality test: PASS (200 random tiny traces)")


def main():
    test_belady()
    cycles = load_cycles()
    total = sum(len(v) for v in cycles.values())
    mean_union = sum(len(U) for v in cycles.values() for U in v) / total
    print(f"total cycles {total}, mean |U_c| {mean_union:.0f}")
    max_cold = int((BUDGET_MS - CORE_MS) / 1e3 * H2D / BYTES)
    print(f"absolute gate: <= {max_cold} cold experts/cycle\n")

    print(f"{'capGB':>5} {'slots':>6} {'policy':>8} {'meanMiss':>9} "
          f"{'p90':>5} {'coldGB':>7} {'moveMs':>7} {'tputUB':>7}")
    results = {"classification": "EK0B_RESIDENCY_R", "abs_max_cold": max_cold,
               "mean_union": round(mean_union, 1), "by_cap": {}}
    for cap_gb in CAPS_GB:
        cap = int(cap_gb * 1e9 / BYTES)
        agg = {"lru": [], "hot": [], "belady": []}
        for cat, cyc in cycles.items():   # per-domain reset (session)
            stream, spans = cycles_to_stream(cyc)
            agg["lru"] += per_cycle_miss(lru_stream(stream, cap), spans)
            agg["hot"] += per_cycle_miss(static_hot_stream(stream, cap), spans)
            agg["belady"] += per_cycle_miss(belady_stream(stream, cap), spans)
        rec = {}
        for pol in ("lru", "hot", "belady"):
            s = summ(agg[pol]); rec[pol] = s
            print(f"{cap_gb:>5} {cap:>6} {pol:>8} {s['mean_miss']:>9} "
                  f"{s['p90_miss']:>5} {s['mean_cold_GB']:>7} "
                  f"{s['mean_move_ms']:>7} {s['mean_tput_UB']:>7}")
        # valid invariant: demand-paging Belady <= demand LRU on same stream.
        # static-hot is a DIFFERENT model (fixed pinned set, global-freq
        # oracle), so it may beat per-cycle Belady on near-zero-reuse traffic.
        assert rec["belady"]["mean_miss"] <= rec["lru"]["mean_miss"] + 1e-6
        results["by_cap"][str(cap_gb)] = {"cap_slots": cap, **rec}

    # optimistic oracle = min of demand-Belady and static-hot pinned set
    best10 = min(results["by_cap"]["10"]["belady"]["mean_miss"],
                 results["by_cap"]["10"]["hot"]["mean_miss"])
    results["best_oracle_10GB_mean_miss"] = best10
    results["verdict"] = ("ORACLE_PASS" if best10 <= max_cold
                          else "GPU_ONLY_CLOSED")
    print(f"\nbest-oracle @10GB mean miss {best10} vs gate {max_cold} -> "
          f"{results['verdict']}")
    Path("artifacts/ek0b_residency.json").write_text(json.dumps(results, indent=1))
    print("wrote artifacts/ek0b_residency.json")


if __name__ == "__main__":
    main()
