"""FRONTIER-0 tier-1 screen: staged vs eager verification demand on the REAL
persisted route trace, against an EXPLICIT warm LRU cache (no compulsory-miss
subtraction).

Improvements over the synthetic demand experiment:
  - real captured routes (artifacts/ek0c_routes.pt), not random experts
  - real evolving LRU residency (not a fixed random 25%)
  - full byte accounting INCLUDING first touches; warm/cold windows split
  - acceptance drawn from the measured E[A]=3.757 (truncated-geometric, W<=6)

Honest limitations (stated, not hidden):
  - The trace is the ACCEPTED autoregressive path. It has no rejected
    speculative candidate branches. Window rows here reuse consecutive
    accepted-path routes as a proxy, which is OPTIMISTIC: real rejected
    candidates route to DIFFERENT experts and would raise cold demand.
  - This is a demand+cache SIMULATION, not latency or achieved throughput.
  - Horizon is 256 positions (~68 cycles); warm split is indicative only.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import torch

BYTES = 13369344
W = 6
STORE_GB = [30, 35, 38]
EA = 3.757

# truncated-geometric continue prob p s.t. sum_{i=0..W-1} p^i == EA
def fit_p():
    lo, hi = 0.0, 1.0
    for _ in range(80):
        p = (lo + hi) / 2
        if sum(p ** i for i in range(W)) < EA:
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.od = OrderedDict()

    def __contains__(self, k):
        return k in self.od

    def touch(self, k):
        if k in self.od:
            self.od.move_to_end(k)
        else:
            self.od[k] = 1
            if len(self.od) > self.cap:
                self.od.popitem(last=False)


def window_rows(routes, start, nlayers):
    """global (layer*256+expert) id sets for up to W positions from start."""
    rows = []
    for pos in range(start, min(start + W, len(routes))):
        s = set()
        pm = routes[pos]
        for L in range(nlayers):
            for e in pm.get(L, []):
                s.add(L * 256 + e)
        rows.append(s)
    return rows


POLICIES = {
    "eager6": [W],
    "frontier2_4": [2, 4],
    "frontier1_2_3": [1, 2, 3],
    "sequential": [1] * W,
}


def loaded_row_count(chunks, rows_needed, nrows):
    """rows loaded by a staged policy that stops once it covers rows_needed."""
    n = 0
    for c in chunks:
        n = min(n + c, nrows)
        if n >= rows_needed:
            break
    return n


def main():
    import random
    d = torch.load("artifacts/ek0c_routes.pt", weights_only=False)
    routes = d["routes"]
    nlayers = d["n_layers"]
    p = fit_p()
    rng = random.Random(20260920)

    out = {"classification": "FRONTIER0_SCREEN_REAL_ROUTES",
           "n_positions": len(routes), "accept_p": p, "EA": EA,
           "note": "real routes + warm LRU + measured acceptance; window rows "
                   "use accepted-path routes (OPTIMISTIC vs real rejected "
                   "branches). demand+cache sim only, not latency.",
           "by_store": {}}

    print(f"{'store':>5} {'policy':>13} {'coldAll':>8} {'coldWarm':>8} "
          f"{'GB/cyc':>7} {'rows':>5} {'cyc':>4}")
    for gb in STORE_GB:
        cap = int(gb * 1e9 / BYTES)
        rec = {}
        for name, chunks in POLICIES.items():
            cache = LRU(cap)
            cur = 0
            tot_cold = 0
            cyc = 0
            warm_cold = 0
            warm_cyc = 0
            rows_sum = 0
            while cur < len(routes) - 1:
                a = 1
                while a < W and rng.random() < p:
                    a += 1
                rows = window_rows(routes, cur, nlayers)
                if not rows:
                    break
                rows_needed = min(a + 1, len(rows))
                nload = loaded_row_count(chunks, rows_needed, len(rows))
                union = set()
                for r in rows[:nload]:
                    union |= r
                cold = sum(1 for k in union if k not in cache)
                tot_cold += cold
                rows_sum += nload
                cyc += 1
                if cur >= len(routes) // 2:      # second half = warmer
                    warm_cold += cold
                    warm_cyc += 1
                # commit accepted rows into cache
                for r in rows[:min(a, len(rows))]:
                    for k in r:
                        cache.touch(k)
                cur += a
            rec[name] = {
                "cold_all_mean": round(tot_cold / max(cyc, 1), 1),
                "cold_warm_mean": round(warm_cold / max(warm_cyc, 1), 1),
                "GB_per_cycle": round(tot_cold / max(cyc, 1) * BYTES / 1e9, 3),
                "rows_mean": round(rows_sum / max(cyc, 1), 2),
                "cycles": cyc,
            }
            r = rec[name]
            print(f"{gb:>5} {name:>13} {r['cold_all_mean']:>8} "
                  f"{r['cold_warm_mean']:>8} {r['GB_per_cycle']:>7} "
                  f"{r['rows_mean']:>5} {cyc:>4}")
        base = rec["eager6"]["cold_all_mean"]
        for name in rec:
            rec[name]["reduction_vs_eager_pct"] = round(
                100 * (1 - rec[name]["cold_all_mean"] / base), 1) if base else 0
        out["by_store"][str(gb)] = rec

    Path("artifacts/frontier0_screen.json").write_text(json.dumps(out, indent=1))
    print("\nwrote artifacts/frontier0_screen.json")
    print("reduction% (real routes, warm LRU) at 38GB:")
    for n, v in out["by_store"]["38"].items():
        print(f"  {n:>13}: {v['reduction_vs_eager_pct']}%  "
              f"({v['GB_per_cycle']} GB/cyc)")


if __name__ == "__main__":
    main()
