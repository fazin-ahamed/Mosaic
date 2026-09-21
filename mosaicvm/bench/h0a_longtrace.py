"""H0-A long-trace capacity: cumulative unique bytes + steady miss over time.

Consumes the persisted EK0c long single-session route trace
(artifacts/ek0c_routes.pt: [pos -> {layer: [6 ids]}]). Tracks, over W6/stride4
cycles from ONE session (no reset, no cross-domain union):
  - cumulative distinct (layer,expert) touched -> working-set growth curve
  - per-cycle capacity misses under a fixed store (LRU + Belady), separating
    compulsory first-touch from recurring capacity misses
  - whether the working set saturates below the combined store or keeps growing

This is the sustained-residency evidence H0-A's short trace lacked. Labeled
proxy (W6/stride4 vs real speculative verification).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.ek0b_residency import (
    cycles_to_stream, lru_stream, belady_stream, per_cycle_miss)

BYTES = 13369344
N_LAYERS = 43
W = 6
STRIDE = 4
STORE_GB = [24, 30, 35, 38]


def main():
    p = Path("artifacts/ek0c_routes.pt")
    if not p.exists():
        print("MISSING artifacts/ek0c_routes.pt — run ek0c_longstream first")
        raise SystemExit(1)
    d = torch.load(p, weights_only=False)
    routes = d["routes"]
    print(f"long trace: {len(routes)} decode positions")

    # cumulative working-set growth (single session, no reset)
    seen = set()
    growth = []
    for pm in routes:
        for L in range(N_LAYERS):
            for e in pm.get(L, []):
                seen.add((L, e))
        growth.append(len(seen))
    ws_gb = len(seen) * BYTES / 1e9
    print(f"final working set: {len(seen)} experts = {ws_gb:.2f} GB")
    # growth rate over last quarter (is it saturating?)
    q = len(growth) // 4
    late_rate = (growth[-1] - growth[-q]) / q if q else 0
    print(f"WS growth rate last quarter: {late_rate:.1f} new experts/position "
          f"(early: {growth[q]/q:.1f})")

    # cycles from single session
    cycles = []
    for start in range(0, len(routes) - W + 1, STRIDE):
        U = set()
        for pm in routes[start:start + W]:
            for L in range(N_LAYERS):
                for e in pm.get(L, []):
                    U.add((L, e))
        cycles.append(sorted(U))
    stream, spans = cycles_to_stream(cycles)

    out = {"classification": "H0A_LONGTRACE", "n_positions": len(routes),
           "working_set_experts": len(seen), "working_set_GB": round(ws_gb, 2),
           "late_growth_rate": round(late_rate, 2),
           "ws_growth_curve": growth[::max(1, len(growth)//20)],
           "by_store": {}}
    print(f"\n{'storeGB':>7} {'slots':>6} {'belMiss':>8} {'compulsory':>10} "
          f"{'capacity':>9}")
    # compulsory (infinite cache) per cycle
    inf = per_cycle_miss(belady_stream(stream, 10**9), spans)
    comp_mean = sum(inf) / len(inf)
    for gb in STORE_GB:
        slots = int(gb * 1e9 / BYTES)
        bel = per_cycle_miss(belady_stream(stream, slots), spans)
        bmean = sum(bel) / len(bel)
        cap = bmean - comp_mean
        out["by_store"][str(gb)] = {"slots": slots,
                                    "belady_miss": round(bmean, 1),
                                    "compulsory": round(comp_mean, 1),
                                    "capacity": round(cap, 1)}
        print(f"{gb:>7} {slots:>6} {bmean:>8.1f} {comp_mean:>10.1f} {cap:>9.1f}")

    # verdict: does WS fit a plausible store, and is capacity miss ~0 in steady?
    fits_38 = ws_gb <= 38
    out["fits_38GB_store"] = bool(fits_38)
    out["saturating"] = bool(late_rate < 2.0)
    Path("artifacts/h0a_longtrace.json").write_text(json.dumps(out, indent=1))
    print(f"\nWS {ws_gb:.1f}GB fits 38GB store: {fits_38}; "
          f"saturating: {late_rate < 2.0}")
    print("wrote artifacts/h0a_longtrace.json")


if __name__ == "__main__":
    main()
