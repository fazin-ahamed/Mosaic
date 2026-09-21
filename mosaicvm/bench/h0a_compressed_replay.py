"""H0-A compressed-size sensitivity on the persisted long single-session trace.

Replays artifacts/ek0c_routes.pt under raw (1x) and measured LC0 (1.091x)
compression. Compression does two things: larger effective cache (store*ratio)
AND smaller transfer per miss (bytes/ratio). Reports Belady misses and cold
GB/cycle for both, plus TIME-WINDOWED misses (first vs second half) to show
whether capacity misses are rising/falling over the finite trace rather than
inferring steady state from working-set size.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.ek0b_residency import (
    cycles_to_stream, belady_stream, per_cycle_miss)

BYTES = 13369344
N_LAYERS = 43
W = 6
STRIDE = 4
STORE_GB = [30, 35, 38]
RATIOS = [1.0, 1.091]
NVME = [0.55, 1.36]
CORE_MS = 17.5


def main():
    d = torch.load("artifacts/ek0c_routes.pt", weights_only=False)
    routes = d["routes"]
    cycles = []
    for start in range(0, len(routes) - W + 1, STRIDE):
        U = set()
        for pm in routes[start:start + W]:
            for L in range(N_LAYERS):
                for e in pm.get(L, []):
                    U.add((L, e))
        cycles.append(sorted(U))
    stream, spans = cycles_to_stream(cycles)
    ncyc = len(cycles)
    half = ncyc // 2

    out = {"classification": "H0A_COMPRESSED_REPLAY", "n_cycles": ncyc,
           "note": "single-session long trace; time-windowed", "cases": {}}
    print(f"{'store':>5} {'ratio':>6} {'effGB':>6} {'miss_all':>9} "
          f"{'miss_1st':>9} {'miss_2nd':>9} {'coldGB':>7}")
    for gb in STORE_GB:
        for r in RATIOS:
            eff = gb * r
            slots = int(eff * 1e9 / BYTES)
            miss = per_cycle_miss(belady_stream(stream, slots), spans)
            m_all = sum(miss) / ncyc
            m1 = sum(miss[:half]) / max(half, 1)
            m2 = sum(miss[half:]) / max(ncyc - half, 1)
            cold_gb = m_all * BYTES / r / 1e9    # compressed transfer
            key = f"{gb}GB_x{r}"
            rec = {"eff_GB": round(eff, 1), "slots": slots,
                   "miss_all": round(m_all, 1), "miss_first_half": round(m1, 1),
                   "miss_second_half": round(m2, 1),
                   "cold_GB_per_cycle": round(cold_gb, 3)}
            for bw in NVME:
                cyc = CORE_MS + cold_gb / bw * 1e3
                rec[f"tputUB_nvme{bw}"] = round(3.757 / (cyc / 1e3), 2)
            out["cases"][key] = rec
            print(f"{gb:>5} {r:>6} {eff:>6.1f} {m_all:>9.1f} {m1:>9.1f} "
                  f"{m2:>9.1f} {cold_gb:>7.3f}")
    Path("artifacts/h0a_compressed_replay.json").write_text(json.dumps(out, indent=1))
    print("\ntime-window note: miss_2nd vs miss_1st shows trace-direction, not")
    print("proven steady state. wrote artifacts/h0a_compressed_replay.json")


if __name__ == "__main__":
    main()
