"""H0-A — optimistic combined-memory capacity bound for CPU-hybrid Exact-30.

Grants the hybrid system MORE than any real implementation:
  - usable VRAM + usable host RAM treated as ONE ideal expert store
  - zero CPU/GPU expert compute cost
  - free movement WITHIN that combined store (no PCIe, no copy)
  - perfect future knowledge (Belady) for what lives in-store
Only UNAVOIDABLE NVMe reads (expert not in the combined store) cost time,
at MEASURED cold NVMe bandwidth.

If even this relaxation cannot hit 30 tok/s, hybrid perf engineering stops for
this workload/config. Separates compulsory (cold-start, amortized) from
capacity (recurring) misses so short-trace first-touches don't inflate the
steady-state estimate.

Trace: persisted real routes (short multi-domain corpus, reproduced;
EK0c long-stream agreed within 3%). Raw packed storage first; +1.091x
sensitivity (LC0 measured, zero decompress cost) as a labeled case.
Token accounting: sustained = total accepted tokens / total time, E[A]=3.757.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.ek0b_residency import (
    load_cycles, cycles_to_stream, belady_stream, per_cycle_miss)

BYTES = 13369344
NVME_GBs = 0.55            # MEASURED cold random expert-size aggregate (fadvise DONTNEED)
NVME_GBs_p50 = 1.36        # measured p50 (optimistic)
CORE_MS = 17.5
EA = 3.757
BUDGET_MS = EA / 30 * 1e3  # 125.2
# usable combined store: VRAM(12) - reserve, + RAM(31) - reserve
VRAM_RESERVE_GB = 4.0      # core+KV+shared+activations+staging+cudagraphs
RAM_RESERVE_GB = 9.0       # OS + runtime + proposer + page cache headroom
USABLE_VRAM = 12.0 - VRAM_RESERVE_GB     # 8
USABLE_RAM = 31.0 - RAM_RESERVE_GB       # 22
COMBINED_GB = USABLE_VRAM + USABLE_RAM   # 30 GB ideal store
COMP = 1.091               # LC0 measured lossless ratio (sensitivity case)


def replay(cap_slots):
    """Return (mean_total_miss, mean_compulsory, mean_capacity) per cycle,
    aggregated over domains with per-domain cache reset."""
    cyc = load_cycles()
    tot_miss = comp_miss = cap_miss = 0; ncyc = 0
    for cat, cy in cyc.items():
        stream, spans = cycles_to_stream(cy)
        miss = per_cycle_miss(belady_stream(stream, cap_slots), spans)
        inf = per_cycle_miss(belady_stream(stream, 10**9), spans)  # compulsory
        for i in range(len(miss)):
            tot_miss += miss[i]; comp_miss += inf[i]
            cap_miss += (miss[i] - inf[i])
            ncyc += 1
    return tot_miss / ncyc, comp_miss / ncyc, cap_miss / ncyc


def main():
    print(f"combined ideal store: VRAM {USABLE_VRAM} + RAM {USABLE_RAM} "
          f"= {COMBINED_GB} GB (reserves VRAM {VRAM_RESERVE_GB}, RAM {RAM_RESERVE_GB})")
    out = {"classification": "H0A_CAPACITY",
           "nvme_GBs_cold": NVME_GBs, "nvme_GBs_p50": NVME_GBs_p50,
           "combined_store_GB": COMBINED_GB, "budget_ms": BUDGET_MS,
           "core_ms": CORE_MS, "EA": EA, "cases": {}}

    for label, gb, comp in (("raw_packed", COMBINED_GB, 1.0),
                            ("raw_packed_compressed_1.091", COMBINED_GB, COMP)):
        eff_gb = gb * comp
        cap_slots = int(eff_gb * 1e9 / BYTES)
        tot, compu, capm = replay(cap_slots)
        # only cold (NVMe) misses cost time in the optimistic oracle.
        # steady-state: exclude compulsory first-touch (amortized over long gen)
        # -> capacity misses are the recurring NVMe traffic.
        for miss_kind, miss_val in (("total_incl_compulsory", tot),
                                    ("capacity_only_steadystate", capm)):
            cold_bytes = miss_val * BYTES / comp     # bytes actually read from NVMe
            for bw_label, bw in (("cold0.55", NVME_GBs), ("p50_1.36", NVME_GBs_p50)):
                nvme_ms = cold_bytes / (bw * 1e9) * 1e3
                cyc_ms = CORE_MS + nvme_ms
                tput = EA / (cyc_ms / 1e3)
                key = f"{label}|{miss_kind}|{bw_label}"
                out["cases"][key] = {
                    "eff_store_GB": round(eff_gb, 1), "cap_slots": cap_slots,
                    "miss_per_cycle": round(miss_val, 1),
                    "nvme_GB_per_cycle": round(cold_bytes / 1e9, 3),
                    "nvme_ms": round(nvme_ms, 1), "cycle_ms": round(cyc_ms, 1),
                    "tput_UB": round(tput, 2)}

    print(f"\n{'case':>58} {'miss':>7} {'nvmeGB':>7} {'nvmeMs':>8} {'tputUB':>7}")
    for k, v in out["cases"].items():
        print(f"{k:>58} {v['miss_per_cycle']:>7} {v['nvme_GB_per_cycle']:>7} "
              f"{v['nvme_ms']:>8} {v['tput_UB']:>7}")

    # verdict: best optimistic case (compressed, steady-state, p50 NVMe)
    best = max(v["tput_UB"] for v in out["cases"].values())
    out["best_optimistic_tput_UB"] = best
    out["verdict"] = ("HYBRID_ALIVE" if best >= 30 else
                      "HYBRID_INSUFFICIENT" if best < 30 else "INCONCLUSIVE")
    print(f"\nbest optimistic tput UB: {best} tok/s -> {out['verdict']}")
    print(f"(30 tok/s needs cycle <= {BUDGET_MS:.1f}ms)")
    Path("artifacts/h0_capacity.json").write_text(json.dumps(out, indent=1))
    print("wrote artifacts/h0_capacity.json")


if __name__ == "__main__":
    main()
