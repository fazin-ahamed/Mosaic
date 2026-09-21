"""LINK0 — measured PCIe H2D bandwidth at relevant transfer sizes.

RTX 3060 is PCIe Gen3 x16 (measured via nvidia-smi). Benchmarks sustained
pinned H2D at: large buffer (asymptotic BW), current expert size (13.37 MB
packed FP4), compressed-expert sizes (2-8 MB). Uses cuda events, warmup, p50
over reps. No assumed 26.8 GB/s — all downstream math uses these numbers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

SIZES_MB = [256, 64, 13.37, 8, 4, 2.2, 1.0]   # large -> compressed-expert range
REPS = 30


def bench(nbytes):
    n = int(nbytes)
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(n, dtype=torch.uint8, device="cuda")
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s.record()
        dev.copy_(host, non_blocking=True)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))   # ms
    ts.sort()
    p50 = ts[len(ts) // 2]
    gbs = nbytes / (p50 / 1e3) / 1e9
    return round(p50, 4), round(gbs, 2)


def main():
    torch.cuda.init()
    out = {"classification": "LINK0_BANDWIDTH",
           "pcie": "Gen3 x16 (nvidia-smi)", "reps": REPS,
           "note": "pinned H2D copy_, cuda-event p50", "sizes": {}}
    print(f"{'MB':>8} {'p50_ms':>9} {'GB/s':>8}")
    for mb in SIZES_MB:
        nb = mb * 1e6
        p50, gbs = bench(nb)
        out["sizes"][str(mb)] = {"p50_ms": p50, "GB_s": gbs}
        print(f"{mb:>8} {p50:>9} {gbs:>8}")
    # asymptotic = large-buffer BW; expert-size = 13.37MB
    out["asymptotic_GB_s"] = out["sizes"]["256"]["GB_s"]
    out["expert_13MB_GB_s"] = out["sizes"]["13.37"]["GB_s"]
    Path("artifacts/link0.json").write_text(json.dumps(out, indent=1))
    print(f"\nasymptotic {out['asymptotic_GB_s']} GB/s, "
          f"expert-size {out['expert_13MB_GB_s']} GB/s")
    print("wrote artifacts/link0.json")


if __name__ == "__main__":
    main()
