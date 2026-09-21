"""Build-only memory soak for the FP32 mmap trainer.

Runs the full 1001-prompt union build + FP32 normalization + random batch
touches WITHOUT training. Samples anonymous RSS (VmRSS from /proc/self/status
minus file-backed via RssFile) and MemAvailable. Gate: peak anon RSS < 12 GB.
"""
from __future__ import annotations

import threading
import time

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_scale1000_train import (
    build_union_mmap, normalize_mmap, N_ROWS)

GATE_GB = 12.0


def proc_mem():
    """Return (anon_rss_gb, rss_gb). Anon = RssAnon from /proc/self/status."""
    anon = rss = 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                anon = int(line.split()[1]) / (1024 * 1024)
            elif line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / (1024 * 1024)
    return anon, rss


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return float("inf")


peak = {"anon": 0.0, "rss": 0.0, "avail_min": float("inf")}
stop = threading.Event()


def sampler():
    while not stop.is_set():
        a, r = proc_mem()
        peak["anon"] = max(peak["anon"], a)
        peak["rss"] = max(peak["rss"], r)
        peak["avail_min"] = min(peak["avail_min"], mem_available_gb())
        time.sleep(0.5)


def main():
    torch.set_num_threads(8)
    t0 = time.time()
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    xm, xa, labels = build_union_mmap()
    assert xm.shape[0] == N_ROWS
    print(f"build done ({time.time()-t0:.0f}s) "
          f"anon={peak['anon']:.2f}GB rss={peak['rss']:.2f}GB", flush=True)

    xmn, xan, st = normalize_mmap(xm, xa)
    print(f"normalize done ({time.time()-t0:.0f}s) "
          f"anon={peak['anon']:.2f}GB rss={peak['rss']:.2f}GB", flush=True)

    # touch random batches (simulate training reads)
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        i = int(torch.randint(0, N_ROWS - 512, (1,), generator=g))
        _ = xmn[i:i + 512].float().sum() + xan[i:i + 512].float().sum()
    print(f"batch touch done ({time.time()-t0:.0f}s) "
          f"anon={peak['anon']:.2f}GB rss={peak['rss']:.2f}GB", flush=True)

    stop.set(); th.join(timeout=2)
    print(f"PEAK anon={peak['anon']:.2f}GB rss={peak['rss']:.2f}GB "
          f"min_avail={peak['avail_min']:.2f}GB")
    ok = peak["anon"] < GATE_GB
    print("SOAK:", "PASS" if ok else "FAIL",
          f"(gate anon < {GATE_GB}GB)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
