"""Phase 2.12W-A: real transport measurement on the target machine.

Measures pinned/pageable H2D, copy+compute overlap, and NVMe read throughput
at expert-sized granularities. Replaces the assumed 12 GB/s pipe.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch

from mosaicvm.bench.thermal import snapshot

ARTIFACT = Path("artifacts/transport.json")
EXPERT_PACKED4_BYTES = (2 * 2048 * 4096 + 4096 * 2048) // 2 + \
    (2 * 2048 * 4096 + 4096 * 2048) // 32  # ≈ 12.98 MB
EXPERT_HOT8_BYTES = (2 * 2048 * 4096 + 4096 * 2048) + \
    (2 * 2048 * 4096 + 4096 * 2048) // 32  # ≈ 25.56 MB


def _bench_h2d(src, dst, n_iters=20, warmup=5):
    for _ in range(warmup):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return statistics.median(times)


def _bench_nvme(path, size_mb, n_iters=5):
    """Sequential read benchmark."""
    import subprocess
    cmd = f"dd if={path} of=/dev/null bs=1M count={size_mb} 2>&1 | tail -1"
    times = []
    for _ in range(n_iters):
        t0 = time.time()
        subprocess.run(cmd, shell=True, capture_output=True)
        times.append(time.time() - t0)
    return statistics.median(times)


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    thermal_start = snapshot()
    results = {"gpu": gpu, "thermal_start": thermal_start}

    # --- H2D bandwidth ---
    print("=== H2D bandwidth ===")
    for size_mb, label in [(1, "1MB"), (4, "4MB"), (13, "expert-packed4"),
                            (26, "expert-hot8"), (100, "100MB"), (500, "500MB")]:
        n = size_mb * 1024 * 1024 // 2  # bf16 elements
        # pinned
        src_pinned = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
        src_pinned.normal_()
        dst = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        sec = _bench_h2d(src_pinned, dst)
        gbs = n * 2 / sec / 1e9
        results[f"h2d_pinned_{label}"] = {"sec": sec, "gbs": gbs}
        print(f"  pinned {label:>15}: {gbs:6.1f} GB/s ({sec*1e3:.1f} ms)")
        del src_pinned, dst

        # pageable
        src_page = torch.empty(n, dtype=torch.bfloat16)
        src_page.normal_()
        dst = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        sec = _bench_h2d(src_page, dst)
        gbs = n * 2 / sec / 1e9
        results[f"h2d_pageable_{label}"] = {"sec": sec, "gbs": gbs}
        print(f"  pageable {label:>13}: {gbs:6.1f} GB/s ({sec*1e3:.1f} ms)")
        del src_page, dst

    torch.cuda.empty_cache()

    # --- Copy + compute overlap ---
    print("\n=== Copy + compute overlap ===")
    # simulate: transfer an expert (13MB packed4) while computing a GEMM
    expert_size = EXPERT_PACKED4_BYTES
    n_expert = expert_size // 2
    src = torch.empty(n_expert, dtype=torch.bfloat16, pin_memory=True)
    src.normal_()
    dst_buf = torch.empty(n_expert, dtype=torch.bfloat16, device="cuda")

    # compute: a GEMM that takes about as long as the transfer
    compute_a = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
    compute_b = torch.randn(4096, 2048, dtype=torch.bfloat16, device="cuda")

    # measure transfer alone
    t_xfer = _bench_h2d(src, dst_buf)
    # measure compute alone
    t_compute = _bench_component(lambda: compute_a @ compute_b)
    # measure overlapped (on separate streams)
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    def overlapped():
        with torch.cuda.stream(copy_stream):
            dst_buf.copy_(src, non_blocking=True)
        with torch.cuda.stream(compute_stream):
            compute_a @ compute_b
        torch.cuda.synchronize()

    for _ in range(5):
        overlapped()
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        overlapped()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    t_overlap = statistics.median(times)

    overlap_eff = (t_xfer + t_compute) / t_overlap if t_overlap > 0 else 0
    results["overlap"] = {
        "transfer_sec": t_xfer,
        "compute_sec": t_compute,
        "overlapped_sec": t_overlap,
        "overlap_efficiency": overlap_eff,
        "transfer_gb": expert_size / 1e9,
    }
    print(f"  transfer alone: {t_xfer*1e3:.1f} ms")
    print(f"  compute alone:  {t_compute*1e3:.1f} ms")
    print(f"  overlapped:     {t_overlap*1e3:.1f} ms (eff {overlap_eff:.1%})")
    del src, dst_buf, compute_a, compute_b
    torch.cuda.empty_cache()

    # --- Multi-stream pipelining ---
    print("\n=== Multi-stream expert transfer pipeline ===")
    # simulate streaming 10 experts while computing
    n_experts = 10
    srcs = [torch.empty(n_expert, dtype=torch.bfloat16, pin_memory=True) for _ in range(n_experts)]
    dsts = [torch.empty(n_expert, dtype=torch.bfloat16, device="cuda") for _ in range(n_experts)]
    compute_a = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
    compute_b = torch.randn(4096, 2048, dtype=torch.bfloat16, device="cuda")

    def pipeline():
        # start all copies on copy stream
        with torch.cuda.stream(copy_stream):
            for s, d in zip(srcs, dsts):
                d.copy_(s, non_blocking=True)
        # compute on main stream (overlaps with copies)
        for _ in range(n_experts):
            compute_a @ compute_b
        torch.cuda.synchronize()

    for _ in range(3):
        pipeline()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        pipeline()
    torch.cuda.synchronize()
    t_pipeline = (time.time() - t0) / 10
    pipeline_gbs = n_experts * expert_size / t_pipeline / 1e9
    results["pipeline_10_experts"] = {
        "sec": t_pipeline,
        "total_gb": n_experts * expert_size / 1e9,
        "effective_gbs": pipeline_gbs,
    }
    print(f"  10 experts ({n_experts * expert_size / 1e9:.2f} GB) in {t_pipeline*1e3:.0f} ms")
    print(f"  effective throughput: {pipeline_gbs:.1f} GB/s")
    del srcs, dsts, compute_a, compute_b
    torch.cuda.empty_cache()

    # --- NVMe throughput ---
    print("\n=== NVMe throughput ===")
    # find the project disk
    nvme_path = "/home/fazinahamed/Documents/MosaicVM/.mosaicvm/core/core.safetensors"
    if not os.path.exists(nvme_path):
        nvme_path = "/tmp/test_nvme"
        subprocess.run(f"dd if=/dev/urandom of={nvme_path} bs=1M count=500", shell=True)
    for size_mb in (100, 500):
        sec = _bench_nvme(nvme_path, size_mb)
        gbs = size_mb / 1024 / sec
        results[f"nvme_seq_{size_mb}mb"] = {"sec": sec, "gbs": gbs}
        print(f"  sequential {size_mb} MB: {gbs:.2f} GB/s")

    thermal_end = snapshot()
    results["thermal_end"] = thermal_end

    # Summary
    pinned_avg = statistics.mean([
        results.get(f"h2d_pinned_{l}", {}).get("gbs", 0)
        for l in ("100MB", "500MB")
    ])
    results["summary"] = {
        "pinned_h2d_gbs": pinned_avg,
        "assumed_h2d_gbs": 12.0,
        "actual_vs_assumed": pinned_avg / 12.0,
        "expert_transfer_ms": EXPERT_PACKED4_BYTES / (pinned_avg * 1e9) * 1e3,
        "k56_transfer_budget_gb": pinned_avg * (56 / 30),  # GB during 56/30 s window
        "k56_actual_traffic_gb": 49.7,
        "reduction_needed": 49.7 / (pinned_avg * (56 / 30)),
    }
    print(f"\n=== summary ===")
    print(f"  measured pinned H2D: {pinned_avg:.1f} GB/s (assumed 12.0)")
    print(f"  expert packed4 transfer: {results['summary']['expert_transfer_ms']:.1f} ms")
    print(f"  K56 reduction needed for raw 30: {results['summary']['reduction_needed']:.1f}×")

    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {ARTIFACT}")
    return results


def _bench_component(fn, n_iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return statistics.median(times)


if __name__ == "__main__":
    run()
