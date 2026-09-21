"""Hardware profiler: emits one JSON profile for the current GPU + host.

Measures VRAM bandwidth (DtoD memcpy + SAXPY triad), H2D/D2H throughput
(pageable vs pinned), host RAM bandwidth/latency, CPU GEMM throughput,
volume read throughput, and GPU clock/power under sustained GEMM load.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch


def _time_gpu(fn, iters=10, warmup=3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3  # seconds per iter


def vram_bandwidth() -> dict:
    dev = torch.device("cuda")
    out = {"memcpy_d2d": [], "triad": []}
    free, total = torch.cuda.mem_get_info()
    for mb in [64, 256, 1024, 4096, 8192]:
        nbytes = mb * 1024 * 1024
        if 3 * nbytes + 256 * 1024 * 1024 > free:
            out.setdefault("skipped_mb", []).append(mb)
            continue
        n = nbytes // 2  # bf16 elements
        a = torch.empty(n, dtype=torch.bfloat16, device=dev)
        b = torch.empty_like(a)
        c = torch.empty_like(a)
        a.normal_()
        sec = _time_gpu(lambda: b.copy_(a))
        out["memcpy_d2d"].append({"size_mb": mb, "bw_gbs": (2 * n * 2) / sec / 1e9})
        alpha = 1.0
        sec = _time_gpu(lambda: torch.add(c, a, alpha=alpha, out=c))
        out["triad"].append({"size_mb": mb, "bw_gbs": (3 * n * 2) / sec / 1e9})
        del a, b, c
        torch.cuda.empty_cache()
    return out


def host_device_bandwidth() -> dict:
    dev = torch.device("cuda")
    out = {}
    for kind in ["pageable", "pinned"]:
        h2d, d2h = [], []
        for mb in [1, 16, 64, 256, 1024]:
            n = mb * 1024 * 1024 // 2
            if kind == "pinned":
                host = torch.empty(n, dtype=torch.bfloat16).pin_memory()
            else:
                host = torch.randn(n, dtype=torch.float32).to(torch.bfloat16)
            gpu = torch.empty(n, dtype=torch.bfloat16, device=dev)
            t0 = time.perf_counter()
            for _ in range(5):
                gpu.copy_(host, non_blocking=True)
            torch.cuda.synchronize()
            h2d.append({"size_mb": mb, "gbs": 5 * host.numel() * 2 / (time.perf_counter() - t0) / 1e9})
            t0 = time.perf_counter()
            for _ in range(5):
                host.copy_(gpu, non_blocking=True)
            torch.cuda.synchronize()
            d2h.append({"size_mb": mb, "gbs": 5 * host.numel() * 2 / (time.perf_counter() - t0) / 1e9})
            del host, gpu
        out[kind] = {"h2d": h2d, "d2h": d2h}
    return out


def ram_profile() -> dict:
    out = {}
    n = 512 * 1024 * 1024  # float32 => 2 GB
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    c = np.empty(n, dtype=np.float32)
    t0 = time.perf_counter()
    np.add(a, b, out=c)
    dt = time.perf_counter() - t0
    out["seq_bw_gbs"] = 3 * n * 4 / dt / 1e9
    del a, b, c
    # pointer-chase random latency over 512 MB buffer of uint64 indices
    size = 64 * 1024 * 1024  # entries
    buf = np.arange(size, dtype=np.uint64)
    rng = np.random.default_rng(0)
    perm = rng.permutation(size)
    buf[:] = perm
    idx = np.uint64(0)
    steps = 1_000_000
    t0 = time.perf_counter()
    for _ in range(steps):
        idx = buf[idx]
    out["random_latency_ns"] = (time.perf_counter() - t0) / steps * 1e9
    return out


def cpu_gemm() -> dict:
    out = []
    for m, k, n, dt in [(4096, 2048, 1, "bf16"), (4096, 2048, 1, "int8"), (4096, 2048, 4096, "bf16")]:
        if dt == "bf16":
            a = torch.randn(m, k, dtype=torch.bfloat16)
            b = torch.randn(k, n, dtype=torch.bfloat16)
        else:
            a = torch.randint(-127, 127, (m, k), dtype=torch.int8)
            b = torch.randint(-127, 127, (k, n), dtype=torch.int8)
        t0 = time.perf_counter()
        _ = a @ b
        dt_sec = time.perf_counter() - t0
        flops = 2 * m * k * n
        out.append({"shape": f"{m}x{k}x{n}", "dtype": dt, "tflops": flops / dt_sec / 1e12, "sec": dt_sec})
    return out


def volume_read(path: str) -> dict:
    """Sequential + 4KB random reads from a file on the mounted volume."""
    import os

    out = {"path": path}
    files = sorted(
        os.path.join(root, f)
        for root, _, fs in os.walk(path)
        for f in fs
        if f.endswith(".safetensors")
    )
    if not files:
        out["error"] = "no safetensors found on volume"
        return out
    target = files[0]
    sz = os.path.getsize(target)
    chunk = 8 * 1024 * 1024
    total = 0
    t0 = time.perf_counter()
    with open(target, "rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            total += len(data)
    out["seq_read_gbs"] = total / (time.perf_counter() - t0) / 1e9
    # 4 KB random reads
    rng = np.random.default_rng(1)
    with open(target, "rb") as fh:
        offsets = rng.integers(0, sz - 4096, size=2000)
        t0 = time.perf_counter()
        for off in offsets[:100]:
            fh.seek(int(off))
            fh.read(4096)
    out["rand4k_iops"] = 100 / (time.perf_counter() - t0)
    return out


def power_clocks(duration_s: int = 20) -> dict:
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()
    lim = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
    samples = []
    stop = time.time() + duration_s
    x = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")

    def gemm_loop():
        while time.time() < stop:
            x @ x
        torch.cuda.synchronize()

    import threading

    th = threading.Thread(target=gemm_loop)
    th.start()
    while time.time() < stop:
        w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        mem = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        samples.append({"watts": w, "sm_mhz": sm, "mem_mhz": mem, "temp_c": temp, "util_pct": util})
        time.sleep(0.5)
    th.join()
    return {
        "gpu_name": name,
        "power_limit_watts_note": "Modal does not permit changing power limits; default only recorded",
        "default_power_limit_w": lim,
        "samples": samples,
        "avg_power_w": float(np.mean([s["watts"] for s in samples])) if samples else None,
    }


def run(volume_path: str | None = None) -> dict:
    profile = {"torch": torch.__version__, "device_count": torch.cuda.device_count()}
    props = torch.cuda.get_device_properties(0)
    profile["gpu"] = {
        "name": props.name,
        "cc": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / 1e9, 1),
        "sms": props.multi_processor_count,
    }
    profile["vram_bandwidth"] = vram_bandwidth()
    profile["host_device"] = host_device_bandwidth()
    profile["ram"] = ram_profile()
    profile["cpu_gemm"] = cpu_gemm()
    if volume_path:
        profile["volume"] = volume_read(volume_path)
    profile["power_clocks"] = power_clocks()
    return profile
