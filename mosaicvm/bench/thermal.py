"""Thermal measurement contract utilities (AGENTS.md §5).

Every GPU benchmark must snapshot thermal state and label its results
COOL / CONTROLLED / STEADY / UNKNOWN. UNKNOWN cannot certify a gate.
"""
from __future__ import annotations

import time


def snapshot() -> dict:
    """NVML snapshot: temperature, clocks, power, limits. Falls back to
    {'nvml': 'unavailable'} — callers must then label UNKNOWN."""
    try:
        import pynvml
    except Exception:
        return {"nvml": "unavailable"}
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else str(name)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        try:
            clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        except Exception:
            clock = None
        try:
            mem_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
        except Exception:
            mem_clock = None
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
        except Exception:
            power_mw = None
        try:
            limit_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(h)
        except Exception:
            limit_mw = None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        except Exception:
            util = None
        return {
            "gpu": name,
            "temp_c": temp,
            "sm_clock_mhz": clock,
            "mem_clock_mhz": mem_clock,
            "power_w": None if power_mw is None else power_mw / 1000.0,
            "power_limit_w": None if limit_mw is None else limit_mw / 1000.0,
            "gpu_util_pct": util,
            "ts": time.time(),
        }
    except Exception as exc:
        return {"nvml": f"error:{type(exc).__name__}"}


def warmup_steady(target_seconds: float = 120.0, poll_s: float = 5.0,
                  stable_window: int = 3, temp_tolerance_c: float = 2.0,
                  max_seconds: float = 900.0) -> dict:
    """Burn GPU work until temperature stabilizes (STEADY pre-state).

    Returns {"steady": bool, "seconds": float, "start": snap, "end": snap}.
    A compute burn via small GEMMs; aborts at max_seconds regardless.
    """
    import torch

    start = snapshot()
    if "temp_c" not in start:
        return {"steady": False, "reason": "nvml_unavailable", "start": start, "end": start}
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    temps: list[int] = []
    t0 = time.time()
    steady = False
    while time.time() - t0 < max_seconds:
        t_end = time.time() + poll_s
        while time.time() < t_end:
            a @ b
        torch.cuda.synchronize()
        s = snapshot()
        temps.append(s.get("temp_c"))
        if len(temps) >= stable_window + 1:
            recent = temps[-(stable_window + 1):]
            if max(recent) - min(recent) <= temp_tolerance_c and time.time() - t0 >= target_seconds:
                steady = True
                break
    end = snapshot()
    del a, b
    torch.cuda.empty_cache()
    return {"steady": steady, "seconds": time.time() - t0,
            "temps_c": temps, "start": start, "end": end}


def classify(snaps: list[dict]) -> str:
    """COOL / CONTROLLED / STEADY / UNKNOWN from benchmark snapshots.

    COOL: first snapshot cold-ish (<= 45C) and total drift <= 5C.
    STEADY: drift <= 3C after a warm phase (start temp >= 55C).
    CONTROLLED: caller asserted locked clocks (not detectable here).
    Otherwise UNKNOWN.
    """
    temps = [s.get("temp_c") for s in snaps if isinstance(s, dict) and s.get("temp_c") is not None]
    if not temps or len(temps) < 2:
        return "UNKNOWN"
    drift = max(temps) - min(temps)
    if temps[0] <= 45 and drift <= 5:
        return "COOL"
    if temps[0] >= 55 and drift <= 3:
        return "STEADY"
    return "UNKNOWN"


def ab_interleave(fn_a, fn_b, rounds: int = 10, warmup: int = 1) -> dict:
    """Interleaved A/B (ABAB...) under the §5.2 contract.

    Returns medians and delta = (T_A - T_B)/T_A with thermal snapshots.
    """
    import statistics

    import torch

    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.cuda.synchronize()
    secs = {"A": [], "B": []}
    snaps = [snapshot()]
    for _ in range(rounds):
        for key, fn in (("A", fn_a), ("B", fn_b)):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            secs[key].append(s.elapsed_time(e) / 1e3)
        snaps.append(snapshot())
    ta = statistics.median(secs["A"])
    tb = statistics.median(secs["B"])
    return {
        "t_a_s": ta,
        "t_b_s": tb,
        "delta": (ta - tb) / ta if ta else None,
        "a_s": secs["A"],
        "b_s": secs["B"],
        "thermal": snaps,
        "thermal_label": classify(snaps),
    }
