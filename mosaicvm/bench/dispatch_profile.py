"""Overhead decomposition of the current K=8 sequential MoE path."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from mosaicvm.bench.compute_floor import (
    measure_dense_attention,
    measure_hyper_connection,
    measure_router,
    measure_shared_expert,
)
from mosaicvm.bench.floors import HIDDEN, MOE_INTER, N_LAYERS, TOP_K
from mosaicvm.kernels.w4a16_gemm import pack_w4, w4a16_gemm, w4a16_grouped_moe

K = 8
N_UNIQUE = 32
ASSIGNMENTS = K * TOP_K  # 48
ITERS = 20
WARMUP = 5


def amdahl(components_ms: dict, speedup_moe: float) -> dict:
    """tok/s at K=8 if moe is sped up by speedup_moe, and if moe time -> 0."""
    moe = float(components_ms.get("moe", 0.0))
    rest = sum(
        float(components_ms.get(k, 0.0))
        for k in ("attn", "shared", "router", "hc", "other")
    )
    sped = (moe / speedup_moe) + rest
    zero = rest
    # components are per-layer milliseconds
    def toks(per_layer_ms: float) -> float:
        t_block = (per_layer_ms / 1e3) * N_LAYERS
        return 0.0 if t_block <= 0 else K / t_block

    return {
        "speedup_moe": speedup_moe,
        "toks_at_speedup": toks(sped),
        "toks_if_moe_zero": toks(zero),
        "per_layer_ms_at_speedup": sped,
        "per_layer_ms_if_moe_zero": zero,
    }


def _event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return start.elapsed_time(end)


def _pack_pool(n_e: int, k_in: int, n_out: int):
    packed, scales = [], []
    for _ in range(n_e):
        w = torch.randn(k_in, n_out, dtype=torch.bfloat16, device="cuda")
        p, s = pack_w4(w)
        packed.append(p)
        scales.append(s)
    return packed, scales


def _make_ids() -> torch.Tensor:
    # 32 unique experts, 48 assignments (first 16 experts get 2 tokens).
    ids = torch.cat(
        [
            torch.arange(N_UNIQUE, device="cuda", dtype=torch.int64),
            torch.arange(ASSIGNMENTS - N_UNIQUE, device="cuda", dtype=torch.int64),
        ]
    )
    return ids


class _LaunchCounter:
    def __init__(self):
        self.n = 0
        self._orig = w4a16_gemm

    def __call__(self, *args, **kwargs):
        self.n += 1
        return self._orig(*args, **kwargs)


def _profiled_sequential_moe(x, pg, pu, pd, sg, su, sd, expert_ids, gemm):
    """Sequential unique-expert loop with CUDA events around each tag."""
    m, h = x.shape
    y = torch.zeros(m, h, dtype=x.dtype, device=x.device)

    e_host_s = torch.cuda.Event(enable_timing=True)
    e_host_e = torch.cuda.Event(enable_timing=True)
    e_gate_s = torch.cuda.Event(enable_timing=True)
    e_gate_e = torch.cuda.Event(enable_timing=True)
    e_up_s = torch.cuda.Event(enable_timing=True)
    e_up_e = torch.cuda.Event(enable_timing=True)
    e_silu_s = torch.cuda.Event(enable_timing=True)
    e_silu_e = torch.cuda.Event(enable_timing=True)
    e_down_s = torch.cuda.Event(enable_timing=True)
    e_down_e = torch.cuda.Event(enable_timing=True)
    e_scat_s = torch.cuda.Event(enable_timing=True)
    e_scat_e = torch.cuda.Event(enable_timing=True)

    host_ms = gate_ms = up_ms = silu_ms = down_ms = scat_ms = 0.0

    e_host_s.record()
    unique = expert_ids.unique()
    unique_list = unique.tolist()
    e_host_e.record()
    torch.cuda.synchronize()
    host_ms += _event_ms(e_host_s, e_host_e)

    for e in unique_list:
        e_host_s.record()
        mask = expert_ids == e
        xe = x[mask]
        e_host_e.record()

        e_gate_s.record()
        g_lin = gemm(xe, pg[e], sg[e])
        e_gate_e.record()

        e_silu_s.record()
        g = torch.nn.functional.silu(g_lin)
        e_silu_e.record()

        e_up_s.record()
        u = gemm(xe, pu[e], su[e])
        e_up_e.record()

        e_silu_s.record()
        z = g * u
        e_silu_e.record()

        e_down_s.record()
        ye = gemm(z, pd[e], sd[e])
        e_down_e.record()

        e_scat_s.record()
        y[mask] = ye
        e_scat_e.record()

        torch.cuda.synchronize()
        host_ms += _event_ms(e_host_s, e_host_e)
        gate_ms += _event_ms(e_gate_s, e_gate_e)
        up_ms += _event_ms(e_up_s, e_up_e)
        silu_ms += _event_ms(e_silu_s, e_silu_e)
        down_ms += _event_ms(e_down_s, e_down_e)
        scat_ms += _event_ms(e_scat_s, e_scat_e)

    return y, {
        "host_loop_ms": host_ms,
        "w4_gate_ms": gate_ms,
        "w4_up_ms": up_ms,
        "silu_mul_ms": silu_ms,
        "w4_down_ms": down_ms,
        "scatter_reduce_ms": scat_ms,
        "w4_compute_ms": gate_ms + up_ms + silu_ms + down_ms,
    }


def _profiler_stats(fn) -> dict:
    activities = [torch.profiler.ProfilerActivity.CUDA]
    wall0 = time.perf_counter()
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        fn()
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall0) * 1e3
    events = [e for e in prof.key_averages() if e.device_time_total > 0]
    kernel_ms = sum(e.device_time_total for e in events) / 1e3  # us -> ms in some builds
    # torch profiler device_time_total is microseconds
    kernel_us = sum(e.device_time_total for e in events)
    n_kernels = sum(max(int(e.count), 1) for e in events)
    mean_us = (kernel_us / n_kernels) if n_kernels else 0.0
    gpu_idle_frac = 1.0 - min(1.0, (kernel_us / 1e3) / wall_ms) if wall_ms > 0 else 1.0
    return {
        "profiler_kernel_count": n_kernels,
        "mean_kernel_duration_us": mean_us,
        "sum_kernel_ms": kernel_us / 1e3,
        "profiler_wall_ms": wall_ms,
        "gpu_idle_frac": gpu_idle_frac,
        "top_kernels": [
            {"name": e.key, "count": int(e.count), "device_us": float(e.device_time_total)}
            for e in sorted(events, key=lambda x: -x.device_time_total)[:12]
        ],
    }


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    if "3060" not in gpu.lower():
        print(gpu)
        raise SystemExit(f"unexpected GPU: {gpu}")

    torch.manual_seed(0)
    x = torch.randn(ASSIGNMENTS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    expert_ids = _make_ids()
    unique = int(expert_ids.unique().numel())
    counts = torch.bincount(expert_ids, minlength=N_UNIQUE).float()
    mean_m = float(counts[counts > 0].mean().item())

    pg, sg = _pack_pool(N_UNIQUE, HIDDEN, MOE_INTER)
    pu, su = _pack_pool(N_UNIQUE, HIDDEN, MOE_INTER)
    pd, sd = _pack_pool(N_UNIQUE, MOE_INTER, HIDDEN)

    counter = _LaunchCounter()
    import mosaicvm.kernels.w4a16_gemm as gemm_mod

    gemm_mod.w4a16_gemm = counter  # type: ignore[assignment]
    try:
        for _ in range(WARMUP):
            _profiled_sequential_moe(x, pg, pu, pd, sg, su, sd, expert_ids, counter)
        torch.cuda.synchronize()
        counter.n = 0
        y, tags = _profiled_sequential_moe(x, pg, pu, pd, sg, su, sd, expert_ids, counter)
        launches = counter.n
        # restore and time the production sequential path for wall
        gemm_mod.w4a16_gemm = counter._orig  # type: ignore[assignment]
        for _ in range(WARMUP):
            w4a16_grouped_moe(x, pg, pu, pd, (sg, su, sd), expert_ids)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(ITERS):
            w4a16_grouped_moe(x, pg, pu, pd, (sg, su, sd), expert_ids)
        t1.record()
        torch.cuda.synchronize()
        moe_ms = t0.elapsed_time(t1) / ITERS

        def one_moe():
            w4a16_grouped_moe(x, pg, pu, pd, (sg, su, sd), expert_ids)

        prof = _profiler_stats(one_moe)
    finally:
        gemm_mod.w4a16_gemm = counter._orig  # type: ignore[assignment]

    # extra synchronize around the whole step
    e_sync_s = torch.cuda.Event(enable_timing=True)
    e_sync_e = torch.cuda.Event(enable_timing=True)
    e_sync_s.record()
    w4a16_grouped_moe(x, pg, pu, pd, (sg, su, sd), expert_ids)
    torch.cuda.synchronize()
    e_sync_e.record()
    torch.cuda.synchronize()
    sync_ms = e_sync_s.elapsed_time(e_sync_e)

    router_ms = measure_router()["sec_per_step"] * 1e3
    attn_ms = measure_dense_attention(k=K, ctx=4096)["sec_per_step"] * 1e3
    shared_ms = measure_shared_expert(K)["sec_per_step"] * 1e3
    hc_ms = measure_hyper_connection(K)["sec_per_step"] * 1e3

    components = {
        "moe": moe_ms,
        "attn": attn_ms,
        "shared": shared_ms,
        "router": router_ms,
        "hc": hc_ms,
        "other": 0.0,
    }
    amdahl_2x = amdahl(components, 2.0)
    amdahl_inf = amdahl(components, 1e9)

    launches_per_moe = launches
    if launches_per_moe < 3 * unique:
        print("DIAGNOSIS WRONG: launches_per_moe_layer is", launches_per_moe, "unique=", unique)
        print("tags:", tags)
        print("profiler:", json.dumps(prof, indent=2, default=str))
        raise SystemExit("cuda_kernel_launches_per_moe_layer < 3 * unique_experts")

    out = {
        "floor": "dispatch_profile_k8",
        "gpu": gpu,
        "K": K,
        "top_k": TOP_K,
        "unique_experts": unique,
        "assignments": ASSIGNMENTS,
        "mean_m_per_expert": mean_m,
        "cuda_kernel_launches_per_moe_layer": launches_per_moe,
        "cuda_kernel_launches_per_block": launches_per_moe * N_LAYERS,
        "mean_kernel_duration_us": prof["mean_kernel_duration_us"],
        "host_loop_ms": tags["host_loop_ms"],
        "w4_compute_ms": tags["w4_compute_ms"],
        "w4_gate_ms": tags["w4_gate_ms"],
        "w4_up_ms": tags["w4_up_ms"],
        "silu_mul_ms": tags["silu_mul_ms"],
        "w4_down_ms": tags["w4_down_ms"],
        "scatter_reduce_ms": tags["scatter_reduce_ms"],
        "attn_ms": attn_ms,
        "shared_ms": shared_ms,
        "hc_ms": hc_ms,
        "router_ms": router_ms,
        "sync_ms": sync_ms,
        "moe_wall_ms": moe_ms,
        "gpu_idle_frac": prof["gpu_idle_frac"],
        "profiler": prof,
        "amdahl_moe_2x": amdahl_2x,
        "amdahl_moe_inf": amdahl_inf,
        "y_norm": float(y.float().norm().item()),
        "nsys": "skipped_not_on_path",
    }
    Path("artifacts").mkdir(exist_ok=True)
    path = Path("artifacts/dispatch_profile_k8.json")
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in (
        "cuda_kernel_launches_per_moe_layer",
        "unique_experts",
        "assignments",
        "mean_m_per_expert",
        "host_loop_ms",
        "w4_compute_ms",
        "attn_ms",
        "shared_ms",
        "hc_ms",
        "router_ms",
        "gpu_idle_frac",
        "moe_wall_ms",
    )}, indent=2))
    print("wrote", path)
    return out


if __name__ == "__main__":
    run()
