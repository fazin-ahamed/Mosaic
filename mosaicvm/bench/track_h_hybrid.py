"""Track H1: synthetic GPU+CPU contention probe at N=32.

The GPU runs a full 43-layer surrogate block with a resident synthetic
bank while the CPU repeats one synthetic VNNI v4-B GU matrix. No CPU
DOWN/SwiGLU or production H2D is executed. This measures contention,
not a production hybrid verifier or final output throughput.
"""
from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
from queue import Empty
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, ".")

HIDDEN, INTER, N_LAYERS = 4096, 2048, 43
N_OBJ = 43 * 62          # ~2666 (layer,expert) objects per N32 block
EXPERT_MB = 13.4


def cpu_gu_task_flops(rows=3):
    """Only the GU matrix executed by the CPU worker; DOWN is excluded."""
    return 2 * rows * HIDDEN * (2 * INTER)


def cpu_vnni_worker(request_q, result_q, ready_ev, stop_ev):
    """Execute requested counts of synthetic VNNI v4-B GU GEMMs (M=3)."""
    import numpy as np
    LIB = ctypes.CDLL(str(HERE / "libvnni_kernel.so"))
    I8P = np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")
    U8P = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    F32P = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    LIB.vnni_gemm_tiled.argtypes = [U8P, F32P, I8P, F32P, F32P] + \
        [ctypes.c_int] * 3

    rng = np.random.default_rng(42)
    M = 3
    # pre-build ONE expert's tiled weights (representative; each task
    # re-executes on the same data — measures sustained throughput,
    # not cold-start)
    qW = rng.integers(-12, 13, (2 * INTER, HIDDEN)).astype(np.int8)
    qT = np.zeros((2 * INTER // 8, HIDDEN // 4, 32), np.int8)
    for nt in range(2 * INTER // 8):
        for kt in range(HIDDEN // 4):
            for lane in range(8):
                for kk in range(4):
                    qT[nt, kt, lane * 4 + kk] = \
                        qW[nt * 8 + lane, 4 * kt + kk]
    sW = np.ones((2 * INTER, HIDDEN // 32), np.float32)
    mA = rng.integers(0, 16, (M, HIDDEN)).astype(np.int8)
    sA = np.ones((M, HIDDEN // 128), np.float32)
    aT = np.zeros((M, HIDDEN // 4, 32), np.uint8)
    for m in range(M):
        for kt in range(HIDDEN // 4):
            for lane in range(8):
                for kk in range(4):
                    aT[m, kt, lane * 4 + kk] = mA[m, 4 * kt + kk]
    y = np.zeros((M, 2 * INTER), np.float32)

    ready_ev.set()
    while not stop_ev.is_set():
        try:
            msg = request_q.get(timeout=0.1)
        except Empty:
            continue
        if msg is None:
            return
        n = msg
        t0 = time.perf_counter()
        for _ in range(n):
            LIB.vnni_gemm_tiled(aT, sA, qT, sW, y, M,
                                2 * INTER, HIDDEN)
        dt = time.perf_counter() - t0
        result_q.put(("cpu_done", dt, n))


def main():
    mp.set_start_method("spawn", force=True)
    import torch
    from mosaicvm.bench.thermal import warmup_steady
    from mosaicvm.bench.fp8_core import (
        CORE_PATH, HEAD_DIM, CTX as KV_CTX, load_core_resident,
        alloc_workspace, decode_layer)
    from mosaicvm.kernels.mxint import fused_moe_mxint
    from mosaicvm.bench.phase29r import _allocate_packed4_experts
    from safetensors.torch import safe_open

    assert "3060" in torch.cuda.get_device_name(0).lower()
    warmup_steady(90.0)

    # ---- spawn CPU worker ----
    request_q = mp.Queue()
    result_q = mp.Queue()
    ready_ev = mp.Event()
    stop_ev = mp.Event()
    proc = mp.Process(target=cpu_vnni_worker,
                      args=(request_q, result_q, ready_ev, stop_ev))
    proc.start()
    ready_ev.wait()
    print("CPU worker ready")

    # ---- GPU block setup ----
    K = 32
    HC = 4
    with safe_open(str(CORE_PATH), framework="pt") as f:
        core = load_core_resident(f)
    ws0 = alloc_workspace("cuda")
    bank = _allocate_packed4_experts(48)
    from mosaicvm.bench.true_p2_sweep import (
        hc_pre, hc_post, router_source, shared_source, attn_proxy,
        hc_head_apply)
    x_hc = torch.randn(K, HC, HIDDEN, dtype=torch.bfloat16,
                       device="cuda")
    kv_cache = torch.randn(KV_CTX, 1, HEAD_DIM, dtype=torch.bfloat16,
                           device="cuda")
    routes = [(torch.randint(0, 48, (K, 6), device="cuda",
                             dtype=torch.int64),
               torch.rand(K, 6, device="cuda"))
              for _ in range(N_LAYERS)]

    def gpu_block():
        state = x_hc
        for L in range(N_LAYERS):
            lp = core["layers"][L]
            small = lp["small"]
            decode_layer(lp["fp8"], lp["scales"], ws0)
            residual = state
            x, post, comb = hc_pre(
                state, small["hc_attn_fn"].float(),
                small["hc_attn_base"].float(),
                small["hc_attn_scale"].float())
            x = torch.nn.functional.rms_norm(
                x, (HIDDEN,), small["attn_norm"])
            a = attn_proxy(x, ws0, small, K, kv_cache)
            h = hc_post(a, residual, post, comb)
            residual = h
            x, post, comb = hc_pre(
                h, small["hc_ffn_fn"].float(),
                small["hc_ffn_base"].float(),
                small["hc_ffn_scale"].float())
            x = torch.nn.functional.rms_norm(
                x, (HIDDEN,), small["ffn_norm"])
            router_source(x, small["gate_w"], small["gate_bias"])
            y = fused_moe_mxint(
                x, routes[L][0], routes[L][1], bank=bank,
                weight_repr="packed4", crossover=0, imma_k=4,
                swiglu_limit=10.0)
            y = y + shared_source(x, ws0)
            state = hc_post(y, residual, post, comb)
        return state

    # ---- measured hybrid runs ----
    # f_cpu selects only a synthetic GU task count, not a real work split.
    # GPU still runs the full surrogate block.
    # Wall time = max(gpu, cpu) + join overhead.
    f_cpu = 0.68
    n_cpu_tasks = int(N_OBJ * f_cpu)

    results = {"classification": "H1_SYNTHETIC_CONTENTION_GU_ONLY",
               "production_correct": False,
               "cpu_workload": "one warm GU matrix, M3; no DOWN/SwiGLU",
               "gpu_workload": "full surrogate block; no production H2D",
               "N": 32, "f_cpu": f_cpu,
               "n_cpu_gu_tasks": n_cpu_tasks,
               "nominal_expert_objects": N_OBJ}

    # warm-up both rails
    gpu_block(); torch.cuda.synchronize()
    request_q.put(10)  # warm 10 CPU GU tasks
    assert result_q.get()[0] == "cpu_done"

    for trial in range(6):
        # launch both simultaneously
        t0 = time.perf_counter()
        request_q.put(n_cpu_tasks)
        gpu_block()
        torch.cuda.synchronize()
        t_gpu = time.perf_counter() - t0
        # wait for CPU
        msg = result_q.get()
        t_wall = time.perf_counter() - t0
        tag, t_cpu, n_done = msg
        assert (tag, n_done) == ("cpu_done", n_cpu_tasks)
        print(f"trial {trial}: gpu={t_gpu:.3f}s cpu={t_cpu:.3f}s "
              f"wall={t_wall:.3f}s overlap="
              f"{'FULL' if t_wall < max(t_gpu, t_cpu) + 0.05 else 'PARTIAL'}")
        results[f"trial{trial}"] = {
            "gpu_s": round(t_gpu, 4), "cpu_s": round(t_cpu, 4),
            "wall_s": round(t_wall, 4)}

    walls = sorted(results[f"trial{i}"]["wall_s"]
                   for i in range(6))
    med = walls[3]
    # ---- accounting ----
    cpu_flops_total = n_cpu_tasks * cpu_gu_task_flops()
    # measure CPU effective rate from the trials
    cpu_rates = [results[f"trial{i}"]["cpu_s"] for i in range(6)]
    cpu_med = sorted(cpu_rates)[3]
    cpu_tflops = cpu_flops_total / cpu_med / 1e12

    results["accounting"] = {
        "nominal_expert_flops_block_total": 0.42e12,
        "nominal_expert_bytes_block_total": 34.1e9,
        "cpu_gu_flops": cpu_flops_total,
        "nominal_cpu_task_fraction": f_cpu,
        "cpu_effective_tflops_eq": round(cpu_tflops, 4),
        "gpu_block_time_median_s": round(
            sorted(results[f"trial{i}"]["gpu_s"]
                   for i in range(6))[3], 4),
        "cpu_block_time_median_s": round(cpu_med, 4),
        "wall_median_s": round(med, 4),
        "h2d_note": "GPU bank is 48-slot resident (no H2D in timed "
                    "region for the GPU rail; H2D for the "
                    "gpu-assigned expert share would be additional "
                    "in production — this is the CPU-rail isolation "
                    "measurement)",
        "verdict": None,
    }
    results["accounting"]["verdict"] = "CONTENTION PROBE ONLY; H2 REQUIRED"
    print("\n=== ACCOUNTING ===")
    print(json.dumps(results["accounting"], indent=1))
    Path("artifacts/track_h_hybrid_v2.json").write_text(
        json.dumps(results, indent=1))
    stop_ev.set()
    request_q.put(None)
    proc.join()
    print("wrote artifacts/track_h_hybrid_v2.json")


if __name__ == "__main__":
    main()
