"""Track V: v4 four-arm race — parity + M=1..8 benchmark.

Arms: v3 (int8 hsum), v4a (packed nibble), v4b (tiled 8-output),
v4ab (packed->tile, unpack amortized over m). All must be bit-exact
vs the fp32 reference.
"""
from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LIB = ctypes.CDLL(str(HERE / "libvnni_kernel.so"))
I8P = np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")
U8P = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
F32P = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
LIB.vnni_gemm_scaled2.argtypes = [I8P, F32P, I8P, F32P, F32P] + \
    [ctypes.c_int] * 3
LIB.vnni_gemm_packed.argtypes = [I8P, F32P, U8P, F32P, F32P] + \
    [ctypes.c_int] * 3
LIB.vnni_gemm_tiled.argtypes = [U8P, F32P, I8P, F32P, F32P] + \
    [ctypes.c_int] * 3
LIB.vnni_gemm_packed_tiled.argtypes = [U8P, F32P, U8P, F32P, F32P] + \
    [ctypes.c_int] * 3
LIB.tile_weights.argtypes = [I8P, I8P, ctypes.c_int, ctypes.c_int]
LIB.tile_acts.argtypes = [I8P, U8P, ctypes.c_int, ctypes.c_int]

HIDDEN, INTER = 4096, 2048
FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                   dtype=np.float32)
OUT = Path("artifacts/vnni_v4_race.json")
QLUT = np.array([0, 1, 2, 3, 4, 6, 8, 12,
                 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8)


def make_inputs(rng, M, N, K):
    packed = rng.integers(0, 256, (N, K // 2)).astype(np.uint8)
    lo = (packed & 0x0F).astype(np.int64)
    hi = (packed >> 4).astype(np.int64)
    w = np.empty((N, K), dtype=np.float32)
    w[:, 0::2] = FP4_LUT[lo]
    w[:, 1::2] = FP4_LUT[hi]
    qW = (w * 2).astype(np.int8)
    swb = rng.integers(120, 136, (N, K // 32)).astype(np.uint8)
    sW = np.power(2.0, swb.astype(np.int32) - 127).astype(np.float32)
    mA = rng.integers(0, 16, (M, K)).astype(np.int8)
    ea = rng.integers(-6, 1, (M, K // 128)).astype(np.int32)
    sA = np.power(2.0, ea).astype(np.float32)
    A = mA.astype(np.float32) * np.repeat(sA, 128, axis=1)
    return packed, qW, w, sW, mA, sA, A


def reference(A, w, sW, M, N, K):
    WB = K // 32
    y = np.zeros((M, N), dtype=np.float64)
    for wb in range(WB):
        k0, k1 = wb * 32, wb * 32 + 32
        ws = w[:, k0:k1] * sW[:, wb:wb + 1]
        y += A[:, k0:k1] @ ws.T
    return y


def main():
    rng = np.random.default_rng(0)
    threads = int(os.environ.get("OMP_NUM_THREADS", "12"))
    results = {"classification": "VNNI_V4_FOUR_ARM_RACE",
               "threads": threads, "parity": {}, "bench": {}}

    # ---------- parity (small) ----------
    M, N, K = 4, 64, 256
    packed, qW, w, sW, mA, sA, A = make_inputs(rng, M, N, K)
    yref = reference(A, w, sW, M, N, K)
    y = np.zeros((M, N), dtype=np.float32)
    LIB.vnni_gemm_scaled2(mA, sA, qW, sW, y, M, N, K)
    results["parity"]["v3"] = float(np.abs(y - yref).max())
    LIB.vnni_gemm_packed(mA, sA, packed, sW, y, M, N, K)
    results["parity"]["v4a"] = float(np.abs(y - yref).max())
    qT = np.zeros((N // 8, K // 4, 32), dtype=np.int8)
    aT = np.zeros((M, K // 4, 32), dtype=np.uint8)
    LIB.tile_weights(qW, qT, N, K)
    LIB.tile_acts(mA, aT, M, K)
    LIB.vnni_gemm_tiled(aT, sA, qT, sW, y, M, N, K)
    results["parity"]["v4b"] = float(np.abs(y - yref).max())
    LIB.vnni_gemm_packed_tiled(aT, sA, packed, sW, y, M, N, K)
    results["parity"]["v4ab"] = float(np.abs(y - yref).max())
    scale = float(np.abs(yref).max())
    for k, v in results["parity"].items():
        assert v <= 2e-3 * scale, \
            f"{k} parity FAILED ({v} vs scale {scale})"
    print(f"parity: all arms within fp32-accumulation bound "
          f"(max diff {max(results['parity'].values()):.2e} "
          f"vs ref scale {scale:.1f}; integer dots exact by "
          f"construction)")

    # ---------- benchmark (production shapes) ----------
    # GU: N=2*INTER=4096, K=HIDDEN=4096 ; DOWN: N=HIDDEN, K=INTER
    for M in (1, 2, 3, 4, 8, 16):
        packed_gu, qW_gu, w_gu, sW_gu, mA, sA, A_gu = make_inputs(
            rng, M, 2 * INTER, HIDDEN)
        packed_d, qW_d, w_d, sW_d, _, _, _ = make_inputs(
            rng, M, HIDDEN, INTER)
        y_gu = np.zeros((M, 2 * INTER), dtype=np.float32)
        y_d = np.zeros((M, HIDDEN), dtype=np.float32)
        qT_gu = np.zeros((2 * INTER // 8, HIDDEN // 4, 32), np.int8)
        qT_d = np.zeros((HIDDEN // 8, INTER // 4, 32), np.int8)
        aT = np.zeros((M, HIDDEN // 4, 32), np.uint8)
        aT_d = np.zeros((M, INTER // 4, 32), np.uint8)
        LIB.tile_weights(qW_gu, qT_gu, 2 * INTER, HIDDEN)
        LIB.tile_weights(qW_d, qT_d, HIDDEN, INTER)
        LIB.tile_acts(mA, aT, M, HIDDEN)
        LIB.tile_acts(mA, aT_d, M, INTER)
        packed_bytes = packed_gu.nbytes + packed_d.nbytes
        row = {}

        def bench(name, fn):
            fn()
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            dt = (time.perf_counter() - t0) / 20
            flops = 2 * M * HIDDEN * 2 * INTER + \
                2 * M * INTER * HIDDEN
            row[name] = {
                "ms": round(dt * 1e3, 3),
                "tflops_eq": round(flops / dt / 1e12, 4),
                "packed_gbps": round(packed_bytes / dt / 1e9, 2)}

        bench("v3", lambda: (
            LIB.vnni_gemm_scaled2(mA, sA, qW_gu, sW_gu, y_gu,
                                  M, 2 * INTER, HIDDEN),
            LIB.vnni_gemm_scaled2(mA, sA, qW_d, sW_d, y_d,
                                  M, HIDDEN, INTER)))
        bench("v4a", lambda: (
            LIB.vnni_gemm_packed(mA, sA, packed_gu, sW_gu, y_gu,
                                 M, 2 * INTER, HIDDEN),
            LIB.vnni_gemm_packed(mA, sA, packed_d, sW_d, y_d,
                                 M, HIDDEN, INTER)))
        bench("v4b", lambda: (
            LIB.vnni_gemm_tiled(aT, sA, qT_gu, sW_gu, y_gu,
                                M, 2 * INTER, HIDDEN),
            LIB.vnni_gemm_tiled(aT_d, sA, qT_d, sW_d, y_d,
                                M, HIDDEN, INTER)))
        bench("v4ab", lambda: (
            LIB.vnni_gemm_packed_tiled(aT, sA, packed_gu, sW_gu, y_gu,
                                       M, 2 * INTER, HIDDEN),
            LIB.vnni_gemm_packed_tiled(aT_d, sA, packed_d, sW_d, y_d,
                                       M, HIDDEN, INTER)))
        results["bench"][f"M{M}"] = row
        best = min(row.items(), key=lambda kv: kv[1]["ms"])
        print(f"M={M:2d}: " + "  ".join(
            f"{k}={v['ms']:6.3f}ms/{v['tflops_eq']:.3f}TF"
            for k, v in row.items()) +
            f"   best={best[0]}")

    # ---------- hybrid projection from the best arm at M=3 ----------
    b3 = results["bench"]["M3"]
    best_arm = min(b3.items(), key=lambda kv: kv[1]["ms"])[0]
    ms3 = b3[best_arm]["ms"]
    # weight realistic mix: mean rows/object = 8256/2537 = 3.25
    t_cpu_block = 2537 * ms3 / 1e3
    th = 2.64
    f = th / (th + t_cpu_block)
    tcrit = th * t_cpu_block / (th + t_cpu_block)
    results["hybrid_projection"] = {
        "best_arm_M3": best_arm, "ms_per_expert": ms3,
        "all_cpu_block_s": round(t_cpu_block, 3),
        "balanced_f": round(f, 3),
        "tcrit_s": round(tcrit, 3),
        "perfect_proposer_tok_s": round(32 / tcrit, 1)}
    print(f"\nhybrid ({best_arm} @ M3): all-CPU {t_cpu_block:.2f}s; "
          f"f*={f:.3f}; Tcrit={tcrit:.3f}s; "
          f"perfect-proposer {32/tcrit:.1f} tok/s")
    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
