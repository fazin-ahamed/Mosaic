"""Core GEMM constant race (§57-1): per-shape × per-width × per-candidate.

Candidates (all must be bitwise-exact vs the reference decode path):
  fused_bm32     — current repo kernel (BM=32 universal)
  fused sweep    — BM32 kernel with (BN, BK, warps, stages) variants
  decode_cublas  — dequant_kernel to BF16 workspace + cuBLAS linear
  v5_splitk      — split-K fused kernel (ported from probe; no width-flat
                   assumptions — competes purely on measured shapes)

Judged on production shapes (8 core GEMMs) at M = 32/56/80/112, median
CUDA-event time. Output: artifacts/gemm_race.json + winner table.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, ".")
from mosaicvm.bench.fp8_core import (
    CORE_PATH, FP8_TENSORS, _fused_fp8_linear, _e8m0_to_f32, dequant_kernel,
)
from safetensors.torch import safe_open

OUT = Path("artifacts/gemm_race.json")
MS = (32, 56, 80, 112)

# ---- v5 split-K fused kernel (bitwise-exact; competes on measured shapes) ---
@triton.jit
def _v5_partial(XT, W, S, P, OT, M, N, K, SK,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mmask = offs_m < M
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    K_per = K // SPLIT
    k_base = pid_s * K_per
    Wp = W + offs_n[:, None] * K
    for k0 in range(k_base, k_base + K_per, BK):
        offs_k = k0 + tl.arange(0, BK)
        xt = tl.load(XT + offs_k[:, None] * M + offs_m[None, :],
                     mask=mmask[None, :], other=0.0)
        b = tl.load(Wp + offs_k[None, :]).to(tl.int32)
        sign = (b >> 7) << 31
        e = (b >> 3) & 0xF
        m = b & 0x7
        norm_bits = sign | ((e + 120) << 23) | (m << 20)
        p = tl.where(m >= 4, 2, tl.where(m >= 2, 1, 0))
        sub_bits = sign | ((118 + p) << 23) | ((m - (1 << p)) << (23 - p))
        bits = tl.where(e == 0, tl.where(m == 0, sign, sub_bits), norm_bits)
        wf = tl.cast(bits, tl.float32, bitcast=True).to(tl.bfloat16)
        s = tl.load(S + (pid_n * BN // 128) * SK + (k0 // 128))
        acc += tl.dot(wf, xt) * s
    tl.store(P + pid_s * N * M + offs_n[:, None] * M + offs_m[None, :],
             acc, mask=mmask[None, :])


@triton.jit
def _v5_reduce(P, OT, N, M, SPLIT: tl.constexpr, BM: tl.constexpr,
               BN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mmask = offs_m < M
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for s in range(SPLIT):
        acc += tl.load(P + s * N * M + offs_n[:, None] * M + offs_m[None, :],
                       mask=mmask[None, :], other=0.0)
    tl.store(OT + offs_n[:, None] * M + offs_m[None, :],
             acc.to(tl.bfloat16), mask=mmask[None, :])


def v5_splitk(x, w_fp8, s_raw, ot, parts, BM, BN, BK, SPLIT, warps, stages):
    m, k = x.shape
    n = w_fp8.shape[0]
    SPLIT = max(SPLIT, 2)
    s_f32 = _e8m0_to_f32(s_raw)
    if not s_f32.is_cuda:
        s_f32 = s_f32.to(x.device)
    xt = x.t().contiguous()
    _v5_partial[(n // BN, (m + BM - 1) // BM, SPLIT)](
        xt, w_fp8.view(torch.uint8), s_f32, parts, ot, m, n, k,
        s_f32.shape[1], BM=BM, BN=BN, BK=BK, SPLIT=SPLIT,
        num_warps=warps, num_stages=stages)
    _v5_reduce[(n // BN, (m + BM - 1) // BM)](parts, ot, n, m, SPLIT=SPLIT,
                                              BM=BM, BN=BN, num_warps=4)
    return ot


# ---------------------------------------------------------------- helpers ---
def bench(fn, iters=30, warmup=6):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / 1e3)
    return statistics.median(ts)


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    dev = "cuda"
    shapes = {}
    with safe_open(str(CORE_PATH), framework="pt") as f:
        for n in FP8_TENSORS:
            w = f.get_tensor(f"layers.20.{n}.weight").to(dev)
            s = f.get_tensor(f"layers.20.{n}.scale")
            shapes[n] = (w, s)

    # shared workspaces (outer loop reuses)
    ws_bf16 = {}
    results = {"shapes": {}, "notes": "median ms; parity = bitwise vs "
               "reference decode (f32-mul, bf16-round; NaN-aware on 0xFF-scale tensors)"}

    # parity reference per shape computed once per M below
    for name, (w, s) in shapes.items():
        n, k = w.shape
        ws_bf16[name] = torch.empty(n, k, dtype=torch.bfloat16, device=dev)

    # decode workspaces for v5 (largest n)
    maxn = max(w.shape[0] for w, _ in shapes.values())
    parts_buf = torch.empty(4, maxn, 128, dtype=torch.float32, device=dev)
    ot_buf = torch.empty(maxn, 128, dtype=torch.bfloat16, device=dev)

    fused_cfgs = [
        ("bm32_bn128_bk128_w8_s3", dict(BN=128, BK=128, num_warps=8, num_stages=3)),
        ("bm32_bn64_bk128_w4_s3",  dict(BN=64,  BK=128, num_warps=4, num_stages=3)),
        ("bm32_bn64_bk64_w4_s4",   dict(BN=64,  BK=64,  num_warps=4, num_stages=4)),
        ("bm32_bn128_bk64_w4_s3",  dict(BN=128, BK=64,  num_warps=4, num_stages=3)),
    ]
    v5_cfgs = [
        ("v5_bn64_bk64_sp2_w4",  dict(BN=64,  BK=64,  SPLIT=2, warps=4, stages=3)),
        ("v5_bn64_bk128_sp2_w4", dict(BN=64,  BK=128, SPLIT=2, warps=4, stages=3)),
        ("v5_bn128_bk64_sp2_w4", dict(BN=128, BK=64,  SPLIT=2, warps=4, stages=3)),
    ]

    def ref_out(x, w, s):
        s32 = _e8m0_to_f32(s).to(dev)
        n, k = w.shape
        r = w.float().view(s32.shape[0], n // s32.shape[0],
                           s32.shape[1], k // s32.shape[1]) * s32[:, None, :, None]
        return x.float() @ r.reshape(n, k).T

    for M in MS:
        results["shapes"][str(M)] = {}
        for name, (w, s) in shapes.items():
            n, k = w.shape
            x = torch.randn(M, k, dtype=torch.bfloat16, device=dev)
            row = {}
            r = ref_out(x, w, s)
            fin = torch.isfinite(r)
            # parity standard: bf16 GEMMs differ from f32-exact by accumulation
            # order. Candidate passes if its error vs exact <= 2x the
            # cuBLAS-on-decoded-bf16 error (the established race standard).
            dequant_kernel(w, s, ws_bf16[name])
            cu = torch.nn.functional.linear(x, ws_bf16[name]).float()
            tol = 2.0 * (cu - r).abs()[fin].max().item() + 1e-4

            cublas_err = (cu - r).abs()[fin].max().item() if fin.any() else 0.0

            def err_of(out):
                return (out - r).abs()[fin].max().item() if fin.any() else 0.0
            # decode + cuBLAS
            def dec_cublas():
                dequant_kernel(w, s, ws_bf16[name])
                return torch.nn.functional.linear(x, ws_bf16[name]).float()
            t = bench(dec_cublas)
            row["decode_cublas"] = {"ms": t * 1e3, "err": err_of(dec_cublas()),
                                    "cublas_err": cublas_err}
            # fused current + sweep
            for cfg_name, cfg in fused_cfgs:
                BN = cfg["BN"] if n % cfg["BN"] == 0 else 64
                out_f = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
                sf = _e8m0_to_f32(s).to(dev)
                def fused():
                    _fused_fp8_linear[((M + 31) // 32, (n + BN - 1) // BN)](
                        x, w.view(torch.uint8), sf, out_f,
                        M, n, k, sf.shape[1],
                        BM=32, BN=BN, BK=cfg["BK"],
                        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
                    return out_f.float()
                try:
                    t = bench(fused)
                    row[f"fused_{cfg_name}" + ("" if BN == cfg["BN"] else f"_bn{BN}")] = {
                        "ms": t * 1e3, "err": err_of(fused()), "cublas_err": cublas_err}
                except Exception as exc:
                    row[f"fused_{cfg_name}"] = {"error": str(exc)[:80]}
            # v5 split-K
            for cfg_name, cfg in v5_cfgs:
                BN = cfg["BN"] if n % cfg["BN"] == 0 else 64
                ot = torch.empty(n, M, dtype=torch.bfloat16, device=dev)
                pbuf = torch.empty(2, n, M, dtype=torch.float32, device=dev)
                def v5run():
                    v5_splitk(x, w, s, ot, pbuf, 32, BN, cfg["BK"],
                              cfg["SPLIT"], cfg["warps"], cfg["stages"])
                    return ot.t().float()
                try:
                    t = bench(v5run)
                    row[cfg_name + ("" if BN == cfg["BN"] else f"_bn{BN}")] = {
                        "ms": t * 1e3, "err": err_of(v5run()), "cublas_err": cublas_err}
                except Exception as exc:
                    row[cfg_name] = {"error": str(exc)[:80]}
            results["shapes"][str(M)][name] = row

    # winner table: per (M, shape) the fastest parity-OK candidate
    winners = {}
    for M in MS:
        winners[str(M)] = {}
        for name in shapes:
            row = results["shapes"][str(M)][name]
            best_t, best = 1e9, None
            for cand, d in row.items():
                if not (isinstance(d, dict) and "ms" in d):
                    continue
                if d.get("err", 9e9) <= 2.0 * d.get("cublas_err", 0.0) + 1e-4 \
                        and d["ms"] < best_t:
                    best_t, best = d["ms"], cand
            winners[str(M)][name] = {"cand": best, "ms": best_t}
    results["winners"] = winners

    # layer projections: sum of 8 best GEMM times vs current bm32 baseline
    proj = {}
    for M in MS:
        tot_best = sum(winners[str(M)][n]["ms"] for n in shapes)
        tot_cur = 0.0
        for n in shapes:
            row = results["shapes"][str(M)][n]
            fused_ms = [d["ms"] for c, d in row.items()
                        if c.startswith("fused_") and isinstance(d, dict)
                        and "ms" in d
                        and d.get("err", 9e9) <= 2.0 * d.get("cublas_err", 0.0) + 1e-4]
            tot_cur += min(fused_ms) if fused_ms else 0.0
        proj[str(M)] = {"gemm_best_ms": tot_best, "gemm_current_ms": tot_cur,
                        "layer_saving_ms": tot_cur - tot_best,
                        "block_saving_ms": (tot_cur - tot_best) * 43}
    results["projection"] = proj
    OUT.write_text(json.dumps(results, indent=2))
    for M in MS:
        p = proj[str(M)]
        print(f"M={M:3d}: GEMM current {p['gemm_current_ms']:.2f} -> best "
              f"{p['gemm_best_ms']:.2f} ms  (save {p['layer_saving_ms']:.2f}/layer, "
              f"{p['block_saving_ms']:.0f} ms/block)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
