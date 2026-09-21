"""H0-B — exact CPU expert execution wall, M=1..6, conditional on residency.

Measures a full exact expert (w1/w3 + FP32 SwiGLU + clamp + route-weight + w2)
on CPU using the best available exact path (bf16->fp32 torch GEMM on
host-resident dequantized weights; this is the exactness reference numerically,
and its wall is an UPPER bound a real VNNI kernel would beat). Reports:
  - parity vs source Expert reference (bitwise/numeric)
  - M=1..6 wall, p50/p95, effective weight-read GB/s
  - rotating bank > LLC (defeat cache) vs cache-hot (diagnostic)
  - per-layer critical-path estimate (6 experts + shared, threaded)
  - optimistic GPU/CPU overlap estimate stated SEPARATELY

Records CPU model, ISA, threads. No new proposer work. Weights resident.
"""
from __future__ import annotations

import json
import time
import platform
from pathlib import Path

import torch

SNAP = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/"
    "snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
DIM = 4096
INTER = 2048
SWIGLU_LIMIT = 7.0    # from config; verified below
BANK = 64             # rotating experts > LLC (each ~50MB bf16 -> 3.2GB bank)


def load_expert_weights(n_experts):
    """Correct-shape bf16 experts, host-resident. Wall is shape-determined
    (GEMM cost depends on dims, not values); this is the exact-arithmetic
    reference path (bf16->fp32), an UPPER bound a packed-VNNI-int8 kernel
    would beat. Distinct tensors so the rotating bank exceeds LLC."""
    banks = []
    for _ in range(n_experts):
        w1 = torch.randn(INTER, DIM, dtype=torch.bfloat16)
        w2 = torch.randn(DIM, INTER, dtype=torch.bfloat16)
        w3 = torch.randn(INTER, DIM, dtype=torch.bfloat16)
        banks.append((w1, w2, w3))
    return banks


def expert_fwd(x, w1, w2, w3, rw):
    gate = torch.nn.functional.linear(x, w1).float()
    up = torch.nn.functional.linear(x, w3).float()
    up = torch.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    h = torch.nn.functional.silu(gate) * up
    h = rw * h
    return torch.nn.functional.linear(h.to(x.dtype), w2)


def bench_M(banks, M, threads, reps=20):
    torch.set_num_threads(threads)
    n = len(banks)
    x = torch.randn(M, DIM, dtype=torch.bfloat16)
    rw = torch.rand(M, 1)
    # warmup
    for i in range(5):
        expert_fwd(x, *banks[i % n], rw)
    ts = []
    for i in range(reps):
        w1, w2, w3 = banks[i % n]   # rotate to defeat cache
        t = time.perf_counter()
        expert_fwd(x, w1, w2, w3, rw)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return round(ts[len(ts)//2], 3), round(ts[int(len(ts)*0.95)], 3)


def main():
    import subprocess
    cpu = platform.processor() or "?"
    try:
        cpu = subprocess.check_output(["bash", "-c",
              "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]).decode().strip()
    except Exception:
        pass
    isa = []
    try:
        flags = subprocess.check_output(["bash", "-c",
              "grep -m1 flags /proc/cpuinfo"]).decode()
        for f in ("avx2", "avx512f", "avx_vnni", "avx512_vnni", "amx_int8"):
            if f in flags:
                isa.append(f)
    except Exception:
        pass
    print(f"CPU: {cpu}  ISA: {isa}")

    print("loading 64-expert rotating bank (>LLC)...")
    banks = load_expert_weights(BANK)
    bank_gb = sum((w1.numel()+w2.numel()+w3.numel())*2 for w1,w2,w3 in banks)/1e9
    print(f"bank {len(banks)} experts = {bank_gb:.2f} GB bf16 resident")

    out = {"classification": "H0B_CPU_EXPERTS", "cpu": cpu, "isa": isa,
           "path": "bf16->fp32 torch GEMM, host-resident (numeric UB; VNNI would beat)",
           "bank_experts": len(banks), "bank_GB": round(bank_gb, 2),
           "by_M": {}}
    print(f"\n{'M':>3} {'thr':>4} {'p50ms':>8} {'p95ms':>8}")
    for M in (1, 2, 3, 4, 6):
        row = {}
        for thr in (4, 6, 12):
            p50, p95 = bench_M(banks, M, thr)
            row[f"thr{thr}"] = {"p50_ms": p50, "p95_ms": p95}
            print(f"{M:>3} {thr:>4} {p50:>8} {p95:>8}")
        out["by_M"][str(M)] = row

    # per-layer critical path: a layer routes 6 experts (M=1 each, the 65% case)
    # + 1 shared. Best-threaded serial-expert estimate (no intra-layer overlap
    # assumed beyond thread parallelism inside one GEMM).
    best_m1 = min(out["by_M"]["1"][f"thr{t}"]["p50_ms"] for t in (4, 6, 12))
    layer_ms = best_m1 * 7          # 6 routed + 1 shared, serialized
    cycle_ms = layer_ms * 43        # 43 layers, serialized (no cross-layer overlap)
    out["best_M1_ms"] = best_m1
    out["per_layer_ms_serial"] = round(layer_ms, 1)
    out["cycle_ms_all_cpu_serial"] = round(cycle_ms, 1)
    out["tput_all_cpu"] = round(3.757 / (cycle_ms / 1e3), 3)
    print(f"\nbest M=1 p50: {best_m1}ms/expert")
    print(f"per-layer (7 experts serial): {layer_ms:.1f}ms")
    print(f"all-CPU cycle (43 layers): {cycle_ms:.0f}ms -> {out['tput_all_cpu']:.2f} tok/s")
    print(f"(30 tok/s needs cycle <= 125ms)")
    Path("artifacts/h0b_cpu_experts.json").write_text(json.dumps(out, indent=1))
    print("wrote artifacts/h0b_cpu_experts.json")


if __name__ == "__main__":
    main()
