# Session Report: H0-B — Exact Expert Execution Wall (CPU + GPU-resident)

**Date:** 2026-09-20
**Status:** Conditional-compute measured. IF the session working set fits
VRAM+RAM (H0-A fit-case), the bottleneck is expert-kernel LAUNCH overhead, not
physics: a standard grouped-GEMM MoE kernel projects to ~30-55 tok/s;
per-expert-launch naive path ~14-20 tok/s. All-CPU is too slow (10 tok/s). The
gate is now KERNEL ENGINEERING + confirming the working set fits.

---

## 1. Machine

CPU: 12th Gen Intel i5-12400F, ISA avx2 + **avx_vnni** (no AVX512/AMX). 6 cores.
GPU: RTX 3060 (measured). All weights host/VRAM resident for these numbers.

## 2. Exact CPU expert wall (M=1..6, rotating 3.2GB bank > LLC)

bf16->fp32 torch GEMM (exact-arithmetic reference path; a packed VNNI-int8
kernel would beat this — it is an UPPER bound):

| M | best p50 ms/expert |
|---|---|
| 1 | 1.17 |
| 2 | 1.58 |
| 4 | 2.94 |
| 6 | 4.45 |

M=1 dominates real routing (65% of calls). Note >6 threads hurts (oversubscribe).

## 3. GPU-resident expert wall (M=1..6)

| M | p50 ms/expert |
|---|---|
| 1 | 0.195 |
| 2 | 0.223 |
| 4 | 0.223 |
| 6 | 0.225 |

**Launch-overhead dominated** (M=1 == M=6). This is the critical fact: batching
expert rows is nearly free, so a grouped/fused MoE kernel amortizes launches.

## 4. Cycle assembly (fit-case: session WS resident, no NVMe)

Per W6 verify cycle: ~865 distinct (layer,expert) kernels; core 17.5ms.

| execution model | cycle ms | tok/s |
|---|---|---|
| all-CPU serial (1806 execs x 1.17ms) | 2113 | 1.8 |
| all-CPU 12-thread parallel | 176 | 19.4 |
| all-GPU-resident, per-expert kernel (865 x 0.195) | 186 | 20.2 |
| **all-GPU-resident, grouped MoE (3 launches/layer)** | **43** | **88** |
| all-GPU-resident, grouped MoE (6 launches/layer) | 68 | 55 |

## 5. What this establishes

Two regimes, decided entirely by whether the single-session working set fits
the combined VRAM+RAM store (the open H0-A question):

- **FITS (aggressive reserves + compression, ~38 GB store vs 32-37 GB WS):**
  no steady NVMe. Expert compute is launch-bound. A standard grouped-GEMM MoE
  kernel (well-understood engineering, exact-preserving) projects **~30-88
  tok/s** depending on launch amortization. **Exact-30 is PLAUSIBLE.** The gate
  is writing that kernel + confirming residency, not physics.
- **DOES NOT FIT (conservative reserves, or long-context WS growth):** the
  shortfall churns through NVMe at 0.55-1.36 GB/s and caps throughput in the
  single digits regardless of kernel quality.

## 6. Honest limitations

- CPU number is bf16->fp32 GEMM (numeric-exact reference), an UPPER bound; a
  real packed-FP4->VNNI-int8 kernel is unbuilt (the prior 149ms was a naive
  numpy strawman, discarded). Parity of a real kernel is deferred to that build.
- GPU "grouped MoE" is a PROJECTION from measured per-kernel launch cost + the
  standard grouped-GEMM structure, NOT a measured fused kernel. It assumes
  weights resident (fit-case). Labeled projection, not measurement.
- The 865-distinct-kernels/cycle figure and the fit question both depend on the
  SHORT-trace working set. The long-trace recovery (next) tightens both.
- W6/stride-4 remains a proxy for real speculative verification requests.
- No cross-layer overlap assumed (layers are sequential); intra-layer thread
  parallelism only.

## 7. Verdict

```text
all-CPU exact experts          ~10-19 tok/s (too slow alone; VNNI could lift)
GPU-resident per-kernel        ~20 tok/s (launch-bound)
GPU-resident grouped MoE       ~30-88 tok/s PROJECTED (exact-preserving kernel)
=> IF working set fits:        Exact-30 PLAUSIBLE, gated on grouped-MoE kernel
=> IF it does not fit:         NVMe-bound single digits, dead
```

Neither H0-A nor H0-B alone establishes Exact-30. Together they say: **the
machine is NOT closed. Exact-30 is plausible IFF (a) the single-session working
set fits VRAM+RAM and (b) a grouped-GEMM resident MoE kernel is built.** Both
are now concrete, testable engineering targets rather than open physics.

## 8. Next

1. **Recover + persist the long route trace** (256-position, every
   (position,layer,expert)) — tightens the working-set-fits question with real
   eviction churn over a sustained session. This is the residency evidence H0-A
   lacked.
2. If WS fits at realistic reserves -> build the grouped-GEMM resident MoE
   kernel (exact) and measure a real cycle. If not -> Exact-30 closed on this
   hardware; deliver measured ceiling.

Artifacts: h0b_cpu_experts.json (+ h0_capacity.json, link0/lc0/lc1).
