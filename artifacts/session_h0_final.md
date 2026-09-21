# Session Report: H0 FINAL — Feasibility evidence (corrected)

**Date:** 2026-09-20
**Status:** ENGINEERING STOP for the current Exact-30 push is justified. No
tested configuration demonstrates a viable route to single-stream Exact-30. The
long-trace residency model predicts substantial storage traffic at every
evaluated memory budget. **Achieved production throughput is NOT established
here** beyond the reference-capture baseline. This report strictly separates
measured, simulated, conditional, and projected evidence.

---

## 0. Evidence classes (read this first)

- **MEASURED** — wall-clock or hardware counter on this box.
- **SIMULATED** — cache/residency replay over the finite captured trace.
- **CONDITIONAL** — arithmetic bound given a measured input + a stated assumption.
- **PROJECTED** — unimplemented; not evidence of achieved performance.

## 1. MEASURED

- **Reference exact capture: 256 decode tokens in 1697 s = 0.151 tok/s**
  (source torch-shim, includes per-position expert streaming + hook overhead).
  This is the only ACHIEVED end-to-end throughput number on record. It is a
  reference-implementation speed, not an optimized runtime.
- Non-expert core (earlier): 2.92 ms/position (43 layers, experts excluded).
- H2D pinned (LINK0): 12.84 GB/s, flat across 1-256 MB.
- NVMe cold random expert-size reads (fadvise DONTNEED): **0.55 GB/s aggregate,
  1.36 GB/s p50**. Measured on this storage path; NOT confirmed as a hardware
  maximum (queued/coalesced reads not swept).
- GPU-resident single-expert kernel: 0.195 ms M=1, ~0.22 ms M=6 (launch-bound).
- CPU single-expert (bf16->fp32 torch GEMM, rotating 3.2GB bank): 1.17 ms M=1
  p50. This is a numeric-exact reference path, an upper bound; a packed VNNI
  kernel is UNBUILT.
- FP4 lossless compressibility (LC0): zstd-19 = 1.091x aggregate over 105
  sampled tensors.

## 2. SIMULATED (finite 256-token single-session trace)

- Distinct experts touched: **6666 = 89.12 GB** cumulative storage. STILL
  GROWING at trace end (7.3 new experts/position; early 67/pos) — the session
  did not reach a saturated working set within 256 tokens.
- Belady per-cycle misses (W6/stride4 proxy cycles), averaged over this finite
  trace:

  | store (usable) | Belady miss/cyc | compulsory | capacity (avg over trace) |
  |---|---|---|---|
  | 24 GB | 214.8 | 105.7 | 109.2 |
  | 30 GB | 172.2 | 105.7 | 66.5 |
  | 35 GB | 143.5 | 105.7 | 37.8 |
  | 38 GB | 131.6 | 105.7 | 25.9 |

  These are trace-averaged capacity misses, NOT a proven steady state. Longer
  generation could raise or lower them; not measured.

## 3. CONDITIONAL bounds (measured input + stated assumption)

- IF the trace-averaged 25.9 capacity misses/cycle at a 38 GB store held in
  steady state, AND each is served from NVMe at the measured 0.55-1.36 GB/s,
  THEN weight-read time alone is ~255-630 ms/cycle -> a movement-bound ceiling
  of ~6-14 tok/s. This is a CONDITIONAL I/O bound, not an achieved runtime.
- The GPU-residency oracle (EK0b/EK0c) is 3.85x over the 125ms movement gate at
  10 GB even with perfect future knowledge — CONDITIONAL on the W6 proxy.

## 7. Completed measurements (were pending)

**1. Compressed-size sensitivity (long trace, time-windowed).** LC0 1.091x
applied as both larger effective store and smaller transfer. Belady misses
(single-session trace) with first/second-half split:

| store x ratio | eff GB | miss avg | miss 1st-half | miss 2nd-half | coldGB/cyc |
|---|---|---|---|---|---|
| 30 x1.0 | 30.0 | 172.2 | 199.9 | 145.3 | 2.30 |
| 38 x1.0 | 38.0 | 131.6 | 176.4 | 88.1 | 1.76 |
| 38 x1.091 | 41.5 | 122.3 | 176.4 | 69.9 | 1.50 |

Second-half misses FALL (176->70 at 38GBx1.091): the working set partially
saturates, so trace-averaged misses OVERSTATE steady state. Compression helps
modestly (122 vs 132 avg). Neither reaches the ~104-expert movement gate.

**2. NVMe bandwidth range (measured, fadvise DONTNEED).** Random expert-size
reads 0.55 GB/s; coalesced 64 MB reads **1.61 GB/s**. Use 0.55-1.61 GB/s for
conditional bounds.

**3. Conditional I/O ceilings** (measured misses x measured NVMe):
- best (38GBx1.091, 2nd-half 69.9 miss, coalesced 1.61): **6.3 tok/s**
- mid (trace-avg 122 miss, p50 1.36): **3.1 tok/s**
- worst (30GB, 1st-half 200 miss, random 0.55): **0.8 tok/s**

**Achieved (MEASURED), only real number: reference shim 256 tok / 1697 s =
0.151 tok/s.** No faster exact path is implemented; this is the baseline until
one is built.

## 8. Remaining (if pursued later)

Build a parity-verified optimized exact path (grouped-GEMM resident MoE +
NVMe/CPU tier for capacity misses) and measure it end-to-end. The conditional
bounds above (0.8-6.3 tok/s I/O-limited) predict it stays well under 30 unless
the session working set is made resident — which the 89 GB (growing) trace says
it cannot be at <=41.5 GB. That is the specific, measured obstacle.

## 5. Corrections applied vs the prior draft

- Removed the "achieved 6-14 tok/s" claim — that was a CONDITIONAL I/O bound.
  The only ACHIEVED number is 0.151 tok/s (reference capture).
- Removed "89 GB/cycle" and the "E[A]=24" calculation — 89 GB is CUMULATIVE
  distinct storage over 256 tokens, NOT per-cycle traffic. Per-cycle union is
  ~865 experts (EK0), and per-cycle capacity misses are 26-109 (§2).
- Removed "forever" / proven-steady-state language — misses are trace-averaged.
- Removed the invalid CPU thread-division projection.
- 0.55 GB/s retained as a MEASURED storage-path result, not a hardware max.
- 1.091x compression sensitivity on the long trace: PENDING (§7 next step),
  not asserted.

## 6. Defensible conclusion

> No tested configuration demonstrates a viable route to single-stream Exact-30
> on RTX 3060 (12 GB VRAM + ~31 GB RAM + NVMe). The long-trace residency model
> predicts substantial storage traffic (26-109 capacity misses/cycle) under
> every evaluated memory budget up to 38 GB, and the measured storage path
> (0.55-1.36 GB/s) makes that traffic a dominant cost. Achieved production
> throughput remains unmeasured beyond the 0.151 tok/s reference capture.

This justifies an ENGINEERING STOP on the current Exact-30 implementation push.
It does NOT prove every possible exact implementation is physically impossible,
and it does NOT establish a measured optimized ceiling.

## 7. Remaining measurements to finalize the baseline (next task)

1. 1.091x compressed-size sensitivity replay on the persisted long trace
   (both larger effective capacity and smaller transfer), time-windowed misses.
2. NVMe bandwidth range with queued/coalesced representative reads (confirm
   physical I/O), to bound the conditional estimates properly.
3. Best already-implemented, parity-verified exact path end-to-end: output
   tokens / wall time, prefill vs decode, startup vs steady, RAM/VRAM, bytes
   read, implementation identity. If only the reference path exists, report its
   actual speed (0.151 tok/s stands as that baseline until a faster exact path
   is built).

## 8. State

```text
proposer                    FROZEN (E[A] 3.757 fresh) — not the bottleneck
BLIND                       SEALED
exactness                   intact
current Exact-30 push       STOP (no viable tested route)
achieved exact throughput   0.151 tok/s (reference capture; only measured number)
conditional I/O ceiling     ~6-14 tok/s (assumption-dependent, not achieved)
next                        compressed replay + NVMe range + real exact-path timing
```
