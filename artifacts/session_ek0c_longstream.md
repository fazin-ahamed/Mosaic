# Session Report: EK0c — Long single-stream confirmation. Ceiling FINAL.

**Date:** 2026-09-20
**Status:** GPU-only Exact-30 is FINAL closed. A real 256-token single-stream
generation shows no material locality improvement over the short prompts;
sustained decoding does NOT develop a VRAM-fittable hot expert set.

---

## 1. What was run

Generated 256 tokens from one prompt through the exact source model
(reference shim, ~6.4s/token), hooking every MoE.gate to log per-position
top-6 routed expert IDs across all 43 layers (decode-shape gated: only the
generated position's route recorded, prefill excluded). Fed the single 256-cycle
stream through the corrected EK0b residency sim (Belady brute-force verified).

## 2. Result — sustained generation, real single stream

Adjacent-position expert reuse: **0.285** (vs 0.24-0.44 on short prompts — no
improvement; if anything mid-range). Best-oracle miss/cycle vs the ≤103 gate:

| VRAM (routed) | LRU | static-hot | Belady | best-oracle |
|---|---|---|---|---|
| 2 GB | 1004 | 884 | 858 | 858 |
| 6 GB | 1004 | 723 | 568 | 568 |
| **10 GB** | 999 | 605 | **397** | **397** |

Best oracle @10GB = **397 misses/cycle vs 103 gate = 3.85x over.** Movement-only
tok/s upper bound (Belady, 10GB, zero expert compute) = **9.1 tok/s.**

## 3. This closes the last caveat

The one open limitation in EK0b was "short 32-token prompts might understate
sustained locality." EK0c refutes it directly: over 256 real consecutive decode
tokens the oracle still misses 3.85x the affordable rate. The 40 score-routed
layers spread across the 256-expert population fast enough that no hot set forms.

```
EK0   cold streaming              896ms vs 125ms      CLOSED
EK0b  short-trace oracle @10GB    384 miss vs 103     CLOSED
EK0c  long-stream oracle @10GB    397 miss vs 103     CLOSED (FINAL)
```

## 4. FINAL verdict: GPU-only Exact-30 is physically impossible on RTX 3060

Proven against an optimal (Belady) oracle with future knowledge, on real
sustained-generation routes, at an optimistic 10GB routed-cache budget. No
cache policy, CUDA kernel, or proposer E[A] can close a 3.85x weight-movement
gap. This is bytes-vs-bandwidth-vs-capacity physics.

## 5. What remains alive (per roadmap): CPU-hybrid bound (H0)

GPU-only is closed; the machine is not yet fully closed. The surviving question:
can VRAM + host-RAM(CPU-executed) + NVMe together serve the ~397 cold
experts/cycle without PCIe? Two cheap bounds decide it:

- **H0-A (capacity oracle):** even assuming zero CPU compute cost and perfect
  scheduling, does VRAM+RAM capacity eliminate enough NVMe/PCIe traffic? The
  working set is 104 GB; host RAM is ~24-31 GB. VRAM(10) + RAM(~24) ≈ 34 GB of a
  104 GB working set → ~67% still on NVMe (2-5 GB/s, slower than H2D). This
  looks likely to fail too, but must be computed exactly.
- **H0-B (throughput):** measured VNNI M=1 expert wall (65% of calls are M=1)
  across the per-layer critical path.

If H0-A's optimistic capacity oracle also gives <30 tok/s, **Exact-30 on this
hardware is fully physically closed** and the honest deliverable is the measured
maximum sustainable exact throughput (~9 tok/s movement-bound upper, less in
practice) with a documented capacity/bandwidth floor.

## 6. State

```text
GPU-only Exact-30              FINAL CLOSED (EK0/EK0b/EK0c, oracle 3.85x over)
long-stream caveat             RESOLVED (adjacent reuse 0.285, no hot set)
movement-only tput UB          ~9 tok/s (10GB oracle, zero compute)
working set                    104 GB  >>  VRAM 12 + RAM ~28
NEXT                           H0-A capacity oracle (VRAM+RAM+NVMe), then H0-B VNNI
proposer / BLIND               frozen / sealed (irrelevant to physics)
engineering                    feasibility-bounding phase; not a % gate
```

## 7. Bottom line

Three independent measurements (cold stream, short-trace oracle, long-stream
oracle) agree: the exact 43-layer DeepSeek-V4 MoE cannot be verified at 30
tok/s on a 12GB/12.9GB-s GPU because each cycle needs ~5-12 GB of distinct
expert weights drawn from a 104 GB working set that fits in neither VRAM nor
host RAM. The remaining CPU-hybrid rail must be bounded next (H0), but the
optimistic arithmetic already suggests the true deliverable is a measured
sub-30 tok/s ceiling with a physics-grade explanation — the first rigorous
answer the project has ever had to "why not 30."
