# Session Report — T1 Multi-State (WEAK) + Track H Hybrid (1.11s — WEAK)

**Date:** 2026-09-06
**Roadmap:** A1.5 Fully Closed — T1 + Track H Are the Two Remaining
Architectural Gates

Both co-primary gates ran to completion this session. Neither passed.

## Track T1 — temporal multi-state: E[A] = 1.57, P(h2|h1) = 0.128 — WEAK

4-state context (h40,t-3:t) + explicit deltas on top of all validated
T0 pieces. ~43M params, 14 epochs on the existing 1076-window dataset.

| metric | T0 | T1 | Δ |
|---|---|---|---|
| P(h2\|h1) | 0.121 | **0.128** | +6% |
| S2 | 0.0526 | 0.0631 | +20% |
| E[A] | 1.498 | **1.568** | +4.6% |
| KV latency | 22.5ms | 24.0ms | +1.5ms |

**The acceptance-critical conditional barely moved** (0.121 → 0.128).
The multi-state context provides real information (S2/S3 improved
proportionally more than S1), but the conditional quality at h2 is
the bottleneck, not context richness. The chain still dies at h2 for
~87% of h1-correct examples.

**Four architectures tested, P(h2|h1) stuck at 0.12-0.13**:
static (h2 collapses), GRU (destroys seed), T0 causal (0.121),
T1 multi-state (0.128). This convergence suggests the limiting factor
is the target's local conditional entropy at h2, not drafter
information access.

## Track H — real hybrid block: wall = 1.11s — NEEDS ANOTHER GAIN

CPU VNNI v4-B (separate process, zero GIL contention) running
concurrently with the full 43-layer GPU block (imma4).

| measurement | value |
|---|---|
| GPU block (median) | 0.662s |
| CPU rail (68% of expert objects) | 1.110s |
| **Wall time (median)** | **1.113s** |
| CPU effective throughput | **0.246 TFLOPS-eq** |
| Overlap efficiency | 0.612 |

**The CPU rail is the critical path at 1.11s**, not the GPU. The
measured CPU throughput (0.246 TFLOPS-eq under concurrent GPU load)
is between the cold (0.21) and warm (0.378) isolated measurements —
RAM contention with GPU DMA costs ~35% of isolated warm throughput.

Per the gate table: **wall > 1.00s = NEEDS ANOTHER GAIN.**

### Root cause analysis

The 0.68 CPU fraction was the modeled optimum from isolated
measurements. Under real contention:
- CPU effective: 0.246 TF (vs 0.378 isolated warm)
- The optimal split shifts: less work on CPU, more on GPU
- But GPU+H2D for the remaining 32% costs ~2.64 × 0.32 = 0.84s of
  H2D — still slow

**The fundamental problem: even at the best split under contention,
neither rail alone nor their combination reaches sub-0.85s.** The
movement architecture needs either: (a) higher CPU throughput (the
VNNI kernel is DRAM-bound; L3 reuse was not exploited in this test
— same expert re-executed, so it WAS L3-warm... the 0.246 is
contention-degraded warm), or (b) a different GPU/CPU ratio, or
(c) strategic VRAM residency to reduce total expert work.

### Honest caveat

This benchmark measured a SYNTHETIC CPU workload (same expert
repeatedly) — the L3 was fully warm. Production cold-start would be
worse. However, it also measured the GPU WITHOUT H2D for its share
(48-slot resident bank) — production H2D for the GPU rail would be
additional. These two approximations partially offset.

## Where the project stands

Both remaining architectural gates returned WEAK/NEEDS-GAIN:

```text
Proposer:    E[A] = 1.57 (need ~24-28)     — the harder gap
Movement:    wall = 1.11s (need ≤ 0.85s)    — the wider gap
```

The movement gap (1.11 vs 0.85) is ~24% — closer than the proposer
gap (1.57 vs 24, ~15×). Movement optimization paths remain: better
CPU/GPU split under contention, strategic VRAM residency to reduce
CPU work, L3-aware scheduling. The proposer gap is architectural.

## Artifacts

```text
artifacts/t1_multistate_result.json    T1 results
artifacts/track_h_hybrid.json          Track H accounting
mosaicvm/bench/t1_multistate.py        T1 architecture
mosaicvm/bench/track_h_hybrid.py       Track H benchmark
/tmp/tiny_causal_t1.pt                 T1 checkpoint
```
