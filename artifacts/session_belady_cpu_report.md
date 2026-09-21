# Session Report — Belady Gate + CPU Dual-Rail Strawman (Tracks R & C)

**Date:** 2026-09-05
**Roadmap:** Stage-1 Closed + Per-Layer Expert Universe Correction

## Track R — lineage audit + Belady upper bound

### Lineage audit (`route_lineage_audit.md`)
- **UNAFFECTED**: all timing-only consumers (TRUE-P2 fits, GU race,
  dispatch split, routed decomposition, wide_verify family) — route
  content doesn't affect timing conclusions.
- **TAINTED**: bootstrap-derived synthetic route distributions,
  frequency-selected cold sets.
- **INVALIDATED**: hotbank study, deadline sim (also had its own
  no-eviction bug), residency oracle outputs.

### Belady verdict (`belady_upper_bound.json`) — the §32 gate

Belady-optimal on genuine (layer, expert) traces, capacities 0.5-20GB:

| capacity | byte hit rate |
|---|---|
| 2GB | 0.428 |
| 4GB | 0.557 |
| 12GB | 0.687 |
| 20GB (all RAM) | **0.692 = ceiling** |

The ceiling is EXACTLY the within-block repeat fraction (2537 unique
objects / 8256 accesses) — the 34.1GB per block is **compulsory
first-touch traffic**. No cache policy can avoid it. Cross-block reuse
(0.47-0.56 measured) would halve it at best — but the prior block's
34.1GB working set exceeds the 20GB RAM tier, so even that is
uncapturable in full.

**PURE RESIDENCY/PREFETCH P3 = CLOSED** (roadmap §75 gate).
Reuse distances: p50=261, p90=2577, p99=5674 accesses.

## Track C — CPU Dual-Rail first measurement (`cpu_expert_bench.json`)

**Naive strawman** (numpy/torch unpack-then-GEMM): **140-185ms per
expert** vs **1.04ms H2D** — ~130× worse. The strawman materializes
67MB of fp32 weights per GEMM (3× RAM traffic of the packed data) and
single-threads the LUT unpack. **Naive CPU Dual-Rail: DEAD.**

**Theoretical analysis** (for the real-kernel decision):

- Block expert FLOPs at N=32 ≈ 0.42 TFLOP; CPU (i5-12400F, 6P+4E,
  AVX2+VNNI, no AVX-512) ≈ 0.1 TFLOPS fp32 effective → all-CPU block
  ≈ 4.2s.
- Best GPU/CPU split balances H2D (12.9GB/s) against CPU compute:
  fp32 kernel → f≈0.39 on CPU → cycle ≈1.6s → **~20 tok/s**.
  VNNI int8 kernel (~0.4 TFLOPS) → f≈0.72 → H2D 0.74s fully
  overlapped → **~30-35 tok/s — marginal, and requires a real
  packed-fp4 VNNI kernel (major engineering, never mind exactness
  validation).**

## The corrected P3 feasibility picture

At N=32 on this hardware, exact V4 with 6-expert routing demands
34.1GB of expert traffic per 32-token block against 12.9GB/s PCIe:

```text
pure GPU + H2D            12.1 tok/s absolute ceiling (perfect
                           proposer, perfect overlap)
+ perfect residency        CLOSED (compulsory union > RAM tier)
+ naive CPU rail           DEAD (130x off)
+ fp32 CPU rail (theory)   ~20 tok/s
+ VNNI int8 rail (theory)  ~30-35 tok/s — the only path to Exact-30
                           visible, marginal, high engineering risk
```

Width does not escape it (N=16: 23.1GB vs 0.43s compute — worse
ratio). Sequential N=1 is likewise movement-bound (3.45GB/token,
87% of cycle).

## Honest project status

The two open gates from the rebase remain open, now with measured
boundaries:

1. **Movement/execution**: residency closed; the VNNI CPU rail is the
   remaining candidate — a decisive engineering bet, not a sure thing.
2. **Proposer**: one-state closed; recurrent/temporal drafters untested.

If the VNNI kernel cannot reach ~0.4 TFLOPS on packed-fp4 expert GEMMs
under real conditions, **Exact-30 on this hardware requires
re-examining the problem itself** (different width/movement trade,
or accepting the ~12-20 tok/s class).

## Artifacts

```text
route_lineage_audit.md          corrupt-corpus consumer audit
belady_upper_bound.json         §32 gate result (residency CLOSED)
cpu_expert_bench.json           strawman + theoretical split analysis
mosaicvm/bench/belady_bound.py, cpu_expert_bench.py
```
