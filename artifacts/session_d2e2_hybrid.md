# Session Report — D2-E2: Hybrid Tail Parity = BITWISE PASS

**Date:** 2026-09-10
**Roadmap:** A-Race Complete (§43–§46)

## E2 result: the production hybrid topology is exact on tested traffic

Topology under test:
```text
GPU source attention/core/router
+ CPU VNNI routed experts (E1-validated adapter)
+ GPU source continuation/shared/head
```

Replaying captured exact h40 trajectories from 4 validation
prompts (convo/math/coding/json, d3_data_105) through both the
pure-GPU source tail and the hybrid:

| comparison | result |
|---|---|
| hybrid h42 vs GPU-reference h42 | **4/4 bitwise, max_abs 0.0** |
| hybrid h42 vs CAPTURED source h42 | **4/4 bitwise, max_abs 0.0** |
| L41/L42 routes | all equal |
| head greedy tokens | all equal |

Classification:
```text
D2-E2 = PASS (bitwise, on tested traffic)
HYBRID TAIL TOPOLOGY = PROMOTED as the D2 production candidate
```

## Interpretation (within evidence limits)

1. On this traffic, CPU VNNI expert execution contributes ZERO
   divergence to the tail output — the executed expert calls all
   fell in E1's 680/684 bitwise class. Combined with E1's
   GPU-vs-CPU-reference 683/684 bitwise result, the old
   all-CPU-tail max_abs 4.0 h42 drift is now localized to the
   NON-expert CPU arithmetic (attention/DSA/compressor/core) —
   which the hybrid topology keeps on GPU.
2. **E1's 4 ULP-boundary expert cases are closed per-case by
   one-at-a-time substitution ablation (D2-E2b), scoped to the
   tested traffic.** Canonical scoping (roadmap S36):
   - ON THE FOUR E2 PROMPTS: the hybrid routed-expert rail
     contributes ZERO final h42 divergence (4/4 bitwise,
     max_abs 0.0; routes and greedy also match).
   - FOR THE FOUR KNOWN E1 DELTAS (L41.w3.e125 M3
     @convo_13/prefill; L41.w1.e254 M1 @convo_13/decode0;
     L42.w3.e205 M1 @math_13/decode6; L42.w2.e180 M4
     @coding_13/prefill): each washes out INDIVIDUALLY — the
     ablation substituted exactly one call per run (triggered
     once each) and h42 stayed bitwise in 4/4.
   - THE HISTORICAL ALL-CPU max_abs=4.0 PROBE: causality remains
     UNLOCALIZED until that exact probe/traffic is replayed
     through the hybrid topology (optional D2-E2c, roadmap S37).
     The E2 prompts are not that probe; expert combos outside
     E1's 684-case sample (~900 of 1536) remain un-enumerated.
   E1 stays recorded as 680/684 at the expert level; the exact
   activations are persisted (d2e1_nonbitwise_activations.pt).

3. No timing claim (D2-P is the next system gate: adapter-
   inclusive source-path timing, then the complete tail wall).

## Method notes

- HybridLinear moves each routed-expert activation to CPU,
  runs the source act_quant (E1: codes bitwise 684/684 GPU/CPU)
  and the VNNI GEMM, returns bf16 to GPU. Everything else in the
  blocks runs unmodified on CUDA.
- The exact-replay discipline is inherited: replays use captured
  real h40 trajectories and tokens, not synthetic states.

## Artifacts

```text
artifacts/d2e2_hybrid_tail.json       4-prompt parity + per-prompt rows
mosaicvm/bench/d2e2_hybrid_tail.py    harness
```

## System-track state after E1+E2

```text
D2-E1  PARTIAL (4 ULP expert cases open; adapter bitwise)
D2-E2  PASS bitwise on tested traffic (4 ULP-case coverage open)
NEXT   D2-P source-adapter timing, then complete tail wall (K1..K32)
THEN   H2 production verifier with the certified hybrid topology
```
