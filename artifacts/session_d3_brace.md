# Session Report — B-Race: Truncation Removed, Depth Learned, Mean Unchanged

**Date:** 2026-09-10
**Roadmap:** B-Race Complete — Preserve Early Fidelity While
Learning Uninterrupted Long-Horizon Dynamics

## Verified results (21 val prompts, exact source tail, K32)

| arm | OS E[A] | CL E[A] | S1 | S2 | S3 | deepest | acc |
|---|---|---|---|---|---|---|---|
| A2 reference | 2.905 | **1.762** | **0.429** | 0.333 | 0 | 2 | 9/21 |
| B0 (instrumented control) | 2.905 | 1.762 | 0.429 | 0.333 | 0 | 2 | 9/21 |
| B1 (relaxed threshold) | 2.190 | 1.762 | 0.333 | 0.238 | **0.048** | **6** | 7/21 |
| B2 (horizon curriculum) | 2.667 | 1.571 | 0.333 | 0.238 | 0 | 2 | 7/21 |
| B3 (prefix weights) | 2.429 | 1.762 | 0.381 | 0.333 | 0.048 | 3 | 8/21 |

B0 reproduces A2 exactly (identical metrics, same seed/recipe) —
clean control. No B arm reaches the CL E[A] ≥ 4 promotion gate;
no B arm improves the mean.

## The mechanism findings (the real deliverable)

1. **B0 confirms the truncation hypothesis by measurement**: the
   0.75 threshold resets 419/2688 positions (15.6%) at round 24,
   mean uninterrupted segment 4.5, max 17, ZERO full-32 chains,
   all 84 prompts reset.
2. **B1 removes the truncation almost entirely**: final-round
   2/2688 resets (0.07%), mean segment 31.23, max 32.
3. **Removing truncation is NOT sufficient for mean E[A]** —
   but it changes the acceptance GEOMETRY: B1 produces the
   project's first deep real run — **6 accepted speculative
   tokens on convo_13 (E[A]=7)** — while sacrificing S1 breadth
   (0.429 → 0.333) and S2 (0.333 → 0.238). A2/B0 never accept
   beyond 2; B1's S3–S6 = 1/21 each.
4. **B2 (simple 2→4→8→16→32 horizon curriculum) REGRESSES**
   (CL 1.571, survival ends after S2). Closed in this form.
5. **B3 (harmonic prefix weighting) shifts some probability
   deeper (S3=1/21, deepest 3) without mean gain.**

## Canonical diagnosis

A2 learns the BEGINNING of trajectories (broad early hits);
B1 learns DEEP autonomous dynamics (one 6-token run). Neither
preserves both. The C-race's dual-rail objective — gold-state
anchor + uninterrupted free-run recovery in every update — is
the direct experiment this tradeoff motivates.

## Next

- **C-race** (roadmap §55): C0 = A2 frozen; C1 = A2 + short
  late relaxed fine-tune; C2 = dual-rail anchored/free-run;
  C3 = DAgger off-manifold recovery with drift-balanced states.
  Promotion: CL E[A] ≥ 4, or clear Pareto improvement (S1 ≥ A2
  AND S3+ > 0 with positive trend).
- **D2-P** in parallel: adapter-inclusive CPU expert-path timing
  (transfer, E4M3 adapter, VNNI GEMM, scale reconstruction, bf16
  return, sync) warm/cold/contended.
- Data scaling deferred until the gate or a clear learning curve
  justifies capture cost.

## Artifacts

```text
artifacts/d3_v1_B{0,1,2,3}/student.pt        checkpoints
artifacts/d3_v1_B{0,1,2,3}_tail.json         export-mode evals
artifacts/d3_v1_B{0,1,2,3}_cl.json           closed-loop evals
artifacts/d3_v1_B{0,1,2,3}_telemetry.json    per-round reset stats
```
