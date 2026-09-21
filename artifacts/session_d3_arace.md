# Session Report — D3 Same-Corpus Controlled Race (A0–A4)

**Date:** 2026-09-10
**Roadmap:** D3-v1 Breaks the Real Acceptance Floor (§22–§25)

## Race setup

Same 84-prompt corpus / 21 val prompts / exact source tail / K32,
identical seeds and evaluation harness; only the marked variable
changes. A0 = the original corpus checkpoint.

| arm | hidden | one-step steps | exposure | final mixed loss |
|---|---|---|---|---|
| A0 | 512 | 1000 | 6×40 | 0.451 |
| A1 | 512 | 3000 | 6×40 | 0.448 |
| A2 | 512 | 1000 | **24×60** | **0.314** |
| A3 | **1024** | 1000 | 6×40 | 0.282 |
| A4 | **1024** | 3000 | **24×60** | 0.224 |

## Results

| arm | OS E[A] | OS Top1 | OS Top16 | TF E[A] | TF Top1 | CL E[A] | CL rel-L2 | S1 | S2 |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 1.714 | 0.379 | 0.759 | 1.571 | 0.034 | 1.571 | 30.2 | 0.286 | 0.238 |
| A1 | 1.714 | 0.321 | 0.685 | 1.571 | 0.025 | 1.571 | **10728** | 0.286 | 0.286 |
| **A2** | **2.905** | **0.418** | 0.769 | **1.762** | **0.152** | **1.762** | **1.42** | **0.429** | 0.333 |
| A3 | 1.952 | 0.438 | 0.774 | 1.524 | 0.046 | 1.524 | 2.15 | 0.333 | 0.190 |
| A4 | 2.048 | 0.427 | **0.775** | 1.667 | 0.144 | 1.667 | 1.23 | 0.333 | 0.286 |

A2 one_step CI95 (prompt bootstrap): [1.62, 4.48].

## Findings (roadmap §25 promotion logic)

1. **Heavier rollout exposure is the dominant lever (A2).**
   6×40 → 24×60 on the SAME one-step model: one_step E[A]
   1.714 → 2.905 (+69%), marginal TF Top1 0.034 → 0.152 (4.5×),
   chained drift rel-L2 30 → 1.42 (20×), S1 0.286 → 0.429.
   Acceptance widened from 6/21 to 9/21 prompts and BROKE THE
   DOMAIN CONCENTRATION: first coding (coding_14), longform
   (longform_13), and math_14 acceptances. Both local and chain
   metrics improved — per §25, this is the "scale it" branch.
   NOTE: A2 wins the COMBINED ACCEPTANCE OBJECTIVE, not every
   diagnostic — A3 has the best OS Top1 (0.438) and A4 the best
   CL rel-L2 (1.23) and OS Top16 (0.775).
2. **Longer one-step optimization HURTS (A1).** OS Top1 drops
   (0.379 → 0.321), chains explode (rel-L2 10728 — the memorizing
   model is maximally brittle off-manifold). The 3000-step schedule
   overfits; 1000 steps is the right one-step regime for H512.
   A4 (H1024 + 3000 + heavy exposure) also underperforms A2 —
   the overfit interaction persists at larger width.
3. **Capacity alone helps local skill and drift but not
   acceptance (A3).** OS Top1 0.438 (best), drift 2.15, but CL
   E[A] 1.524 ≈ A0. At current signal levels width is secondary
   to rollout training.
4. **The exposure objective has plateaued at this schedule**:
   mixed loss 0.486 → 0.371 → 0.327 → 0.312 → 0.314 (rounds
   1/6/12/18/24). Simply adding identical rounds is exhausted.
   HYPOTHESIS (unmeasured): the 0.75 re-anchor threshold binds
   during training and truncates long trajectories — held-out CL
   rel-L2 ≈ 1.4 does NOT establish training reset frequency or
   causality (build_exposure records no reset counts). B0
   (instrumented A2 control) must measure reset counts,
   uninterrupted-rollout lengths, and pre/post-reset drift
   before any threshold-relaxation claim.
5. **The one_step → cl_loop gap is now local quality.** With
   drift tamed (1.42), A2's one_step 2.905 vs cl 1.762 gap is
   dominated by the 0.42 Top1 — matching the dual-bottleneck
   picture (roadmap §11).

Statistical honesty: 21 prompts; A2's CL CI [1.38, 2.14] overlaps
A0's [1.19, 2.00]. The OS improvement (CI upper 4.48) and S1
(9/21 vs 6/21) are the stronger signals. No production claims.

## B-race specification (next, same corpus)

0. **B0 instrumented A2 control FIRST** (roadmap §18): rerun the
   exact A2 recipe with reset telemetry — per-round reset count,
   reset fraction, uninterrupted-rollout-length histogram,
   mean/max pre-reset drift, post-reset drift, reset positions,
   per-prompt/domain. This converts the threshold hypothesis
   into evidence before B1.
1. **B1 threshold curriculum**: 0.75 → 3.0 relaxation across
   rounds — promoted only if B0 telemetry shows resets actually
   truncating useful trajectories.
2. **B2 horizon curriculum**: H = 2 → 4 → 8 → 16 → 32 stages.
3. **B3 prefix-weighted trajectory loss**: early-horizon weights.
4. **B4**: combine only the winning interventions, after
   single-factor results.
5. All arms vs A2 reference; H512/1000-step one-step base
   throughout; promotion gate CL E[A] ≥ 4 → author 210-prompt
   corpus.

## B0 telemetry result (measured, post-race)

The instrumented A2 control (exact recipe, fixed telemetry)
answers the threshold hypothesis:

| round | resets | fraction | seg mean | seg max | pre-reset drift |
|---|---|---|---|---|---|
| 1 | 1483 | 0.552 | 0.77 | 3 | 1.165 |
| 8 | 806 | 0.300 | 2.11 | 9 | 0.949 |
| 16 | 522 | 0.194 | 3.57 | 12 | 0.900 |
| 24 | 419 | 0.156 | 4.51 | **17** | 0.882 |

- **Zero chains ever reach their full 32 positions** (final-round
  histogram tops out at 17; bulk of mass at lengths 0-5).
- All 84 prompts experience resets even in the final round.
- Pre-reset drift hovers at ~0.88-0.94 — chains live at the
  threshold boundary; the model never trains on the rel-L2 ~1.4
  regime its held-out chains actually run in.
- Post-reset drift (0.58) < pre-reset (0.88): re-anchoring works
  as a stabilizer but caps exposure.

CONCLUSION: the re-anchor curriculum measurably truncates all
long trajectories during training. B1 (0.75 -> 3.0 relaxation)
is motivated by measurement. B2/B3 race the alternative axes.

## Artifacts

```text
artifacts/d3_v1_A{1,2,3,4}/student.pt          race checkpoints
artifacts/d3_v1_A{1,2,3,4}_tail.json           export-mode evals
artifacts/d3_v1_A{1,2,3,4}_cl.json             closed-loop evals
artifacts/d3_v1_B0_telemetry.json              B0 reset telemetry
artifacts/d3_v1_B{0,1,2,3}_*                   B-race (running)
```
