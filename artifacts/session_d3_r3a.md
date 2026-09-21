# Session Report — R3-A Clean: Mixture Does Not Recover Breadth;
# c025 Is the First LOO Winner; R3-B Is NEXT

**Date:** 2026-09-11
**Roadmap:** Teacher Canonicalized — R3 Breadth Recovery (§22–§33)

## The complete corrected race (one variable: total-objective gold mass)

| arm | mean | LOO-best | median | trim | S1 | S2 | S4 | deep | acc |
|---|---|---|---|---|---|---|---|---|---|
| A2 reference | 1.762 | baseline | — | — | **0.429** | 0.333 | 0 | 2 | 9/21 |
| Bridge R2 | 2.333 | 1.650 | 1.0 | 1.505 | 0.333 | 0.238 | 0.143 | 15 | 7/21 |
| A0_CE75 (CE-only diag) | 2.190 | 1.500 | 1.0 | 1.363 | 0.333 | 0.238 | 0.143 | 15 | 7/21 |
| c075 (clean 75:25) | 2.333 | 1.650 | 1.0 | 1.505 | 0.333 | 0.238 | 0.143 | 15 | 7/21 |
| c050 (clean 50:50) | 2.333 | 1.650 | 1.0 | 1.505 | 0.333 | 0.238 | 0.143 | 15 | 7/21 |
| **c025 (clean 25:75)** | **2.571** | **1.900** | 1.0 | **1.385** | 0.333 | 0.286 | **0.143** | 15 | 7/21 |

(Columns: trim = TRUE 20% trimmed mean — floor(0.2·21)=4 dropped
each side, not the earlier 1-from-each-end statistic; the final
column is true S4 = Sk[3], not the summed deep-survival mass
M_deep, which is now reported separately: R2 0.667 / c025 0.810.)
J_BD diagnostic (LOO + 0.5·S1 + M_deep): c025 = 2.876.

## Findings

1. **The mixture mechanism-gate FAILS**: S1 = 0.333 in every
   arm regardless of gold mass — gold-state training mass does
   NOT restore A2's first-position breadth. Per roadmap §65 this
   closes mixture tuning as the breadth lever.
2. **c025 is a MODEST LOO/S2/M_deep signal, not a robust
   population win**: LOO-best 1.900 exceeds A2's 1.762, but the
   true 20% trimmed mean is **1.385** (below A2), median 1.0,
   7/21 accept (A2: 9/21), true S4 = 3/21 = 0.143. Deep-survival
   mass M_deep (= sum_{k>=4} S_k, previously mislabeled "S4+")
   = 0.810. The improvement is concentrated (convo_13 3→7,
   math_15 still 16). More predicted-state mass improves error
   correction on some trajectories without broadening
   acceptance.
3. **c075/c050 exactly reproduce the R2 bridge profile** (mean
   2.333, LOO 1.65) — the objective is insensitive in 0.5-0.75
   gold mass but responds below 0.5 toward predicted.
4. A0_CE75 (CE-only weighting diagnostic) is slightly WORSE than
   every clean arm — selective CE rebalancing is not the hidden
   winner; the corrected total-objective weighting was the right
   race.

## Classification per roadmap §65

```text
S1 RECOVERY VIA MIXTURE        CLOSED (no arm >= 0.429)
c025 = current best robust candidate (LOO 1.90 > A2 1.762)
R3-B POSITION-WEIGHTED DECISION CE = NEXT (breadth lever)
R3-D JOINT UNFREEZE also motivated (recurrence may be the
  plausible upstream bottleneck: A2's representation
  quality — NOT a proven ceiling; a learned bridge can in
  principle correct information the exact tail failed to
  convert)
```

## Artifacts

```text
artifacts/d3_bridge_r3a/R3A0_CE75_eval.json      diagnostic
artifacts/d3_bridge_r3a/bridge_c{075,050,025}_eval.json
artifacts/d3_bridge_r3a/bridge_c{075,050,025}/bridge.pt
```
