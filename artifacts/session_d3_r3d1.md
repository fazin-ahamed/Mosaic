# Session Report — R3-D1 Narrow Unfreeze: REGRESSED; ps Remains Best

**Date:** 2026-09-12
**Roadmap:** R3-A Closed — R3-D Joint Unfreeze NEXT (S23-S26)

## R3-D1 result (A2 head-only unfreeze, lr 3e-5, 200 steps,
## joint with ps bridge recipe)

| stat | ps (frozen control) | D1 (narrow unfreeze) |
|---|---|---|
| mean | **2.667** | 2.143 |
| trim20 | **1.539** | 1.462 |
| LOO-best | **2.000** | 1.900 |
| S1 | 0.381 | 0.381 |
| S2 | **0.333** | 0.286 |
| S4 | 0.143 | 0.143 |
| M_deep | **0.762** | 0.333 |
| deepest | **15** | 6 |
| accepted | 8 | 8 |

Anchor: STABLE (0.4869 → 0.4885, 0.3% drift — no representation
destruction per S25/S26). Training loss converged (0.3465).

## Verdict

D1 REGRESSED the best candidate: depth collapsed (math_15
E[A] 16 → 7; deepest 15 → 6), S2 dropped, mean fell 0.52. S1
did not improve. The head-only unfreeze at this LR moved the
representation in a direction that hurts deep decision quality
without recovering breadth.

Per roadmap S24: D2 (full unfreeze) should NOT proceed from a
negative D1 signal. R3-D is CLOSED for the narrow-unfreeze
variant.

## Standing after R3

```text
BEST ROBUST CANDIDATE   = ps (frozen A2 + bridge)
                         mean 2.667, LOO 2.000, S1 0.381,
                         S4 0.143, deepest 15
BEST BROAD BASELINE     = A2 (mean 1.762, S1 0.429)
MECHANISM GATE          = S4 PASS, S1 0.048 short
CAPABILITY GATE         = not crossed
R3-A mixture            = CLOSED (S1 stuck 0.333)
R3-B position weighting = POSITIVE (S1 0.333→0.381, depth kept)
R3-D1 narrow unfreeze   = REGRESSED
R3-C capacity           = open (next lever)
```

## Artifacts

```text
artifacts/d3_joint_d1/{a2_joint.pt, bridge_joint.pt}
artifacts/d3_joint_d1_eval.json
mosaicvm/bench/d3_joint.py       joint trainer
mosaicvm/bench/d3_stats.py       canonical robust statistics
```
