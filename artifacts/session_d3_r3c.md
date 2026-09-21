# Session Report — R3-C H2048 Capacity: CLOSED (Identical Robust
# Stats, Worse Depth, +38% Latency)

**Date:** 2026-09-12
**Roadmap:** Full R3 Cycle Complete — Bridge Capacity NEXT (§24–§35)

## R3-C result (only variable: H1024 → H2048; frozen ps recipe)

| stat | ps (H1024 control) | H2048 | Δ |
|---|---|---|---|
| mean | **2.667** | 2.381 | −0.29 |
| trim20 | 1.539 | 1.539 | **0** |
| LOO-best | 2.000 | 2.000 | **0** |
| S1 | 0.381 | 0.381 | **0** |
| S2 | 0.333 | 0.333 | **0** |
| S4 | 0.143 | 0.143 | **0** |
| M_deep | **0.762** | 0.476 | −0.29 |
| deepest | **15** | 9 | −6 |
| accepted | 8 | 8 | 0 |
| wall ms/pos | **11.3** | 15.6 | **+38%** |

Every robust statistic is IDENTICAL to the H1024 control.
The only differences: the mean DROPPED (math_15 fell 16 → 10)
and latency increased 38%. 67M parameters bought nothing.

## Classification (roadmap §32 pattern, but stronger)

```text
BRIDGE CAPACITY = CLOSED
```

H2048 did not merely fail to improve — it left every robust
statistic exactly unchanged while degrading the headline mean
and depth. The bridge's S1 ceiling at 0.381 is not a parameter-
count limitation; it is a property of the information A2's
predicted states carry and the decision-CE objective extracts.

## Full R3 closure summary

| arm | result |
|---|---|
| R3-A mixture (75/50/25) | S1 stuck 0.333 — CLOSED |
| R3-B strong position weight | S1 0.333→0.381, depth kept — POSITIVE |
| R3-D1 narrow unfreeze | regressed (deep 15→6) — CLOSED |
| R3-C H2048 capacity | identical robust stats — CLOSED |

**ps (H1024, strong pos-weight, 25:75 mix) remains the frozen
best candidate.** All bridge-side levers are now exhausted; the
S1 gap (0.381 vs 0.429) lives in the representation, not the
mapper.

## NEXT per roadmap §36-40: Tail-Jacobian / TCS

Low-rank finite differences through the exact source tail:
build a downstream-sensitivity basis from actual A2 errors and
true h40 deltas, probe the real tail's decision-margin response,
and train the next student on the tail-sensitive subspace rather
than global h40 MSE. The teacher dataset (canonical, audited)
plus the exact tail (D2-complete) provide all ingredients.

## Artifacts

```text
artifacts/d3_bridge_r3c/bridge_h2048/bridge.pt
artifacts/d3_bridge_r3c/bridge_h2048_eval.json
artifacts/d3_r3_ps_freeze.json            ps frozen baseline
```
