# Session Report — Boundary-Weight Race: CLOSED; Root History Is NEXT

**Date:** 2026-09-12
**Roadmap:** H512 Capacity Race CLOSED (§8–§19)

## Boundary-mass race results (only variable: g_b)

| arm | S1 | Top-4 | splice mean | trim20 | LOO | gate |
|---|---|---|---|---|---|---|
| baseline (g_b=0.031) | **9/21** | **0.810** | **2.762** | **1.692** | **2.100** | **PASS** |
| g_b=0.125 | 9/21 | 0.714 | 2.762 | 1.692 | 2.100 | PASS |
| g_b=0.25 | 7/21 | 0.714 | 2.524 | 1.385 | 1.850 | FAIL |
| g_b=0.50 | 8/21 | 0.714 | 2.667 | 1.538 | 2.000 | FAIL |

## Classification (roadmap §19)

```text
BOUNDARY-MASS WEIGHTING = CLOSED
```

- g_b=0.125 (4× natural mass): S1 and canonical splice
  acceptance unchanged, but standalone Top-4 WORSENS
  (0.810 → 0.714) — the optimization is insensitive on
  acceptance but degrades ranking quality.
- g_b=0.25 and g_b=0.50: REGRESS S1, mean, and trim20 —
  distorting the shared representation away from transition-
  level structure hurts generalization.

The boundary examples are already memorized (train S1=1.0 at
every g_b). Increasing their optimization mass does not improve
held-out boundary decisions; it either does nothing or harms
the model's broader state-understanding.

## Interpretation

Capacity and optimization emphasis have been ruled out under
this recipe. The leading hypothesis is now input representation:
the single-state snapshot may not encode sufficient temporal
information for generalization. HIST directly tests this.

## NEXT per roadmap §20-21: exact-root history / state deltas

The TCS-FD scan found delta-own (true h40 state motion) is
6.64× more decision-sensitive than random. The current DD-1
sees only a single state + tokens. Adding the PREVIOUS exact
state or the state DELTA provides trajectory/velocity
information that a single snapshot cannot encode.

```text
HIST-0  current state only            [control = done]
HIST-1  current + previous state      NEXT
HIST-2  current + Δstate              NEXT (co-primary)
```

This is still direct decision (no future-state prediction) —
it enriches the input with committed exact history.

## Artifacts

```text
artifacts/d3_dd1/dd1_h40_bw{0125,025,050}/dd1.pt
artifacts/d3_dd1/dd1_h40_bw{0125,025,050}_eval.json
artifacts/d3_dd1/dd1_h40_bw{0125,025,050}_splice.json
mosaicvm/bench/d3_dd1.py               (boundary-mass support)
```
