# Session Report — DD-1 H512: REGRESSION; Capacity Race Closed

**Date:** 2026-09-12
**Roadmap:** Breadth+Depth Mechanism Gate PASSED (§12–§17)

## H512 vs H1024 (only variable: DD hidden capacity)

| metric | H1024 (frozen) | H512 | Δ |
|---|---|---|---|
| standalone S1 | **9/21** | 8/21 | −1 |
| standalone Top-4 | **0.810** | 0.714 | −0.096 |
| splice mean E[A] | **2.762** | 2.619 | −0.143 |
| splice S1 | **9/21** | 8/21 | −1 |
| splice trim20 | **1.692** | 1.539 | −0.153 |
| splice LOO-min | **2.100** | 1.950 | −0.150 |
| DD wall (ms) | 9.55 | 8.70 | −0.85 |
| train S1 | 1.0 | 1.0 | both memorize |

H512 is a clear regression on every acceptance metric with
negligible latency benefit (0.85 ms). Per roadmap §16:
"Regression: S1 < .429 without meaningful system benefit."

## Classification

```text
DD-1 H512 = REGRESSION
DD-1 capacity race = CLOSED (H1024 remains)
H256 = NOT RUN (H512 showed no capacity benefit)
```

Train S1 = 1.0 at BOTH H512 and H1024 — the overfitting is not
a capacity problem. Both model sizes memorize the 2688 training
examples. The val S1 difference (8 vs 9) reflects H1024's
marginally better representation, not a generalization
improvement from smaller capacity.

## Splice WARNING note

The splice evaluator prints "WARNING: DD-1 S1 differs from
standalone!" because it hardcodes 9/21 as the expected value.
H512's standalone S1 is legitimately 8/21 (different model) —
the warning is a false alarm, not a bug. The gate correctly
used the counted S1 (8/21 < 9 = FAIL).

## Next per roadmap §18: boundary-weighted training

Since capacity is closed, the next lever is training emphasis:
increase weight on true speculative-boundary examples (j=0)
while retaining all-position regularization. This avoids
collapsing to only 84 boundary samples while emphasizing the
distribution that matters for S1.

## Artifacts

```text
artifacts/d3_dd1/dd1_h40_h512/dd1.pt
artifacts/d3_dd1/dd1_h40_h512_eval.json
artifacts/d3_dd1_h512_splice_eval.json
```
