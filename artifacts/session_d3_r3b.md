# Session Report — R3-B: Strong Position Weighting = Positive
# Breadth/Depth Mechanism (S1 0.333→0.381, Depth Retained);
# Capability Promotion NOT YET

**Date:** 2026-09-11
**Roadmap:** R3-B position-weighted decision CE (§34–§35)

## Full R3 comparison (DEV21, closed loop, exact-root convention)

| arm | mean | LOO-best | median | trim | S1 | S2 | S4 | deep | acc | J_BD |
|---|---|---|---|---|---|---|---|---|---|---|
| A2 reference | 1.762 | baseline | — | — | **0.429** | 0.333 | 0 | 2 | 9/21 | — |
| Bridge R2 | 2.333 | 1.650 | 1.0 | 1.505 | 0.333 | 0.238 | 0.143 | 15 | 7/21 | 2.483 |
| c025 (R3-A best) | 2.571 | 1.900 | 1.0 | 1.385 | 0.333 | 0.286 | 0.143 | 15 | 7/21 | 2.876 |
| pm (mild pos-w) | 2.143 | 1.750 | 1.0 | 1.385 | 0.381 | 0.238 | 0.095 | 9 | 8/21 | 2.369 |
| **ps (strong pos-w)** | **2.667** | **2.000** | 1.0 | **1.539** | **0.381** | **0.333** | **0.143** | **15** | **8/21** | **2.952** |

(Columns: trim = TRUE 20% trimmed mean (4 dropped each side, per
mosaicvm/bench/d3_stats.py — the earlier 1-from-each-end figure
the earlier 1-from-each-end figure was wrong); final column = true S4 = Sk[3] = 3/21.
Deep-survival mass M_deep = sum_{k>=4} S_k is reported
separately: pm 0.429 / ps 0.762. json_13's E[A]=4 = three
accepted tokens → contributes to S1-S3, NOT S4 — hence true S4
is 3/21 not 4/21.)

## ps: positive mechanism; NOT a population win

- **LOO-best 2.000** exceeds A2's 1.762, and mean 2.667 is the
  best observed — BUT the true 20% trimmed mean is **1.539
  (BELOW A2's 1.762)**, median 1.0, and 8/21 accept vs A2's
  9/21. The gain remains partly concentrated (math_15 = 16,
  convo_13 = 6, math_14 = 6, json_13 = 4 — four prompts ≥ 4).
- **The genuine mechanism finding**: decision-position weighting
  moves S1 0.333 → 0.381 while RETAINING S2 = 0.333 (A2's exact
  value), true S4 = 0.143 > 0, and deepest 15 — the first lever
  that improves early-position breadth without sacrificing
  depth.
- Mechanism gate (S1 ≥ 0.429 ∧ S4 > 0): **S4 PASS, S1 0.048
  short**. Capability gates not crossed. BEST ESTABLISHED BROAD
  BASELINE REMAINS A2.

## The R3-B dose-response

mild (1/(1+j)) → breadth partial (S1 0.381) but depth lost
(deep 9, M_deep 0.429); strong (1/(1+j)²) → breadth SAME (0.381)
and depth fully retained (deep 15, M_deep 0.762). The decision-CE
position weighting (distinct from closed B3 state weighting) is
the first lever that moves S1 without sacrificing depth.

## Interpretation

The remaining S1 gap (0.381 vs 0.429) may live upstream: the
bridge drafts from A2's predicted states, and A2's representation
quality is a PLAUSIBLE bottleneck — NOT a proven ceiling (a
learned bridge can in principle correct information the exact
tail failed to convert; that is the bridge's purpose). Position
weighting has extracted most of the easy downstream gain; R3-D
(joint unfreeze — make the representation decision-aware) is the
highest-value next arm, with R3-C capacity secondary. ps + c025
mix is the frozen base for both. Engineering progress stays
92-93%.

## Artifacts

```text
artifacts/d3_bridge_r3b/bridge_{pm,ps}/bridge.pt
artifacts/d3_bridge_r3b/bridge_{pm,ps}_eval.json
artifacts/session_d3_r3a.md                 (R3-A report)
```
