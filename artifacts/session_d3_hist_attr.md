# Session Report — HIST-1 Attribution: Moderate Temporal Signal;
# Ranking on HIST-1 Is NEXT

**Date:** 2026-09-12
**Roadmap:** HIST-1 Wins Narrowly (§8–§25)

## Attribution result: MODERATE_TEMPORAL_SIGNAL

Frozen-model analysis, no training, all 21 DEV roots × 6
auxiliary conditions + 21×21 swap matrix + interpolation.

### The core numbers (from the 21×21 matrix and conditions)

**vs the HIST-0M TRAINED CONTROL, the S1 gain is +1 prompt
(json_14 only)** — 9/21 → 10/21. The advisor's distinction is
correct: the A0 condition (zero-aux applied to the HIST-1
model at eval) is NOT the HIST-0M trained model; the HIST-1
model was trained WITH prev-state input, so removing it at
eval degrades it below its trained baseline.

Within the HIST-1 model's own aux sensitivity:
| signal | count | interpretation |
|---|---|---|
| A0→A1 Top1 flips (same model, different aux) | 3 | json_13, json_14, retrieval_15 |
| negative Top1 flips | 0 | no harm |
| positive margin shifts | 10 | history helps half |
| negative margin shifts | 11 | and hurts half |
| diagonal aux advantage (21×21) | **10/21 positive** | mean +0.109 |

### json_14 deep inspection

- Same-domain swap (json_13/json_15 prev states): **do NOT
  rescue json_14** — same-domain template cue ruled out (but
  see the 21×21 matrix below: many cross-domain prev states
  DO rescue it, so the effect is not json_14-specific)
- Diagonal margin 0.039 vs off-diagonal MEDIAN −0.936 (but
  the off-diagonal DISTRIBUTION spans −1.18 to +1.67 with 8
  hits — see the 21×21 section below; the median hides both
  tails)
- Stem-space interpolation (α in learned aux-embedding space):
  margin rises monotonically (−2.38 → −2.23 → −1.81 → −0.86 →
  +0.04) with the flip only at α=1.0 — a smooth CAUSAL signal
  that barely crosses the decision boundary (json_14's true
  margin is just +0.039, the narrowest possible hit)

### json_14 in the 21×21 matrix: NOT unique

json_14's row has **8 HITs** from 8 different prompts' previous
states — math_14 (+1.67), math_13 (+1.56), longform_14 (+1.35),
coding_14 (+1.24), convo_13 (+0.59), convo_14 (+0.10), math_15
(+0.04), and its OWN (+0.04, the WEAKEST hit). Same-domain
json_13/json_15 MISS — ruling out a domain/template cue, but
many cross-domain prev states rescue json_14 with LARGER
margins than its own. The aux signal is NOT
trajectory-unique; it acts as a generic distributional prior
that happens to shift json_14's boundary case correctly for
many (not all) auxiliary inputs.

### The negative side

11 prompts show NEGATIVE margin shifts. History is not
uniformly helpful — in agent_14 (−4.0) and coding_13 (−1.9)
the previous state actively misleads. The mean margin shift
across all prompts is approximately ZERO.

## Pattern classification per roadmap §22–§25
**MODERATE_TEMPORAL_SIGNAL** (frozen from the 21×21 matrix):
- vs the HIST-0M trained control: +1 prompt only (json_14)
- Within the HIST-1 model: 3 A0→A1 aux flips (but these are
  aux-dependence, not 3 gains vs the control)
- 10/21 positive diagonal advantage, mean +0.109
- 11 negative margin shifts — the aux is not uniformly helpful
- json_14's gain is NOT trajectory-unique (8/21 swapped prev
  states also rescue, several with larger margins)
- The mechanism is a generic distributional prior, not unique
  temporal information
- Stem-space interpolation: smooth causal gradient, flip at
  α=1.0 — the signal is real but boundary-riding

## NEXT per roadmap §22: hard-negative ranking on HIST-1

The evidence supports RANKING CALIBRATION AS THE NEXT BEST
LEVER (not "the bottleneck is ranking, not information" —
that is too definitive). Several misses move substantially
toward the target, but the approximately-zero aggregate margin
shift (+10/−11) also says the auxiliary representation itself
is not broadly helpful yet. The clean experiment is hard-
negative ranking WITHOUT changing HIST-1 inputs or capacity.

Conversion-pool accounting (verified from the conditions):
- 10 positive margin shifts total
- 3 positive Top1 flips (json_13, json_14, retrieval_15)
- 3 positive shifts on already-hits (convo_14, json_15,
  longform_13)
- **4 positive-shift cases still incorrect** — the conversion
  pool: agent_15 (+0.10), coding_15 (+5.42, still −16.9),
  longform_14 (+2.09, still −1.64), retrieval_13 (+0.21,
  still −2.66) — all four in weak domains (agent, coding,
  longform, retrieval)

The ranking race must sharpen these decisions WITHOUT
worsening the half of DEV where the auxiliary currently hurts
(11 negative shifts).

Hard-negative ranking (max-margin against the top wrong
candidates) with the HIST-1 input retained is the designated
next lever. Gate: S1 ≥ 11/21.

## Artifacts

```text
artifacts/d3_hist_attribution.json      full attribution data
mosaicvm/bench/d3_hist_attribution.py   harness
```
