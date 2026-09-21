# Session Report — DD-1 First-Decision Specialist: Val S1 = 9/21

**Date:** 2026-09-12
**Roadmap:** Direct Decision for Breadth (S27-S36)

## DD-1 val measurements (21 DEV21 boundary positions, n=21)

| variant | Top-1 (S1) | Top-2 | Top-4 | Top-8 | Top-16 |
|---|---|---|---|---|---|
| DD-1 h42 | 9/21 = 0.429 | 11/21 = 0.524 | 15/21 = 0.714 | 17/21 = 0.810 | 18/21 = 0.857 |
| DD-1 h40 | 9/21 = 0.429 | 11/21 = 0.524 | 17/21 = 0.810 | 17/21 = 0.810 | 17/21 = 0.810 |

Train boundary S1 = 1.0 for both variants — memorization at
H1024 with 2688 examples; the val number is the measurement.

Alignment regression PASS: 2688 positions verified, DD-1 target
at j=0 is toks[S+1] (never the root toks[S]); 0/84 root-value
coincidences.

## Interpretation (scoped to what was measured)

- DD-1 (exact h42[S-1] + root token) achieves 9/21 Top-1 on val.
- DD-1 (exact h40[S-1] + root token) achieves the same 9/21.
- A2 + exact tail (A2's PREDICTED h40[S]) also achieves ~9/21
  on the same DEV21 set (one_step mode from prior evals).
- These are DIFFERENT inputs: DD-1 uses exact boundary states;
  A2→tail uses predicted h40[S] (after processing the root
  token through 41 layers). The numerical coincidence at n=21
  does NOT establish an information-theoretic ceiling.
- The h40 variant's Top-4 = 17/21 = 0.810 is a single-position
  boundary ranking on exact states — NOT a chained drafter's
  Top-k on predicted trajectories. It does NOT meet the
  BranchPack reopen gate (which requires chained Top-k on the
  drafter's own predicted trajectory logits).

## What was NOT tested

- j=0-only training (all positions were used for data
  efficiency; boundary-specific training might differ)
- longer context history (only S-1 state + 2 tokens + position)
- capacity below H1024 (which might generalize better at n=21)
- richer inputs (multiple committed states, longer token
  history)

## Artifacts

```text
artifacts/d3_dd1/dd1_h42/dd1.pt
artifacts/d3_dd1/dd1_h40/dd1.pt
artifacts/d3_dd1/dd1_h42_eval.json
artifacts/d3_dd1/dd1_h40_eval.json
mosaicvm/bench/d3_dd1.py              trainer
mosaicvm/bench/d3_dd1_eval.py         val evaluator (persisted)
```
