# Session Report — D-Race First Pass: h42-Aux Preserves Breadth;
# Direct-h42 Regresses; Proposer Wall Measured

**Date:** 2026-09-11
**Roadmap:** D2 Complete — Downstream-Aligned Drafting Is NEXT
(§29–§41, §62)

## Arms trained (frozen A2 recipe — 1000 steps + 24×60 reset
## exposure; the tested variable is the objective/target only)

| arm | objective | OS E[A] | CL E[A] | S1 | deepest | acc |
|---|---|---|---|---|---|---|
| A2 reference | h40 state MSE | 2.905 | **1.762** | 0.429 | 2 | 9/21 |
| D1aux (λ=0.5) | h40 + same-position h42 aux | 2.333 | 1.714 | **0.429** | 2 | 9/21 |
| DH42 | direct h42 recurrence + FP32 head | 1.333 | 1.238 | ~0.24 | 2 | 5/21 |

Gate 1 (CL ≥ 4): FAIL on both. Gate 2 (Pareto): FAIL — D1aux
preserves S1 but S3+ = 0.

## Findings

1. **D1aux: the h42 auxiliary is breadth-safe but not yet
   acceptance-positive.** At λ=0.5 it preserves A2's S1 = 0.429
   exactly and Top16 marginally improves (0.777 vs 0.769 OS),
   but OS/CL means drop slightly. The aux signal is compatible
   with the h40 objective; λ sweeping, margin/rank terms, and
   the TCS basis (§32) remain untried — this is a first-pass
   single-weight result, not a closed branch.
2. **DH42: direct-h42 regresses at this recipe.** One-step
   state rel-L2 is 4.11 (vs ~1.0 for h40) — the h42 trajectory
   moves far more per step, making the one-step objective
   intrinsically harder at equal capacity. Closed-loop h42
   chains are dynamically unstable (cl rel-L2 ~2e10). 5/21
   prompts accept (max E[A]=3). The "h42 may be easier to
   align" hypothesis is NOT supported at H512/A2-recipe; it
   would need capacity/objective redesign to retry.
3. **Proposer wall measured (co-primary, §39)**: direct-h42
   drafting costs **10.4 ms/position** (student + GPU FP32 head
   + sync) vs **40.8 ms** with the CPU fp32 head vs the measured
   **13.5 ms/position** source-tail marginal. The latency win
   of removing the tail is real but modest at GPU-head
   placement — and the GPU head needs the 2.1 GB fp32 head
   matrix VRAM-resident. The tail-free architecture's value
   must come primarily from acceptance quality, which DH42 did
   not deliver at this recipe.

## R2-1 teacher dataset: generated AND validated

84 train-prompt records (507MB, artifacts/d3_r2_teacher/):
gold + A2-predicted (one-step, chained) h40; exact-tail h42 of
all three variants; top-16 teacher logits, true-token rank and
target-vs-top-competitor margin per position.

POST-HOC AUDIT (all 84 records, 2688 positions):
- gold h42 replay vs capture: **84/84 bit-equal** (int16 view)
- gold greedy vs true next token: **2688/2688 = 100%**
(D1-C ceiling confirmed on the full train split)

VERDICT: **CANONICAL** — the hardened audit artifact
`artifacts/d3_r2_teacher_audit.json` (superseding the earlier
16-char-prefix version) now carries FULL SHA-256 hashes for all
84 records, captures, the A2 checkpoint, the generator, the
auditor, and the source-snapshot identity, plus the
dataset-global aggregate hash (0b32e1c2...) and byte size
(530,933,812). Checker re-run: PASS 84/84 records, 2688/2688
greedy, records_passed greedy-aware, n==84 enforced, fail-closed.
(Generation omitted the inline gate; the post-hoc audit closes
it. Future regenerations should gate inline and use cache_cap >=
96 — the run's cap of 8 thrashed.)

## R2-2 (Tail-Jacobian VJPs): BLOCKED on a differentiability
## contract

SourceTail.replay/logits are @inference_mode, and the source FP4
quantizer uses bucketize/LUT — no gradient flows through the
exact tail under the current API. Any VJP work first requires an
explicit contract choice:
- finite-difference margin probes through the REAL tail (exact
  but ~4096 tail calls per full vector — only viable for low-rank
  bases or Hutchinson estimation),
- a straight-through-estimator surrogate,
- or a differentiable reference reimplementation certified
  against the exact tail.
No VJP code launched; this is the design gate before R2-2.

## Verdict

First D-race pass: **no gate crossing**. The viable continuations
per roadmap §64: (a) D1aux λ-sweep + margin/rank auxiliaries +
TCS basis on the h40 family (breadth-compatible, untried knobs);
(b) direct logit/low-rank prediction (§38); (c) DH42 only with
capacity/objective redesign, deprioritized behind (a)/(b).

## Artifacts

```text
artifacts/d3_v1_D1aux/{student.pt,*_tail.json,*_cl.json}
artifacts/d3_v1_DH42/{student.pt} + d3_v1_DH42_eval.json
mosaicvm/bench/d3_h42_eval.py       direct-head evaluator + wall
mosaicvm/bench/d3_v1.py             aux head + target-state support
```
