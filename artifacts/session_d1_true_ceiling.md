# Session Report — D1: L42 Information Ceiling is 100%, Not 76%

**Date:** 2026-09-06
**Roadmap:** Current T1 Trees Closed — Exploit the L40→L42
Information Jump Directly

## The decisive D1 finding

**L42 with the fp32 head achieves Top-1 = 100.0% at every position
and every horizon tested.** The prior 76.1% was an artifact of using
bf16 head weights in the exit scan; the model's actual head path
(fp32) reproduces its own greedy choice perfectly from the L42 state.

| exit | Top-1 (bf16 head, prior) | **Top-1 (fp32 head, TRUE)** |
|---|---|---|
| L38 | 42.8% | 37.9% |
| L40 | 49.1% | 43.5% |
| **L42** | 76.1% | **100.0%** |

L42 at future positions (h2): also 100% Top-1/2/4/8.

## What this means

**The proposer problem is now purely a state-prediction problem.**
The L41/L42 transformation converts h40 into perfect next-token
predictions. The information content of the future h42 state IS the
target's greedy choice — there is no residual entropy at h42.

The architecture:

```text
student: (h40,t, history) → predicted h40,t+k
tail:    predicted h40,t+k → real L41 → real L42 → head → draft logits
```

is information-theoretically sound: if the student predicts h42
perfectly, the draft is 100% correct at every horizon. The entire
E[A] gap comes from state-prediction error, not from any
irreducible token entropy.

## The three remaining questions

1. **Tail robustness** (D1b): does the real L41/L42 tail still
   produce useful predictions when given APPROXIMATE h40 states?
   If the tail is chaotic (small h40 error → garbage logits), the
   student needs very high-accuracy state prediction. If it's robust,
   even a rough h40 prediction could suffice.

2. **Tail cost** (D2): what is the actual wall time of L41/L42
   execution? The tail expert bank (~6.86 GB for 2 layers × 256
   experts) fits entirely in host RAM — VNNI v4-B could execute it
   with zero H2D, zero NVMe.

3. **Student capacity**: can a model of practical size learn to
   predict future h40/h42 states well enough? The current T1
   predicted tokens (not states) and was data-starved. State
   prediction may be an easier learning problem (continuous,
   smooth, 4096-dim target vs 129k-dim vocabulary).

## Immediate next experiments

1. **D1b — tail on predicted states**: take T1's h40 state
   predictions (if available) or noisy h40 states, run the real
   L41/L42 tail, measure the resulting Top-k quality degradation
   curve vs state error.

2. **D2 — tail cost**: benchmark L41/L42 execution via VNNI v4-B
   on the RAM-resident tail bank (~6.86GB, fits in host RAM). At
   M≈3-8 rows/expert, the tail should cost ~2× the single-expert
   VNNI time × 256 experts × 2 layers ≈ 0.5-1.0s — needs
   measurement.

3. **DATA**: generate a large corpus (1000+ prompts) recording
   h40 AND h42 at every position for distillation training.

## The corrected oracle ceiling

With the fp32 head, if the student could perfectly predict future
h42 states, the E[A] would be exactly N (the verification width).
The entire ~24-28 token acceptance requirement reduces to a
state-prediction accuracy problem — a much more tractable
formulation than the token-entropy framing.

## Artifacts

```text
artifacts/d1_true_exit_quality.json    fp32-head exit analysis
```
