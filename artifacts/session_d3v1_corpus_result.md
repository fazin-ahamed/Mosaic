# Session Report — D3-v1 105-Prompt Corpus Result: FIRST REAL ACCEPTANCE

**Date:** 2026-09-09
**Roadmap:** 105-Prompt D3-v1 Learnability Run Active — Real Tail
Acceptance Is the Next Decisive Proposer Result

## Headline: E[A] = 1.57 (closed loop) / 1.71 (one-step) — CASE B

The token-conditioned Δh40 student, trained on 84 prompts (2688
transitions) and evaluated through the **real source L41/L42 tail
with FP32 head** on 21 held-out validation prompts (K=32, greedy,
prefix+1 convention), produces the **first speculative acceptance
above the E[A]=1 floor on this evaluator**:

| mode | E[A] | CI95 (prompt bootstrap) | S1 | S2 | S3 |
|---|---|---|---|---|---|
| hold (floor) | 1.00 | [1.00, 1.00] | 0 | 0 | 0 |
| one_step (gold state) | **1.71** | [1.19, 2.29] | 0.286 | 0.286 | 0.048 |
| tf_chain (export) | 1.57 | [1.19, 2.00] | 0.286 | 0.238 | 0.048 |
| **cl_loop (production)** | **1.57** | [1.19, 2.00] | 0.286 | 0.238 | 0.048 |
| tf_loop (true tokens) | 1.57 | [1.19, 2.00] | 0.286 | 0.238 | 0.048 |

Roadmap §33 classification: **CASE B — 1 < E[A] < 4: first genuine
signal.** Production cl_loop: 6 of 21 prompts accept speculative
tokens; per-prompt E[A] = {4, 3, 3, 3, 3, 2} — max 3 accepted
drafts, max production E[A] = 4 (convo_13). The E[A]=6 figure
belongs to one_step on convo_13, which receives the gold previous
state at every step and is therefore diagnostic-only. The previous
PCA/ridge baseline scored exactly 1.0 on this same evaluator.

## The §18 diagnostic decomposition

**ONE_STEP — real local skill.** Given the gold previous state and
the true driver token, the student's predicted state yields
marginal (true-path conditioned) **Top-1 = 37.9%, Top-16 = 75.9%**
through the exact tail. Per-step state rel-L2 0.75-1.00 vs hold
1.02-1.31 (cos 0.57-0.74 vs 0.29-0.44). The token-conditioned
one-step transition is genuinely learned.

**TF_LOOP — rollout drift is the depth-killer.** Chained states
collapse the marginal Top-1 from 0.379 → 0.034 (Top-16: 0.759 →
0.089). Chain state rel-L2 reaches 21-40 within K=32. Acceptance
that survives is front-loaded in the first 1-3 positions where
drift has not yet accumulated.

**CL_LOOP — token feedback adds no additional loss.** cl_loop ==
tf_loop per-prompt (expected: within the accepted prefix the
student's drafts equal the true tokens; the loops diverge only
after the first rejection, which ends the prefix). The §71
token-exposure problem is NOT the current bottleneck.

**E[A] — S_k = 0 beyond k=3.** No prompt accepts more than 5.

## Diagnosis per roadmap decision tree

TWO independent bottlenecks (roadmap §11), not drift alone:
1. **Local transition quality is far from production class.** For
   intuition (constant independent p, K=32): p=0.38 → E[A]≈1.6;
   p=0.70 → ≈3.3; p=0.90 → ≈9.7; p=0.98 → ≈24.3. Exact-30-class
   linear acceptance needs very high local conditional correctness
   unless branching/verifier geometry compensates.
2. **Rollout compounding destroys depth** (Top-1 37.9% → 3.4%).
   The exposure objective reached mixed loss 0.45 and did not
   reach a low-loss solution — but that alone does not
   distinguish insufficient iterations, optimization difficulty,
   capacity, or the re-anchoring curriculum. Heavier exposure is a
   controlled EXPERIMENT (arm A2), not a diagnosed cure.
3. The model underfits (final train loss 0.539; val delta rel-L2
   ~1.23) — width is a candidate lever, also only an experiment
   (arm A3).

Domain pattern: acceptance concentrates in convo (3/3 prompts) and
math (2/3), plus json_15. Coding/agent/retrieval/longform: 0/12.

## Next actions (priority order)

1. **Same-corpus controlled race (roadmap §22)** — A0 = existing
   H512 checkpoint (reproduction control); A1 = H512 longer
   one-step optimization; A2 = H512 heavier rollout exposure;
   A3 = H1024 matched; A4 = H1024 + longer + heavier. Isolates
   optimization vs capacity vs rollout-training effects before
   any new data is authored.
2. **Chunked prediction (§35)**: K4/K8 latent chunks with tail
   feedback between chunks directly attacks the drift problem.
3. **h42/logit auxiliary losses (§69)**: use the stored h42
   trajectories to sharpen tail-relevant directions.
4. **Data scaling to 210 prompts** requires AUTHORING new prompts —
   the 126-prompt manifest is exhausted (84 train used, 21 val, 21
   protected test).
5. **D2-E1 is now unblocked** (GPU free): CPU VNNI expert vs
   source GPU expert parity — the top system-track priority.

## Evaluation integrity

- All 21 prompts passed the exact h42 bit-equal + greedy-label
  replay gate before scoring.
- cl_loop gold-leak guard: structurally enforced + regression-
  tested (test_d3v1_alignment.py).
- No synthetic robustness numbers are used as gates; all acceptance
  above is real-tail, real-student.
- Two evaluator bugs found and fixed before/at the final run:
  TailRunner batch-dim (crashed run 1) and inference-mode KV
  seeding (crashed run 2). Both classes now covered by the
  alignment test's shape-contract asserts.

## A-race monitoring incident (corrected)

An extended-regex error in process checks (`pgrep -f "a\|b"`
matches a literal `|` in POSIX ERE, not alternation) made the
running A-race chain report "dead" while it was alive and
training normally (A1 was progressing 550->600->700 steps). Based
on that false negative, the chain was restarted, creating a
duplicate trainer; the duplicate was killed ~2 min later and the
ORIGINAL chain (never actually dead) preserved. All future
monitoring uses single-pattern pgrep or valid ERE alternation
(`pgrep -af 'x|y'`). No artifacts were lost: training output
writes only at step-end, and the chain script is resume-guarded
per arm. The earlier recorder incidents remain genuine OOMs
(tracebacks in logs; user-confirmed).

## Artifacts

```text
artifacts/d3_v1_corpus/student.pt           trained student (w/ provenance)
artifacts/d3_v1_corpus/diagnostics.json     training + state diagnostics
artifacts/d3_v1_corpus_tail.json            export-mode tail eval (A/B/C views)
artifacts/d3_v1_corpus_cl.json              closed-loop + tf-loop eval
artifacts/d3_data_105/*.pt                  105-prompt corpus (84 train/21 val)
mosaicvm/bench/d3_v1.py                     trainer (train/exposure/export)
mosaicvm/bench/d3_v1_eval.py                closed-loop evaluator
mosaicvm/bench/test_d3v1_alignment.py       S14/S16 regression (passing)
```
