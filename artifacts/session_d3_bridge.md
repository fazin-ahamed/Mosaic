# Session Report — R2: Decision Bridge Delivers the Best Proposer
# Yet; λ-Sweep Closed; Teacher Dataset Canonicalized

**Date:** 2026-09-11
**Roadmap:** D-Race First Pass Complete (R2 execution order §68)

## Headline: Decision Bridge MECHANISM breakthrough; capability
## promotion NOT claimed

Frozen A2 h40 recurrence + learned bridge (H1024/D3, margin+KL,
full-vocab CE) + frozen production norm + FP32 head. NO source
tail in the draft path. DEV21, exact-root convention, K32:

| arm | OS E[A] | CL E[A] | S1 | S2 | S4 | deepest |
|---|---|---|---|---|---|---|
| A2 + exact tail (reference) | 2.905 | 1.762 | 0.429 | 0.333 | 0 | 2 |
| D1aux λ-sweep best (0.1) | 2.619 | 1.762 | 0.381 | — | — | — |
| **Decision Bridge** | **2.524** | **2.333** | 0.333 | 0.238 | **0.095** | **15** |

- **MECHANISM PASS**: a learned tail-free decision mapper
  sustains a 15-token exact accepted prefix (math_15 E[A]=16;
  math_14 five) — depth the entire exposure family never
  produced (deepest 6, once). The exact draft tail is NOT
  mandatory for useful speculative decisions.
- **CAPABILITY NOT PROMOTED — the mean is one-prompt-driven**:
  leave-one-out without math_15 = **1.65, BELOW A2's 1.762**;
  CIs overlap (bridge [1.29, 3.90] vs A2 [1.38, 2.14]); S1
  dropped 0.429 → 0.333. Best observed DEV21 mean is the
  bridge's 2.333, but the best ESTABLISHED broad proposer
  remains A2. Engineering progress stays 92-93%.
- Proposer wall **11.3 ms/position** vs the tail's 13.5 —
  cheaper AND it removes the draft tail's RAM/VRAM/sync cost
  entirely. S1 (0.333 vs 0.429) is the R3 target: recover A2
  breadth while keeping bridge depth.
- Neither promotion gate crossed (CL < 4; S1 < 0.429; LOO below
  A2). R3 races one variable at a time: gold-heavy mixture,
  position-weighted decision CE, capacity.

## Method (what made it work)

1. Teacher dataset: 84 prompts × {gold, A2-one-step, A2-chained}
   h40 variants, exact-tail logits/rank/margins — hash-bound
   audited (84/84 bit-equal, 2688/2688 greedy;
   `d3_r2_teacher_audit.json`, fail-closed checker).
2. The first bridge version FAILED instructively: trained on a
   restricted candidate set (top-16 + competitors) it hit 100%
   train accuracy while full-vocab argmax scored 2-7/32 —
   ~129k untrained head rows overtook the argmax. Diagnosis
   (train-split through-eval-path check) isolated the cause;
   the fix is FULL-VOCAB cross-entropy through the real head.
3. Final objective: full-vocab CE + margin vs top-8 competitors
   (softplus, μ=1) + restricted top-16 KL, per-chunk
   forward/backward (chunk 512; frozen head requires_grad_(False);
   one logit block live at a time) — the OOM/retain-graph traps
   are documented in the harness history.
4. Restricted-objective baseline preserved
   (`bridge_restricted_baseline.pt`) as the negative control.

## λ-sweep (R2-3): CLOSED

λ ∈ {0.03, 0.1, 0.25} (+ R1's 0.5): no value beats A2's mean;
λ=0.1 ties (1.762) at lower S1. The h42-auxiliary family cannot
improve A2's acceptance — closed as a standalone lever.

## Artifacts

```text
artifacts/d3_bridge_r2/bridge.pt               trained bridge
artifacts/d3_bridge_r2/bridge_restricted_baseline.pt
artifacts/d3_bridge_r2_eval.json               DEV21 closed-loop
artifacts/d3_r2_teacher_audit.json             hash-bound audit
mosaicvm/bench/d3_bridge.py                    trainer (full-vocab)
mosaicvm/bench/d3_bridge_eval.py               closed-loop evaluator
artifacts/d3_v1_D1aux_l{003,01,025}_*           λ-sweep evals
```

## Teacher dataset: CANONICAL (audit hardened same session)

Full SHA-256 audit artifact: per-record + capture + A2 +
generator + auditor + source-snapshot hashes; dataset aggregate
hash 0b32e1c2..., 530,933,812 bytes, 84/84 records, 2688/2688
greedy, records_passed greedy-aware, checker fail-closed,
re-run PASS (`d3_r2_teacher_audit.json`).

## Next (R3, one variable at a time)

- R3-A gold-heavy bridge mixtures (75:25 / 50:50 / 25:75)
- R3-B position-weighted full-vocab CE
- R3-C capacity (H2048) — only if S1 improves
- R3-D joint unfreeze after the best frozen-A2 recipe
- Robust DEV stats every arm: mean/median/trimmed/LOO/CI
- BLIND-ADJ only after DEV breadth recovery; not on this mean
