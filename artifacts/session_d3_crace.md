# Session Report — C-Race: All Arms Fail Both Gates; Generic h40
# Recovery Exposure Is Exhausted as a Scaling Path

**Date:** 2026-09-10
**Roadmap:** D2-P Complete — C2 Bounded Collector / Recovery
Learning Phase (§65–§67)

## C-race results (21 val prompts, exact source tail, K32,
## identical evaluation)

| arm | OS E[A] | CL E[A] | S1 | S2 | S3 | S4+ | deepest | acc |
|---|---|---|---|---|---|---|---|---|
| A2 reference | 2.905 | **1.762** | **0.429** | 0.333 | 0 | 0 | 2 | 9/21 |
| B1 reference | 2.190 | **1.762** | 0.333 | 0.238 | 0.048 | 0.143 | 6 | 7/21 |
| C1 (A2 → late relax) | 1.952 | **1.762** | 0.333 | 0.190 | 0.095 | 0.143 | 6 | 7/21 |
| C2 (collector rail) | 1.667 | 1.333 | 0.238 | 0.095 | 0 | 0 | 2 | 5/21 |
| C3 (DAgger buckets) | 1.619 | 1.429 | 0.238 | 0.190 | 0 | 0 | 2 | 5/21 |

Gate 1 (E[A]_CL ≥ 4): **FAIL** on all arms.
Gate 2 (Pareto: S1 ≥ 0.429 AND S3+ > 0): **FAIL** on all arms —
C1 and B1 reach S4+ = 0.143 but at S1 = 0.333.

## The consolidated finding across seven training variants

A2 (reset), B1 (relax), B2 (horizon), B3 (prefix), C1
(late-relax), C2 (collector), C3 (dagger) — **no exposure
construction exceeds A2's mean, and every variant that gains
depth (S3+ > 0) does so by sacrificing S1 breadth** (0.429 →
0.333 or below). The C2 collector rail — mechanically the
healthiest training distribution (98% retention, zero resets,
genuine uninterrupted trajectories) — REGRESSES both mean and
breadth. The breadth-for-depth tradeoff is robust to how the
off-manifold distribution is constructed, including distributions
the training model never generated.

Canonical per roadmap §67:
```text
GENERIC h40 RECOVERY/EXPOSURE SCALING = CLOSED
```

## What this localizes

The one-step transition (ONE_STEP Top1 ≈ 0.38-0.44) remains the
binding constraint. Rollout-distribution engineering has now been
swept thoroughly; it modulates WHERE acceptance mass lands
(breadth vs depth) but cannot create more of it. The next
proposer lever must improve the LOCAL predictor's downstream
alignment:

1. **Downstream-sensitive supervision** (roadmap §53): h42
   auxiliary loss, FP32 logit KL, target-rank loss — the stored
   h42/logit trajectories exist in every capture.
2. **Alternative branches** (§52): D3-H42 (direct future h42),
   D3-LOGIT (distilled final logits), D3-BRIDGE (learned
   h40→h42 compressor).
3. A2's OS E[A]=2.905 with gold states shows the local ceiling
   is not the input information — it is what plain state-MSE
   teaches the model to preserve.

## C2 collector mechanism: PASS (frozen, roadmap S28)

Round-24 telemetry: threshold 3.0, 2667/2688 positions retained
(0.78% dropped), ZERO resets, 81 full-length 32-step + two 27 +
one 21 trajectories, mean uninterrupted length 31.75, max
collector drift 2.8836. The collector generated a genuinely
uninterrupted, bounded training distribution as designed — the
STUDENT trained on it still regressed, which is what closed the
generic-recovery branch.

## R1 artifacts frozen (pre-R2)

D1aux/DH42 checkpoints + evals: see session_d3_drace1.md hashes.

## C1's note

C1 reproduces B1's acceptance geometry from the A2 checkpoint
(6-token deepest run, S4+ = 0.143, mean 1.762) — the late
fine-tune does not need the full curriculum to shift the
tradeoff. It is retained as the best depth-carrying checkpoint
but is not promoted.

## Artifacts

```text
artifacts/d3_v1_C{1,2,3}/student.pt         checkpoints (frozen)
artifacts/d3_v1_C{1,2,3}_tail.json          export-mode evals
artifacts/d3_v1_C{1,2,3}_cl.json            closed-loop evals
artifacts/d3_c2_preflight.json              STAGE/BOUND diagnosis
artifacts/session_d2p_timing.md             D2-P + pre-flight record
```
4ad820c2cf01f415  artifacts/d3_v1_A2/student.pt
44b9712a51ce98bb  artifacts/d3_v1_C1/student.pt
2591bae8f239c59d  artifacts/d3_v1_C2/student.pt
ad670fab98eaa684  artifacts/d3_v1_C3/student.pt
c7031222b0b88aa5  artifacts/d2_complete_tail.json

## Frozen checkpoints (2026-09-10, pre-D-race)
A2/C1/C2/C3 student.pt + d2_complete_tail.json hashes appended above; D-race trains against this frozen baseline.
