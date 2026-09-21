# Session Report: BAPM Pilot — EARLY STOP (no gain)

**Date:** 2026-09-20
**Status:** COMPLETE. BAPM 100-step pilot FAILS the early-kill gate.
**Verdict:** dEA -0.014 (CAP-DEV70 B4), -0.057 (SCALE-DEV3 B4). Below the
+0.10 gate. Branch-aware prefix-margin fine-tuning does NOT close the gap.
ps1001 remains the frozen continuation model. BLIND stays SEALED.

---

## 1. What was tried

Branch-Aware Prefix Margin (BAPM): fine-tune ps1001 (100 steps from its
checkpoint) for the exact deployed policy — top1 margin at greedy depths,
top4 margin at the d3 branch depth, stop-grad prefix-survival weighting,
restricted teacher-anchor KL. Sparse candidate head (~17 rows vs 129k) →
4.2 s/step (~15x faster than DepthScale). Reused cached 96,096-row union;
no new capture. Training loss fell cleanly 0.100 -> 0.059.

## 2. Result — no improvement

| set | metric | ps1001 | ps1001_bapm | dEA |
|---|---|---|---|---|
| CAP-DEV70 | B4 E[A] | 3.757 | 3.743 | **-0.014** |
| CAP-DEV70 | B8 E[A] | 3.757 | 3.800 | +0.043 |
| SCALE-DEV3 | B4 E[A] | 4.014 | 3.957 | -0.057 |

Early-kill gate (CAP-DEV70 B4 dEA >= 0.10): **FAIL**.

Per-domain CAP-DEV70 B4: math +0.30, everything else flat or slightly down
(convo -0.30, agent -0.10). The three weak domains that carry the gate miss
(longform 1.50, agent 2.10, coding 2.70) did **not** move. BAPM shifted a
little mass around within already-solved domains; it did not repair the
depth-drift domains that matter.

## 3. Why it failed (diagnosis)

BAPM optimizes the *ranking/margin* of the continuation model's decision.
But CAP-DEV70's decomposition already showed the weak-domain failures are
**UNRANKABLE state drift** — the true token is rank>=16, not a near-miss the
margin loss can nudge into Top4. Training loss fell (margins on the
already-rankable rows improved) but that mass was already accepted; the
drift tail is not a margin problem, so a margin objective cannot reach it.

This confirms, from the opposite direction, the CAP-DEV70 finding: the
residual is **structural continuation drift in long reasoning/generation
trajectories**, not decision calibration. A loss-level fix on the existing
ps features is insufficient.

## 4. Consequence for the roadmap

```text
BAPM (branch-aware margin fine-tune)   FAIL / EARLY STOP
loss-level continuation repair         CLOSED for this gap
ps1001                                 remains frozen continuation model
capability gate (fresh)                still 3.757 (unchanged)
BLIND                                  SEALED
```

The next lever must attack drift STRUCTURALLY, not via reweighting or
margin shaping. Per roadmap, the ordered fallbacks are:

1. **Training-time rollout / on-policy continuation** (AAL-2): train ps on
   its OWN drafted trajectory (not just gold/os/chain teacher states), so
   the drift distribution it fails on at inference is the distribution it
   trains on. This is the smallest structural change that directly targets
   the UNRANKABLE tail. Reuses HIST1001 + A2 rollout machinery; no capture.
2. **Targeted Hydra H2/H3/H4** — direct future-token heads for depths 2-4,
   predicting the token decision (not future h40). Larger change.
3. **Root axis** — still deferred; fresh S1 .543 is not the dominant limiter.

## 5. What did NOT happen (guardrails)

- No domain reweighting (BAPM was the predeclared mechanism-first attempt;
  it failed, but reweighting remains deferred — the drift is not label-fixable).
- BLIND untouched. No fresh authority captured (correctly — no candidate
  cleared the readiness buffer).
- CAP-DEV70 / SCALE-DEV3 used only as burned development diagnostics.

## 6. Artifacts

- `ps1001_bapm.pt` (retained; not promoted)
- `bapm_capdev70.json`, `bapm_scaledev3.json`
- `d3_bapm.py`, `d3_bapm_finish.sh`

## 7. Bottom line

BAPM was the correct cheap first attempt and it cleanly answered the
question: the gap is NOT margin/ranking calibration. It is continuation
state drift in hard trajectories. The honest next lever is **on-policy
rollout training of ps** (train on the drift distribution it actually
fails on), which is a structural fix reusing existing machinery and no new
capture. Engineering unchanged 92-93%; capability still 3.757 on fresh.
