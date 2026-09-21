# Session Report: Targeted Hydra-D2/D3/D4 — FAIL

**Date:** 2026-09-20
**Status:** Hydra continuation architecture CLOSED. Worse than ps1001 on fresh.
**Verdict:** CAP-DEV70 B1(greedy) 3.100 -> 2.671; B4 3.757 -> 2.929. Direct
sequential token heads do not beat the ps recurrence on unseen prompts.
ps1001 remains frozen best. BLIND SEALED.

---

## 1. What was built

Hydra-D2/D3/D4: sequentially-dependent direct-token heads replacing the ps
recurrence at speculative depths 2-4. q_d = Head_d(committed root h40 state,
sum of preceding draft-token embeddings, depth). Shared trunk + per-depth
adapters, frozen source head geometry, direct token prediction (EAGLE-3 lesson).
Prompt-level 80/20 split to catch memorization; teacher-forced v1.

## 2. Pilot: generalizes but with a train->holdout gap

100-step pilot on 798 train / 203 holdout prompts (Hydra never saw holdout):

| depth | train top1 | holdout top1 | holdout top4 |
|---|---|---|---|
| d2 | 1.00 | 0.586 | 0.719 |
| d3 | 0.997 | 0.335 | 0.611 |
| d4 | 0.999 | 0.389 | 0.581 |

Holdout >> random (0.005) -> heads learn *something* generalizable, but the
train(1.0)->holdout(0.3-0.6) gap shows heavy memorization even with a shared
trunk. Passed the "not flat" kill gate, so Stage-2 was justified.

## 3. Full splice on CAP-DEV70: WORSE than ps1001

Stage-2 (trained on full 1001), spliced HIST root + Hydra d2/d3/d4 + ps1001
tail, evaluated fresh on CAP-DEV70 vs frozen ps1001 baseline:

| B | Hydra E[A] | ps1001 E[A] | delta |
|---|---|---|---|
| 1 greedy | 2.671 | 3.100 | **-0.429** |
| 4 | 2.929 | 3.757 | **-0.829** |
| 8 | 3.086 | 3.757 | -0.671 |

Even at B1 (identical greedy regime, no branching asymmetry) Hydra is strictly
worse. The architecture change did not help; it hurt.

## 4. Why Hydra failed

The ps recurrence carries the A2-predicted continuation STATE forward — a rich
16384-dim signal that, despite drift, encodes far more context than Hydra's
inputs (committed root + a bag-of-preceding-token-embeddings). By discarding the
propagated state in favor of "root + prev tokens," Hydra threw away the very
signal that makes ps1001's fresh d2/d3 decent. Sequential token conditioning did
not compensate. On memorized train it looked fine (1.0); on fresh prompts the
thin conditioning could not generalize.

## 5. The exhausted lever space

```text
more ordinary data (SCALE1000)         PLATEAU
better continuation data (DepthScale)  reached 3.757 fresh
branching (d3 B4)                       +0.66, saturated
loss reshaping (BAPM)                   CLOSED (margin not the gap)
on-train rollout (AAL-2)                VACUOUS (memorized)
off-manifold repair (ps_old OFDM)       CLOSED (forgetting)
continuation architecture (Hydra)       CLOSED (worse; thin conditioning)
```

Every "same information, different loss/architecture/data-manipulation" lever
has now been tried. The consistent signal: ps1001 at 3.757 fresh is the ceiling
of what the 1001-prompt information content supports for this proposer family.

## 6. What is genuinely untested

Only ONE lever remains that changes the *information*, not the manipulation:
**fresh current-policy exposure data** (ExposureBank) — capturing NEW prompts
and harvesting ps1001's own first-reject states on them. This is the only path
that adds information the 1001 corpus does not contain. It costs ~3.5h capture.

Everything cheap and capture-free is now exhausted. The evidence strongly
justifies paying for ExposureBank OR accepting 3.757 and pivoting to the H2
production-economics / P3 track with the current proposer.

## 7. State

```text
ps1001 + d3 B4                 FROZEN BEST, fresh CAP-DEV70 B4 3.757
Hydra                          CLOSED (worse on fresh)
capability gate 4.0            NOT MET (3.757)
gap                            0.243
BLIND                          SEALED
CAP-DEV70                      BURNED
evaluator baseline bug         FIXED (probe now prints delta vs frozen)
engineering                    92-93%
next levers                    (A) ExposureBank fresh capture, or
                               (B) accept 3.757, pivot to H2/P3 economics
```

## 8. Artifacts

- `hydra.pt` (pilot), `hydra_full.pt` (stage-2), `hydra_capdev70.json`
- `d3_hydra.py`, `d3_hydra_probe.py`
- evaluator fix in `d3_depth_branch_probe.py` (capability block vs frozen 3.757/4.014)
