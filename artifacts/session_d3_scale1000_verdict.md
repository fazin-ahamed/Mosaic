# Session Report: D3 SCALE1000 Verdict — PLATEAU

**Date:** 2026-09-19
**Status:** COMPLETE. SCALE1000 = PLATEAU_NO_AUTOMATIC_SCALE.
**Capability gate E[A] >= 4:** still OPEN.

---

## 1. Headline

Scaling the training union 700 -> 1001 prompts (+301) did **not** improve
breadth on frozen SCALE-DEV3. Both the HIST and WS ledgers return
`PLATEAU_NO_AUTOMATIC_SCALE`. This is the flattening the roadmap flagged as
possible near ~1000 prompts. No automatic scale to 2000.

## 2. Primary ledger — HIST700 -> HIST1001 (SCALE-DEV3, n=70)

| Metric | HIST700 | HIST1001 | delta |
|---|---|---|---|
| S1 | 46/70 | 45/70 | **-1** |
| E[A] splice mean | — | 2.20 | dEA 0.0 |
| trim20 | — | 2.05 | -0.048 |
| LOO | — | — | 0.0 |
| S4 | — | 0.071 | — |

- Conversions G=8, losses L=9 -> net -1.
- Domain gains: agent +3, longform +3, math +2. Domain losses: coding -3,
  longform -3, convo -1, math -1, retrieval -1. Domains net gain: 2.
- Gates: mechanism_S28 false, robust_S29 false, strong_S30 false,
  **plateau_S32 true**, fail_S33 false.
- Standalone Top-k: 1=0.643, 2=0.814, 4=0.886, 8=0.90, 16=0.943.
  mean_target_rank 18.19, mean_nll 1.637, wall 12.66 ms.

## 3. Secondary ledger — WS700 -> WS1001 (SCALE-DEV3, n=70)

| Metric | WS700 | WS1001 | delta |
|---|---|---|---|
| S1 | 41/70 | 43/70 | **+2** |
| E[A] splice mean | — | 2.16 | dEA +0.057 |
| trim20 | — | — | +0.048 |
| S4 | — | 0.071 | — |

- Conversions G=7, losses L=5 -> net +2.
- Gates: mechanism_S28 false, **robust_S29 true**, strong_S30 false,
  **plateau_S32 true**, fail_S33 false.
- WS gained slightly but did not cross the strong gate; overall still plateau.

## 4. Continuity (no regression off SCALE-DEV3)

| Set | model | standalone S1 | splice mean | splice S1 |
|---|---|---|---|---|
| SCALE-DEV2 (70) | hist1001 | 0.70 | 2.37 | 49/70 |
| LEGACY-DEV21 (21) | hist1001 | 0.476 | 2.71 | 10/21 |
| LEGACY-DEV21 (21) | ws1001 | 0.476 | 2.76 | 10/21 |

DEV21 splice S1 10/21 matches the historical HIST-1 comparator — no continuity
regression from the larger corpus.

## 5. Infrastructure result (all held)

The reboot/OOM that killed the earlier in-RAM run is resolved WITHOUT touching
the scientific recipe:

- **FP32 out-of-core mmap trainer** (`d3_scale1000_train.py`): raw + normalized
  union live in disk-backed FP32 tensors (`cache/*.f32`, 9.8 GB); anonymous RSS
  peak **4.91 GB** (soak gate < 12 GB PASS).
- **Bitwise equivalence proven**: `d3_scale1000_equiv.py` on orig84 — 10/10
  torch.equal (raw xm/xa, mean/std, normalized xm/xa, row order). The mmap path
  is an execution-only transform; ledger comparability intact.
- **bf16 variant rejected** for the primary recipe (would contaminate the
  scaling comparison); preserved as `d3_scale1000_train_bf16.py`.
- **cgroup memory bounds** (systemd --user scope, MemoryHigh=16G MemoryMax=20G):
  trainer dies before host. No OOM, no reboot across the full 600-step run.
- **Checkpointing** every 50 steps to `ckpt/hist_ckpt.pt` (atomic, RNG-complete,
  auto-resume) + per-step MemAvailable watchdog. Persistent logs (no /tmp).
- Steady 23.2 s/step (full-vocab head matmul x 63 chunks/step), ~3.9 h total.

## 6. Provenance

- hist1001.pt sha256 780e2daa732c0369…
- ws1001.pt sha256 3360bb9d62ffd438…
- Union: 84 orig + 126 s210 + 490 s700 + 301 s1000 = 1001 prompts / 32,032 rows.
- Teacher audits (all overlap-empty), captures 371/371 deep-validated.

## 7. Interpretation and next (roadmap sections 28/30)

- Capability gate E[A] >= 4 remains OPEN (splice mean 2.2). Far off.
- Raw breadth scaling (84 -> 210 -> 700 -> 1001) has stopped paying at ~1000.
  Plateau verdict blocks a blind 1001 -> 2000.
- Per roadmap: promote a **new decision representation** rather than more HIST
  tuning or more prompts. Primary fallback: **low-rank direct-logit latent**
  (root features -> q in R^r -> E q, r in {128,256,512}). Alternatively the
  deferred **DepthScale / chained Top-k** requalification.
- Full-reboot root cause still UNKNOWN (cgroup now bounds the trainer regardless).
  If it recurs: `journalctl -b -1 -k` for the prior boot.

## 8. Classification

```text
SCALE1000 capture / teacher / union      COMPLETE
FP32 out-of-core trainer                 PASS (bitwise-equivalent, memory-safe)
HIST700 -> HIST1001 (primary)            PLATEAU (S1 46 -> 45)
WS700 -> WS1001 (secondary)              PLATEAU (S1 41 -> 43, robust gate only)
continuity DEV2 / DEV21                  no regression
capability E[A] >= 4                     OPEN
raw-breadth scaling                      PLATEAUED at ~1000
next lever                               low-rank direct-logit latent / DepthScale
engineering                              92-93%
```
