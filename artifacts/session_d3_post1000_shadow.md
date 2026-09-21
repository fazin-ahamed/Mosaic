# Session Report — Post-SCALE1000 shadow sequence
# (canonical SCALE1000 recipe UNCHANGED)

**Date:** 2026-09-16
**Live capture:** PID 3723149, SCALE-DEV3 phase A. Do not touch.
**This file is not a new master roadmap.** It records a
post-run order. SCALE1000 gates, HIST1001 training, and
the paired ledger stay exactly as armed.

## Live state at write time

```text
captures     41/371
val          41/70
train        0/301
baseline     flag not set
recorder     3723149
SpecCapture  --run-gpu exit 2, top-level torch imports = []
```

## Three bottlenecks (classification only)

1. Exact-data generation wall (~300 s/prompt decode)
2. Root breadth scaling with independent prompts (working)
3. Continuation depth not moving with the same data
   (E[A]≥4 is a depth problem if S1 stays ~0.7)

Do not treat (3) as a SCALE1000 training change.
HIST1001 still trains the frozen HIST-1 root recipe.

## Closed / deferred while 3723149 is alive

```text
Radix-KV                 CLOSED for this run (R=0.175 on 17-token prompts)
SpecCapture GPU replay   SHADOW / UNMEASURED; bitwise gate first
WaveCapture-X            DEFERRED (post-run exactness gate)
CostSpecCapture          DEFERRED
CaptureWave / prefetch   DEFERRED
Fused CE / compile       DEFERRED (update-equivalence)
BDE / ACH / ScoutSelect  POST-SCALE1000
CETNA / EMA / ZKD        POST-SCALE1000
HydraDraft / ABP2        POST-SCALE1000
DepthScale-1001 / EAO    POST-SCALE1000
BranchPack               stays CLOSED until CTRQ on HIST1001
```

## Fail-closed GPU rule (now actually before torch)

`d3_speccapture_killgate.py --run-gpu`:
- pgrep any `stage1_record.py` (not a hardcoded PID)
- exit 2 **before** `import torch`
- verified: `top_level_torch []`, refuse pids=['3723149']

`--prepare-gold-only` sets `CUDA_VISIBLE_DEVICES=""` and
CPU default device. Never imports `stage1_record`.

## Exact next sequence (live, unchanged)

```text
70/70 SCALE-DEV3
→ freeze HIST700 + WS700 baselines + hashes
→ 301 train captures
→ 371/371 validate
→ teacher 301
→ HIST1_1001 (frozen recipe)
→ SCALE-DEV3 primary eval
→ HIST700→HIST1001 paired ledger
→ WS1_1001
→ secondary ledger
→ census
→ SCALE1000 verdict
```

If E[A]≥4 on SCALE-DEV3: freeze proposer, open BLIND once.
If strong pass but E[A]<4: scaling-law fit, no automatic 2000.
If |ΔH|≤2: plateau, stop auto-scale.

## Post-SCALE1000 order (shadow; GPU idle required)

Cheap first. No new source captures until (3) if WaveCapture
fails its bitwise gate.

1. **CTRQ** — chained Top-k on HIST1001's own trajectory
   (K=2,4,8,16; d=1..8). BranchPack stays closed unless
   chained coverage is actually strong.
2. **DepthScale-1001** — same `ps` architecture, 1001 corpus,
   root proposer frozen. Compare HIST1001+ps_old vs
   HIST1001+ps_1001. Then EAO only after that baseline.
3. **WaveCapture-X** exactness on the 20 gold hashes
   (tokens + torch.equal h40 + torch.equal h42). W=2,4,8,16.
   Weight-major residency, M1 arithmetic per row.
4. **CostSpecCapture** — K chosen by expected accepted tokens
   over expert-union + core time. Bitwise gate first.
   If K=2 already fails h40/h42, stop; do not benchmark
   approximate variants.
5. **ABP2** only if CTRQ says chained Top-k survived.
6. HydraDraft-4, then 8 if horizon useful.
7. CETNA (exact-tail certified neighborhoods).
8. Cascade / multi-seed (training is cheap vs capture).
9. ScoutSelect + ACH before the next large corpus.
10. Fused loss / compile / ScaleGrow / ZKD.

## What this does not authorize

- No claim that E[A]≥4 is easy
- No 1.5–2.2× capture speedup
- No SCALE2000 precommit
- No HIST1001 recipe change from any shadow result
- No reopening BranchPack from single-position Top-k
- Engineering remains 92–93%. Capability OPEN. BLIND sealed.
  H2 co-primary.

## Artifacts already on disk

```text
artifacts/d3_scale1000/prefix_audit.json
artifacts/d3_speccapture/gold_hashes.json
mosaicvm/bench/d3_speccapture_killgate.py
artifacts/session_d3_speccapture_shadow.md
```
