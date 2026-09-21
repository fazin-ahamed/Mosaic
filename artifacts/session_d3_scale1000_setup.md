# Session Report — SCALE1000 Setup: SCALE-DEV3 Frozen,
# 301 Train Additions Frozen, Pipeline Armed

**Date:** 2026-09-16
**Roadmap:** SCALE700 STRONG PASS — Fresh SCALE1000
Generalization Phase Is NEXT (§12–§26, §65 steps 2–7)

## Frozen SCALE-DEV3 (authority, val-only)

70 prompts, 10/domain, IDs `{dom}_127`–`{dom}_136`.
Structurally distinct tasks (not template × entity).
Val-only: train split empty in the DEV3 file.

Blocked set (fail-closed, independent re-audit PASS):
stage1, **stage05**, SCALE210, SCALE700, G1/G2
(`routes_overlap_corpus_3060`, `g2_cert`, `g2_calibration`).
ID overlap empty. Normalized-text overlap empty.

```text
artifacts/d3_scale1000/prompts_scale_dev3.json
  sha256 41c5b929560e4605cb66ce139fef761b136d4dad7be3c0fbddac4ac18718eacf
artifacts/d3_scale1000/overlap_audit_dev3.json
  sha256 1bf71c2744f950da2c4348b0fa5d32b4338aae34d1fc5612e9ba52735ff3f061
author mosaicvm/bench/d3_scale1000_author.py
  sha256 40db9a8facac92033b914822539291677486fc5f6f710d88fd82fff21058b435
```

**NO EDITS** after baseline evaluation. Orchestrator
re-checks this sha before baselining.

## Frozen SCALE1000 train additions

301 new prompts, 43/domain, IDs `{dom}_137`–`{dom}_179`.
Authored **after** DEV3 freeze; DEV3 sha pinned in the
train author (`DEV3_SHA`) and re-checked at generation.

Combined capture manifest (val block first for phase order):
371 prompts = 70 DEV3 val + 301 train.

```text
artifacts/d3_scale1000/prompts_scale1000.json
  sha256 a023922b0359e4d013c72b1ab74190af2b34add39293eef06b847623c2edbde5
artifacts/d3_scale1000/overlap_audit_train.json
  sha256 d0399beba7693ee5ac7174ddfcf7210dc150a59e518f5ad7a40c5a0bfc317a84
```

Independent re-audit: train ID/text overlap vs all blocked
sources including frozen DEV3 = empty. 301 unique texts.

Union after teacher: 84 + 126 + 490 + 301 = **1001**
prompts / **32,032** transitions (7 × 143).

## Pipeline armed (fail-closed, phase-aware)

Same SCALE700 pattern, with the two live fixes already in:

- startup grace (zero files ≠ stall)
- partial-file hardening on relaunch
- `load_captures(..., require_both=False)` on eval/census
  (phase-gate val-only dir)

```text
engine       mosaicvm/bench/d3_scale1000_engine.sh
orch         mosaicvm/bench/d3_scale1000_orchestrate.sh
teacher      mosaicvm/bench/d3_scale1000_teacher.py
trainer      mosaicvm/bench/d3_scale1000_train.py
             frozen HIST-1 recipe; n == 1001*32 asserted
eval MODELS  hist1001 / ws1001 added to d3_scale_eval.py
```

Launch (2026-09-16 20:12:55):
- engine pid live; phase A recorder 3723149
- orchestrator waiting on 70 SCALE-DEV3 captures
- capture log: `core loaded / 70 prompts to record`

Sequence:
1. capture 70 DEV3
2. baseline HIST700 + WS700 + census on SCALE-DEV3
3. flag → capture 301 train
4. validate 371
5. teacher 301 gold replays
6. HIST1_1001 then WS1_1001 (frozen recipe)
7. SCALE-DEV3 evals + censuses
8. continuity SCALE-DEV2 + DEV21
9. paired ledgers hist700→hist1001 (primary),
   ws700→ws1001 (secondary)

## Cost (from SCALE700 measured ~340 s/capture)

371 × ~340 s ≈ 35 h capture + teacher/train/eval ≈ 6–8 h
≈ **~42 h** wall. Phase gate (~6–7 h) then 301 train.

## Gates (predeclared, unchanged)

On fresh SCALE-DEV3:
ΔH≥4 ∧ ΔE[A]>0 ∧ S4>0 mechanism;
robust extras; ΔH≥6 strong;
|ΔH|≤2 plateau; ΔH≤−3 fail;
E[A]≥4 → freeze + one-shot BLIND-ADJ.

SCALE-DEV2 is burned. Do not use it as the SCALE1000
authority.

## Next pickup

```text
/tmp/scale1000_engine.log
/tmp/scale1000_orchestrator.log
/tmp/scale1000_capture.log
artifacts/d3_scale1000_train/
  baseline_{hist,ws}700_scaledev3.json (+ censuses)
  hist1001.pt  ws1001.pt
  {hist,ws}1001_scaledev3.json (+ censuses)
  ledger_hist700_vs_hist1001.json   <- PRIMARY
  ledger_ws700_vs_ws1001.json       <- secondary
```
