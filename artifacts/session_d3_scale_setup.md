# Session Report — SCALE210 Setup: Corpus Authored, Capture
# Marathon Running, Autonomous Pipeline Armed

**Date:** 2026-09-13
**Roadmap:** DTR-R1 CLOSED — Controlled Corpus Scaling Is NEXT
(§10–§23, §64 steps 2–13)

## Completed this session

### 1. DTR-R1 negative frozen (step 1)
`session_d3_dtr.md` + census + eval + hashes (prior phase).

### 2. DEV21 relabeled LEGACY-DEV (step 2)
Continuity only; SCALE-DEV70 becomes the scale-phase
authority. BLIND reserve (21 test prompts, no captures)
untouched.

### 3. SCALE corpus authored (steps 3–4)
196 new prompts, 28/domain, matched to existing domain
styles, programmatically verified disjoint from all 126
existing prompts AND G1/G2 protected texts:

- **SCALE210 train**: {dom}_19..{dom}_36 (18/domain, 126) →
  with the existing 84 = 210 prompts / 6,720 transitions
- **SCALE-DEV70**: {dom}_37..{dom}_46 (10/domain, 70),
  disjoint from train, legacy DEV21, and the blind reserve

Manifest: `artifacts/d3_scale/prompts_scale.json`
**sha256 e97254bc5f6f50cb0907b949ee76cf70b91d21a26ed2a629bc4a480177a3add5**

### 4. Teacher-data fidelity verified
The existing teacher records' gold decisions come from
deterministic SourceTail replay (verified: fresh replay
reproduces `h42_gold_future` **bitwise**, max diff 0.000000).
Direct hc_head+norm+head reproduction from capture h42 drifts
±0.02 logits with rank-16 boundary swaps (true_rank agrees
32/32) — so scale teacher records will use the SAME exact
gold-only replay (7.6 s/prompt → 126 prompts ≈ 16 min),
keeping provenance identical to the existing 84. os/chain
variants are not recorded (not consumed by HIST/WS training).

### 5. Capture marathon armed (step 6)
`stage1_record.py` S1_SOURCE=1 (exact source class), batch 1,
SCG_CACHE_CAP=96: measured **~316–340 s/prompt** → 196
prompts ≈ **17–18.5 h** background job (survives session
ends; recorder skips already-captured prompts → resumable).
Writing to `artifacts/d3_scale_data/`. One capture validated
end-to-end (coding_37: schema, S, h40/h42, split fields).

### 6. Autonomous orchestrator armed (steps 5, 7–13)
`d3_scale_orchestrate.sh` — waits for ALL 196 captures
(sequential GPU: the capture process holds ~9.0 GB and a
concurrent eval OOMs — tested and restructured), then runs
unattended:

1. validate all 196 captures load cleanly (fail-closed)
2. baseline WS1_84 + HIST1_84 on SCALE-DEV70 (§17)
3. `d3_scale_teacher.py` — gold replay records + hash-bound
   audit + overlap checks (§23)
4. `d3_scale_train.py --stage hist` — HIST-1@210, EXACT
   frozen architecture/recipe, only dataset changes (§19, 21)
5. `--stage ws` — WS-1@210 (frozen trunk, same E256/Q recipe)
6. evals: ws210 + hist210 on SCALE-DEV70 (primary) and
   legacy DEV21 (continuity)
7. `d3_scale_ledger.py` — paired G/L flip ledger and gates

### Gates predeclared (§28–§33)
Mechanism: ΔH ≥ 4 ∧ ΔE[A] > 0 ∧ S4 > 0. Robust: trim20/LOO
positive, ≥2 domain gains. Strong: ΔH ≥ 6 → authorize 700.
Plateau: |ΔH| ≤ 2. Fail: ΔH ≤ −3 → revisit architecture.

## Status

- Capture marathon: running (2/196 at arm time)
- Everything downstream: written, syntax-checked, key paths
  CPU-verified (model-stack wiring passes on a real new
  capture)
- Expected completion: ~17–18 h captures + ~1.5 h pipeline
  (teacher 16 min, HIST210 ~55 min, WS210 ~10 min, evals
  ~10 min)
- WS-1 + A2 + ps remains operational baseline; HIST-1
  quality comparator; capability gate open; engineering
  92–93%

## Next session pickup

```text
/tmp/scale_capture.log        capture progress
/tmp/scale_orchestrator.log   pipeline log
artifacts/d3_scale210/        all outputs when complete:
  baseline_{ws84,hist84}_scaledev70.json
  ws210_scaledev70.json  hist210_scaledev70.json
  {ws,hist}210_legacy_dev21.json
  ledger_ws84_vs_ws210.json   <- gates verdict
```
Then: write the SCALE210 session report from the ledger
verdict; per §42 repeat the DTR-0 candidate census on WS1_210.

## Artifacts (this session)

```text
artifacts/d3_scale/prompts_scale.json    corpus manifest
mosaicvm/bench/d3_scale_author.py        authoring + guards
mosaicvm/bench/d3_scale_teacher.py       gold replay + audit
mosaicvm/bench/d3_scale_train.py         HIST210/WS210 trainer
mosaicvm/bench/d3_scale_eval.py          scale eval harness
mosaicvm/bench/d3_scale_ledger.py        paired gates
mosaicvm/bench/d3_scale_orchestrate.sh   autonomous pipeline
```
