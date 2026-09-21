# Session Report — SCALE700 Setup: Corpus Restructured,
# Phase-Aware Pipeline Armed (~58h Autonomous Run)

**Date:** 2026-09-14
**Roadmap:** SCALE210 PASSED — Fresh SCALE700 Generalization
Phase Is NEXT (§9–§23, §48 steps 3–13)

## Completed

### Step 3 — WS1_210 root latency remeasured
Same-session SCALE-DEV70 measurements: WS1_210 4.57 ms vs
HIST1_210 11.08 ms per root decision — the compression track
retains its ~2.4× root advantage at 210-scale.

### Steps 4–5, 7 — 560-prompt corpus authored and frozen
70 SCALE-DEV2 ({dom}_047..056, 10/domain) + 490 SCALE700
train ({dom}_057..126, 70/domain).

**Defect caught and fixed before freezing** (advisor review
of the first draft): SCALE-DEV2 was template-degenerate in
4/7 domains (json: one fieldset × 10 entities; retrieval:
one angle × 10 docs; agent: ONE task × 10 objects; coding:
dev2 tasks reappearing in train with suffix variants), and
the val-first manifest sort silently no-oped (2-digit
endswith vs 3-digit categories). Restructured generators now
guarantee per-domain task-disjointness across the dev2↔train
boundary, verified structurally:

```text
coding:    task-phrase overlap    0
json:      entity overlap         0
retrieval: document overlap       0
agent:     task-family overlap    0
math:      dev2 spans all 10 families (round-robin)
longform/convo: 80 distinct topics each
global:    560/560 unique texts
```

Manifest (val-block first):
`artifacts/d3_scale700/prompts_scale700.json`
**sha256 62ecaf2bc6b8c07010ba4c66f17754948b93f1c916885e98d6a70360312be687**
All texts disjoint from the 322 prior prompts and G1/G2
protected sets.

### Steps 6–13 — pipeline armed (fail-closed, phase-aware)
- **Engine** (`d3_scale700_engine.sh`): phase A = capture 70
  dev2 prompts; PHASE GATE = waits for the orchestrator's
  baseline flag before phase B (490 train captures);
  auto-relaunch on death/45-min stall; resumable; exits at
  560.
- **Orchestrator** (`d3_scale700_orchestrate.sh`,
  `set -euo pipefail`): wait 70 dev2 captures → deep-validate
  → baseline HIST1_210 + WS1_210 on SCALE-DEV2 (§10:
  standalone+splice+census, frozen before any training) →
  flag → wait 560 → validate all (prompt hashes, finiteness,
  horizon, teacher class, snapshot revision) → 490 gold
  teacher replays + hash audit → HIST1_700 (primary, exact
  frozen recipe, 22,400 rows) → WS1_700 (secondary, frozen
  trunk + unchanged Q recipe) → SCALE-DEV2 evals + censuses →
  SCALE-DEV70 + legacy DEV21 continuity → paired ledgers
  (hist210→hist700 primary; ws210→ws700 secondary) with the
  predeclared gates (ΔH≥4 / robust / ΔH≥6 strong / |ΔH|≤2
  plateau / ΔH≤−3 fail; E[A]≥4 → BLIND).
- Model-stack wiring CPU-verified through the new eval paths.
- **Phase-gate deadlock caught and fixed before it fired**
  (advisor review): `load_captures` required BOTH train+val
  captures in the directory, but at the phase gate only the
  70 val captures exist — the first baseline would have
  crashed, `set -euo pipefail` would have killed the
  orchestrator before setting the flag, and the engine
  would idle forever. Fix: `require_both` kwarg (default
  True preserves the invariant for all other callers);
  eval + census pass False. Verified: old path still
  raises on val-only; new path loads and split-filters
  val-only captures correctly.
- **Engine livelock caught and fixed during phase A**: the
  stall detector treated "zero capture files" as a stall
  (newest-age sentinel 999999), killing the healthy recorder
  during its ~8-min startup on every 300s cycle. Fixed with
  a startup grace (recorder process age < 3600s with zero
  files = healthy) plus partial-file hardening: any
  stall-relaunch verifies existing .pt files load and
  removes pkilled-mid-save partials (which the recorder
  would otherwise skip forever). First capture verified
  fully valid after the fix (math_047).

### Costs (measured)
~340 s/capture × 560 ≈ 53 h + pipeline ≈ 5 h ≈ **58 h total**.
Baselines run ~6.6 h from start (phase gate), everything
else completes ~2.5 days from launch.

## Gates predeclared (§19–§23)
Unchanged from SCALE210: ΔH≥4 ∧ ΔE[A]>0 ∧ S4>0 mechanism;
robust extras; ΔH≥6 strong (→1000+ if needed); |ΔH|≤2
plateau (STOP automatic scaling); ΔH≤−3 fail; E[A]≥4 →
freeze + one-shot BLIND-ADJ.

## Artifacts

```text
artifacts/d3_scale700/prompts_scale700.json   560-prompt manifest
mosaicvm/bench/d3_scale700_author.py          guarded generator
mosaicvm/bench/d3_scale700_teacher.py         gold replay + audit
mosaicvm/bench/d3_scale700_train.py           HIST700/WS700 trainer
mosaicvm/bench/d3_scale700_census.py          census harness
mosaicvm/bench/d3_scale700_engine.sh          phase-aware capture engine
mosaicvm/bench/d3_scale700_orchestrate.sh     fail-closed pipeline
mosaicvm/bench/d3_scale_eval.py               + hist700/ws700 registry
```

## Next pickup

```text
/tmp/scale700_engine.log        capture phases
/tmp/scale700_orchestrator.log  pipeline log
artifacts/d3_scale700_train/    outputs:
  baseline_{hist,ws}210_scaledev2.json (+ censuses)
  hist700.pt  ws700.pt
  {hist,ws}700_scaledev2.json (+ censuses)
  hist700_scaledev70.json  {hist,ws}700_legacy_dev21.json
  ledger_hist210_vs_hist700.json   <- PRIMARY verdict
  ledger_ws210_vs_ws700.json       <- secondary
```
