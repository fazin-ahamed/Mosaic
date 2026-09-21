# Session Report — SCALE210: PASS (ΔH=+5 on Fresh Holdout;
# SCALE700 Authorized; Corpus Hypothesis Confirmed)

**Date:** 2026-09-14
**Roadmap:** DTR-R1 CLOSED — Controlled Corpus Scaling Is NEXT
(§10–§42, §64)

## Predeclared verdict (S27–§30)

Paired ledger WS1_84 → WS1_210 on SCALE-DEV70 (fresh 70-prompt
authority, disjoint from all training and the blind reserve):

$$
G=9,\quad L=4,\quad \Delta H=+5
$$

$$
S_1: 32/70 \rightarrow 37/70\ (0.457\rightarrow0.529)
$$

- **Gates: mechanism S28 PASS** (ΔH≥4, ΔE[A]=+0.114>0,
  S4=0.0143>0), **robust S29 PASS** (Δtrim20=+0.119,
  ΔLOO=+0.087, net domain gains 2), strong S30 (ΔH≥6) not
  reached, no plateau, no fail
- **Verdict: PASS — SCALE700 AUTHORIZED** (§34: 210→700,
  7×100, requires fresh SCALE-DEV2 first per §35)

Gains: agent ×5 (agent_37/39/41/42/44), coding_42, json_43,
longform_38, math_43. Losses: json_37/39, longform_41,
math_40. **The agent domain — 0.00 S1 on every previous
model — moved to 0.50 under WS1_210.**

## Full numbers (SCALE-DEV70)

| model | splice S1 | mean E[A] | trim20 | LOO | Top4 | NLL | mean rank |
|---|---|---|---|---|---|---|---|
| WS1_84 (baseline) | 32/70 | 1.857 | 1.667 | 1.826 | .671 | 3.47 | 617 |
| HIST1_84 (baseline) | 29/70 | 1.757 | 1.548 | 1.725 | .700 | 3.79 | 759 |
| WS1_210 | 37/70 | 1.971 | 1.786 | 1.913 | .800 | 2.17 | 111 |
| **HIST1_210** | **41/70** | **2.114** | **1.952** | **2.058** | **.857** | **1.97** | **71** |

Training: 6,720 transitions (210 prompts), identical frozen
recipe, only dataset size changed. Train argmax reached 1.000
(step 200) as before — the gain is pure generalization.

## Observations (scoped, not acted on)

1. **HIST1_210 (full head) outperforms WS1_210 (rank-256 Q)
   at 210-scale: 41/70 vs 37/70, ΔH=+12 vs its own 84
   baseline.** At 84-scale the ordering was reversed (32 vs
   29). The predeclared ledger pairing was WS-vs-WS (PASS);
   the HIST-210 result is the stronger measured outcome and
   the new best proposer. Possible causes (not diagnosed):
   the frozen 600-step Q recipe underfits at 2.5× data, or
   rank-256 becomes binding once the trunk improves. No
   recipe change made (§18: one frozen test).
2. **Legacy DEV21 regression (continuity eval only):** WS1_84
   10/21 → WS1_210 7/21 (HIST: → 8/21), with depth preserved
   (deepest 15, S4 0.143). After 20+ architecture decisions
   iterated against DEV21, its 10/21 partially reflected
   selection; SCALE-DEV70 is the authority and shows +5/70.
   Both numbers reported; no DEV-based action.
3. **C2 drop (.75→.65) is a denominator artifact again**: S2
   count is 24 on both baseline and scaled; C2 moved only
   because S1 rose 32→37.
4. **§42 census follow-up:** coverage scaled with quality —
   HIST1_210 recall@16 = .943 (WS1_210 .871 vs WS-84's .810
   on the easier DEV21); only 4/70 prompts remain unreachable
   (coding_39/43, json_40/45, ranks 78–3139); 25 rerankable
   misses at rank 2–64.

## Consequences

- **SCALE700 authorized** (§30/§34): 7×100=700 training
  prompts, 22,400 transitions. Per §35, a FRESH SCALE-DEV2
  (70 prompts, fully disjoint) must be created before SCALE700
  training — SCALE-DEV70 is now a development-decisions set.
- The corpus hypothesis is CONFIRMED at 84→210: the
  8–10/21 plateau across 11 scorer variants was substantially
  data-limited, as hypothesized (not proven as the only
  factor — the 4 remaining unreachable prompts and the
  WS/HIST-210 gap show headroom in both data and
  representation).
- Frozen DEV21 = legacy; SCALE-DEV70 = current authority
  (until SCALE-DEV2 exists).
- Capability gate E[A]≥4: best now 2.114 (HIST1_210 on
  SCALE-DEV70) — still OPEN. Engineering 92–93% (no
  capability gate crossed; the scale pass is a development
  result, not a production claim).

## Artifacts

```text
artifacts/d3_scale210/ledger_ws84_vs_ws210.json   verdict
artifacts/d3_scale210/baseline_{ws84,hist84}_scaledev70.json
artifacts/d3_scale210/{ws,hist}210_scaledev70.json
artifacts/d3_scale210/{ws,hist}210_legacy_dev21.json
artifacts/d3_scale210/census_ws210_scaledev70.json
artifacts/d3_scale210/hist210.pt  ws210.pt
artifacts/d3_scale_teacher/audit.json             126 records
artifacts/d3_scale_data/                          196 captures
```
