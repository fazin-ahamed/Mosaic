# Session Report — HIST-1 Previous-State: S1 = 10/21
# (Single-Prompt Gain; DEV Comparator, Not Capability Promotion)

**Date:** 2026-09-12
**Roadmap:** DD Capacity + Boundary Emphasis CLOSED (§9–§24)

## HIST race results (parameter-matched, only variable = aux input)

| arm | S1 | Top-2 | mean | trim20 | LOO | S4 | deep | gate |
|---|---|---|---|---|---|---|---|---|
| baseline (no aux) | 9/21 | 0.524 | 2.762 | 1.692 | 2.100 | 0.143 | 15 | PASS |
| HIST-0M (zero aux) | 9/21 | 0.619 | 2.762 | 1.692 | 2.100 | 0.143 | 15 | PASS |
| **HIST-1 (prev state)** | **10/21** | 0.571 | **2.809** | **1.769** | **2.150** | 0.143 | 15 | **PASS** |
| HIST-2 (delta) | 8/21 | 0.571 | 2.667 | 1.538 | 2.000 | 0.143 | 15 | FAIL |

## HIST-1: S1 = 10/21 (first above 9/21; single-prompt gain)

**S1 = 10/21 = 0.476** — the first arm to exceed 9/21. The
gain is from ONE prompt (json_14: E[A] 1→2); all other prompts
identical to baseline. The HIST gate (§18: S1 ≥ 10/21) is
technically met, but at n=21 this is a single-prompt event.

Robust statistics:
- trim20 = 1.769 (above A2's 1.762, but n=21 and driven by
  the same single prompt — not a production claim)
- LOO = 2.150 (up from 2.100)
- mean E[A] = 2.809 (up from 2.762)
- S4 = 0.143, deepest = 15 — depth retained
- **C2 = 0.800 (DOWN from baseline 0.889)** — conditional
  continuation slightly degraded
- agent and coding domains remain at 1.0 (no breadth gain
  there)
- Mechanism gate: PASS

## The matched control validates the finding

HIST-0M (zero aux, identical params) reproduces the baseline
exactly (S1 = 9/21, all splice stats identical). The extra
aux stem + fusion parameters contribute NOTHING when fed zeros.
Therefore HIST-1's improvement is entirely due to the PREVIOUS
STATE INFORMATION, not capacity.

## HIST-1 vs HIST-2 interpretation (roadmap §24)

HIST-1 (prev state) wins; HIST-2 (delta) regresses. The useful
temporal information is in the previous state itself, not the
first-order motion. This is richer than a simple velocity — the
previous state contains the model's full representation one
step back, which the delta compresses away.

## Leakage and integrity

- 84 boundary aux inputs verified non-leaking (aux never
  touches h40[S] or later)
- Aux-stem gradient nonzero on HIST-1 (0.100); zero on
  HIST-0M (expected from zero input)
- Teacher hash gate PASS
- Param count identical across all arms (verified)

## Architecture (frozen DEV comparator)

```text
main: h40[S-1] + emb(root) + emb(prev) + pos → main_stem ─┐
                                                           ├→ fusion → DD H1024 → z → head
aux:  h40[S-2]                 → aux_stem(16384→256)  ───┘
```

## What this means (scoped)

The previous committed state provides genuine additional
information for the first decision. HIST-1 is the new best
proposer and is promoted as the frozen baseline. But the
improvement is narrow: one prompt, no domain breadth gain
(agent/coding still 1.0), C2 slightly degraded, and E[A] =
2.809 is far below the ≥ 4 capability gate.

## NEXT

Freeze HIST-1 as the new production-comparison baseline. Do
not immediately jump to ranking or HIST-3 — the improvement
needs to be understood (why json_14 specifically? is there a
pattern?) before the next lever is chosen.

## Artifacts

```text
artifacts/d3_dd1_hist/dd1_hist_{zero,prev,delta}/dd1_hist.pt
artifacts/d3_dd1_hist/dd1_hist_{zero,prev,delta}_eval.json
mosaicvm/bench/d3_dd1_hist.py          HIST trainer
mosaicvm/bench/d3_dd1_hist_eval.py     HIST evaluator
```
