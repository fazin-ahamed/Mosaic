# Session Report: CAP-DEV70 — NEAR-MISS (E[A]=3.76, gate 4.0)

**Date:** 2026-09-20
**Status:** COMPLETE. Frozen candidate evaluated on fresh holdout.
**Verdict:** NEAR-MISS. B4 E[A] = 3.757 on CAP-DEV70 (fresh). Below the
capability gate 4.0 but a strong, honest fresh-generalization result.
**BLIND stays SEALED.**

---

## 1. What this was

The first honest exam of the frozen candidate `HIST1001 + ps1001 + d3 B4`
on 70 genuinely fresh prompts (IDs {dom}_300-309), disjoint from every
train corpus, DEV21, SCALE-DEV70/2/3, G1/G2. Manifest sha 97a654f3…,
overlap-audited empty. Captured via SafeCapture v2 (CUDA recycled /8),
deep-validated 70/70. No SCALE-DEV3 squeezing before the exam.

## 2. Result

| B | E[A] | dEA | S1 | S3 | S4 | nodes |
|---|---|---|---|---|---|---|
| 1 greedy | 3.100 | - | .543 | .314 | .271 | 3 |
| 2 | 3.500 | +.400 | .543 | .357 | .314 | 7 |
| 4 | **3.757** | +.657 | .543 | .414 | .371 | 21 |
| 8 | 3.757 | +.657 | .543 | .414 | .371 | 73 |

B4 canonical: mean 3.757, median 3.0, trim20 2.810, LOO [3.580, 3.797],
bootstrap 95% CI **[3.014, 4.571]**, S4 .371, S8 .114, deepest 15,
accepted 38/70. B4==B8 (branching saturated at B4 on this set too).

## 3. Development vs fresh (generalization gap)

| set | B4 E[A] | greedy E[A] | S1 |
|---|---|---|---|
| SCALE-DEV3 (burned) | 4.014 | 3.300 | .643 |
| CAP-DEV70 (fresh) | 3.757 | 3.100 | .543 |

Drop of 0.257 E[A] and 0.10 S1 on fresh prompts. Modest — the candidate
**generalizes**, it is not a SCALE-DEV3 artifact. But it lands just under 4.0.
The gate-4.0 point on SCALE-DEV3 was partly development-set optimism.

## 4. Per-domain E[A] at B4 (the real story)

```
retrieval  6.30   convo 5.60   json 4.90   math 3.20
coding     2.70   agent 2.10   longform 1.50
```

Bimodal. Retrieval/convo/json already far exceed 4. The gate miss is carried
entirely by **longform (1.50), agent (2.10), coding (2.70)** — reasoning/
generation domains where continuation drifts fastest. Mean 3.757 is 3 strong
domains dragged down by 3 weak ones.

## 5. Verdict classification

```text
CAP-DEV70 primary B4 E[A]            3.757
capability gate (>=4)                NOT MET (near-miss)
fresh generalization                 CONFIRMED (only -0.26 vs dev)
bootstrap CI                         [3.014, 4.571]  (4.0 inside CI)
robust breadth                       PARTIAL (3 domains >>4, 3 domains <3)
BLIND                                SEALED (unchanged)
CAP-DEV70                            now BURNED (informed a decision)
```

Per roadmap section 19 (3.5 <= E[A] < 4): near-miss -> predeclared
decomposition, smallest scoped improvement. NOT random architecture search,
NOT SCALE-DEV3 re-tuning, NOT reusing CAP-DEV70 for tuning.

## 6. Decomposition — where the missing 0.24 lives

The gate miss is domain-concentrated, not uniform:
- **longform 1.50 / agent 2.10 / coding 2.70** — continuation state-drift is
  worst in long reasoning/generation. ps1001 helped globally (+1.1 greedy) but
  these domains remain drift-limited.
- retrieval/convo/json are already solved; more work there is wasted.

The scoped lever is therefore **domain-targeted continuation**, not more
breadth and not root work (root S1 .543 is not the dominant limiter here —
these domains fail at depth, not root).

## 7. Options (user decision — all involve training or new capture)

1. **Prefix Survival Loss on ps** (roadmap section 23): retrain ps1001 with
   PSL weighting so late-depth tokens in drift-prone domains get survival-
   weighted gradient. No new capture; reuses the 1001 union. Cheapest.
2. **Domain-reweighted continuation**: upweight longform/agent/coding rows in
   the ps loss. No new capture. Risk: overfit 3 domains.
3. **Targeted Hydra H2/H3/H4** (roadmap section 24/30): direct future-token
   heads for depths 2-4. More work; addresses drift structurally.
4. **Accept 3.76 as the honest number**, measure d3-B4 production cost (H2
   expert-union), and treat P3 economics as the parallel track while a scoped
   continuation fix runs.

## 8. What did NOT happen (guardrails held)

- BLIND untouched.
- No SCALE-DEV3 re-tuning to inflate the dev number before the exam.
- Candidate frozen exactly as developed; the exam measured it as-is.
- CAP-DEV70 is now burned; a further fresh authority is needed before a
  future promotion attempt.

## 9. Artifacts

- `capdev70_candidate.json` — full probe result
- `prompts_capdev70.json` (sha 97a654f3…) + `overlap_audit.json`
- `d3_capdev70_author.py`, `d3_capdev70_supervisor.sh`, `d3_capdev70_finish.sh`
- `d3_capdev70_data/` — 70 validated captures

## 10. Bottom line

The candidate is real and generalizes, but lands at **3.76, not 4.0**, on a
clean holdout. 4.0 sits inside the bootstrap CI, and 3 of 7 domains already
clear it comfortably — the miss is a focused longform/agent/coding depth-drift
problem, not a broad failure. The cheapest honest next step is a domain-aware
continuation fix (PSL or reweighting), then a NEW fresh authority before any
BLIND attempt.
