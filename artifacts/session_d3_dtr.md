# Session Report — DTR-0 Census + DTR-R1 Reranker: REGRESSION
# (Reranker CLOSED on This Corpus; WS-1 Stands)

**Date:** 2026-09-13
**Roadmap:** Constrained LRDL CLOSED — Decision-Native Token
Ranking Is NEXT (§9–§24, §59 steps 3–8)

## DTR-0: candidate coverage census (zero training)

Exact full-vocab target ranks, both frozen models, DEV21:

| M | HIST-1 | WS-1 | union |
|---|---|---|---|
| 1 | .476 | .476 | .476 |
| 2 | .571 | .667 | .667 |
| 4 | .714 | .762 | .809 |
| 8 | .809 | **.809** | .809 |
| 16 | .809 | **.809** | .809 |
| 64 | .857 | .809 | .857 |
| 256 | .905 | .809 | .905 |
| 512 | .905 | .857 | 1.000 |

- **7 rerankable misses, ALL at WS-1 rank 2–5**: coding_14
  (r2), longform_14 (r2), longform_15 (r2), math_13 (r2),
  agent_15 (r3), retrieval_13 (r3), retrieval_14 (r5)
- **4 unreachable**: coding_13 (r509), agent_14 (r1547),
  agent_13 (r2023), coding_15 (r17745) — candidate
  generation (DTR-G) territory, not rerankable
- Branch A fired: reranker ceiling 17/21 at M≤16

## DTR-R1: shared nonlinear reranker — REGRESSION 8/21

Architecture per §18–19: r=P_r(q) [128], e=P_e(E_t) [128],
features [r, e, r⊙e, |r−e|, ℓ_base], MLP 513→256→1
zero-init, s_t = ℓ_base + Δ_t. ~200k shared params, zero
per-token rows. Frozen trunk + Q. Candidate CE + 0.5
candidate-KL. M=16, true token force-included in TRAINING
only.

- Step-0 equivalence: **PASS** (bitwise; S1 10/21)
- Training retrieval: **2688/2688 natural** — every training
  target sits in WS-1's top-16
- Train candidate accuracy: **1.000 by step 100** (memorized)
- **End-to-end DEV: S1 = 8/21** (recall@16 0.810;
  rerank-accuracy-given-retrieved 8/17 = 0.471)

### The overturn ledger (5 rerank decisions moved top-1)

| prompt | base rank | outcome |
|---|---|---|
| longform_14 | 2 | **CONVERTED** |
| json_15 | 1 | **LOST** (correct base decision overturned) |
| longform_13 | 1 | **LOST** (correct base overturned) |
| math_14 | 1 | **LOST** (correct base overturned) |
| agent_13 | 2023 | moved, still miss |

The other 6 rerankable near-misses (coding_14 r2,
longform_15 r2, math_13 r2, agent_15 r3, retrieval_13 r3,
retrieval_14 r5) were NOT moved. The reranker is worse than
the base ranking it replaced among retrieved targets
(0.471 < 10/17 = 0.588). Splice skipped (S1 < 10).

$$
\boxed{\text{DTR-R1}=\text{REGRESSION}}
$$

## Accumulated evidence — the corpus, not the geometry

Eleven trained scorer/head variants have now been evaluated
on DEV21 under the same frozen trunk inputs:

```text
DD-1, HIST-0M, HIST-1, HIST-2, rank-LOW/MED/HIGH,
LRDL-0 r256, LRDL-0 r512, LRDL-1A, LRDL-1B, DTR-R1
```

Every one reaches train accuracy 1.0 (memorization of 2,688
transitions) and lands at DEV S1 = 8–10/21. Every variant
that adds trainable degrees of freedom beyond the base
recipe disturbs razor-thin boundary hits. Output-geometry
interventions are exhausted (hinge, basis, rank, row-scale,
residual, nonlinear reranker); input information is frozen
at the HIST-1 feature set; candidate coverage is NOT the
constraint (17/21 within rank 5).

The consistent failure mode — instant train memorization
plus fragile DEV boundaries — identifies the 2,688-transition
corpus as the binding constraint on any additional trained
scorer.

## Status and decision point

- **DTR-R1 = CLOSED (regression)**. Per §59.14 the
  precommitted fallback is a fully independent direct token
  classifier — but the 11-variant record predicts the same
  memorization outcome on this corpus.
- **WS-1 + A2 + ps remains the operational baseline**
  (10/21, E[A] 2.809, 3.3 ms root). HIST-1 remains quality
  comparator.
- The roadmap now faces a measured deadlock: capability
  (E[A] ≥ 4) appears unreachable without more training data,
  and §40 gates data scaling behind capability + BLIND. The
  architecture question the gate was protecting is now
  settled (WS-1 is a validated, integrity-certified decision
  path). Resolving the deadlock — scaling the corpus
  (84 → 210 → 700) versus one more classifier family — is a
  roadmap-level decision.
- Capability gate OPEN. Engineering 92–93%.

## Artifacts

```text
artifacts/d3_dtr0/census.json            coverage census
artifacts/d3_dtr_r1/dtr_r1.pt            reranker
artifacts/d3_dtr_r1/dtr_r1_eval.json     end-to-end eval
mosaicvm/bench/d3_dtr0_census.py
mosaicvm/bench/d3_dtr_r1.py
```
