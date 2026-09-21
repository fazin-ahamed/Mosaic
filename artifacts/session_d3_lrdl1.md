# Session Report — LRDL-1 Constrained Adapters: PLATEAU ×2
# (Family CLOSED; WS-1 Remains Fast Baseline)

**Date:** 2026-09-13
**Roadmap:** Warm-Start Deconfound PASSED — LRDL-1
Constrained Vocabulary Geometry Is NEXT (§11–§32, §60)

## LRDL-1A: row-scale adapter — PLATEAU at 10/21

E' = diag(s)·E_r, s = exp(0.25·tanh(a)), a∈R^V (129,280
params), zero-init. λ_s = 71.880 from a training-only
gradient-ratio smoke (0.1× rule; scale-grad 4.5e-5 vs CE-grad
0.032). Frozen: trunk, Q256, E_r, corpus, split, A2, ps,
evaluator, recipe (AdamW 1e-3, cosine 600, clip 1.0, seed).

- Step-0 equivalence: **bitwise PASS** (s=1.0 exact multiply;
  logits torch.equal WS-1; S1 10/21; margins to 4.9e-5)
- Final: S1 = 10/21, **identical hit set to WS-1**, splice
  identical (2.809/10/21/S4 .143/deep 15)
- Margins barely move (max |Δ| 0.61, json_13 +4.48→+5.08);
  **zero rank changes on all 21 prompts**
- Scale usage tiny: max |log s| 0.067 of the 0.25 bound; only
  4,663/129,280 rows active; NLL slightly worse (4.849→4.871)
- **Diagnosis: row MAGNITUDE is not the limiting factor —
  directional geometry is** (near-misses: longform_14 −0.78
  r2, math_13 −1.13 r2, coding_14 −3.24 r2, retrieval_13
  −2.58 r3)

## LRDL-1B: rank-4 residual adapter — PLATEAU at 10/21 (swap)

E' = E_r + AB, A∈R^{V×4} (zero-init), B∈R^{4×256}
(N(0,0.02²)); ΔE=0 exactly at step 0. λ_d = 16.680 (same
smoke rule). Norm cap ‖ΔE‖/‖E‖ ≤ 0.10 by projection;
**actual final ratio 0.0117** (12% of cap). Trainable ~518k.

- Step-0 equivalence: **bitwise PASS** (10/21, 4.9e-5)
- Final: S1 = 10/21 via a SWAP, not a gain:
  - **longform_15 −0.385 → +0.374 (converted, healthy)**
  - **longform_13 +0.004 → −0.152 (razor-thin hit LOST)**
  - json_13 strengthened +4.48 → +5.58
- Top4 15/21 (down from WS-1's 16); Top8 17/21; NLL 4.824
  (best of the three heads)
- Splice identical: 2.809 / 10/21 / S4 .143 / deep 15

## Family closure per §32

Both precommitted conditions are met:

```text
row-scale adapter     PLATEAU 10/21 (no conversion)
rank-4 residual       PLATEAU 10/21 (swap, no net gain)
robust splice         no improvement (identical 2.809)
```

$$
\boxed{\text{constrained LRDL family}=\text{CLOSED}}
$$

§30 joint Q+adapter refinement is NOT authorized: LRDL-1B did
not preserve all ten baseline hits and no arm reached 11/21.
No adapter-rank sweeps, no finer bounds, no free vocabulary
matrix (§31 remains closed: ~33M params vs 2,688 rows).

## Standing interpretation

The adapter arms confirm the WS finding's mechanism: the
frozen-basis decision geometry is essentially complete after
Q — the remaining near-misses are NOT fixable by small
vocabulary-geometry perturbations (magnitude or rank-4
direction). The binding constraint is upstream: the
information the trunk/Q path extracts from the root features
— the same constraint every HIST/DD variant hit (S1 8–10/21
across nine trained variants).

## Frozen state

- **Quality comparator: HIST-1 + A2 + ps** (10/21, E[A]
  2.809) — unchanged
- **Fast-equivalent baseline: WS-1** (10/21 identical hit
  set, splice identical, 3.3 ms root decision) — unchanged,
  remains the operationally preferred implementation
- LRDL-1A/1B checkpoints retained as closed-experiment
  records (both reproduce 10/21; neither promoted)
- Next family per §33: **direct token/rank representation**
  (pure discriminative proposer, no source-head-compatible
  state) or retrieval+rerank — new roadmap required
- Capability gate E[A] ≥ 4 OPEN. Engineering 92–93%.

## Artifacts

```text
artifacts/d3_lrdl1a/smoke.json             scale smoke
artifacts/d3_lrdl1a/lrdl1a.pt              row scales
artifacts/d3_lrdl1a/lrdl1a_eval.json       telemetry + splice
artifacts/d3_lrdl1b/smoke.json             residual smoke
artifacts/d3_lrdl1b/lrdl1b.pt              A, B factors
artifacts/d3_lrdl1b/lrdl1b_eval.json       telemetry + splice
mosaicvm/bench/d3_lrdl1a.py                row-scale harness
mosaicvm/bench/d3_lrdl1b.py                residual harness
```
