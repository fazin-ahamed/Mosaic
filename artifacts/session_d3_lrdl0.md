# Session Report — LRDL-0 Frozen-Basis Control: DEV REGRESSION
# (S1 8/21 at r256 AND r512; HIST-1 Remains DEV Comparator)

**Date:** 2026-09-13
**Roadmap:** Hard-Negative Ranking CLOSED — Low-Rank Direct
Logits NEXT (§15–§34, §59 steps 6–11)

## What was run

**LRDL-0 (§18):** rank-256 frozen source-head vocabulary basis
E_r from randomized SVD of W_eff = W_head·diag(w_norm), trunk
DDHistNet H1024/D3/aux256 with head replaced by
Linear(1024→256); identical HIST-1 inputs (current h40 +
previous h40 aux + tokens + pos); objective full-vocab CE +
λ_KL=0.5 teacher KL only (no hinge, no stored-margin); all
other recipe values frozen (AdamW 1e-3, cosine 600, CHUNK 512,
clip 1.0, seed 104729).

**r512 (exploratory):** the §22/§59 escalation to r512 is
conditioned on CLEAR UNDERFIT. That condition was NOT met —
r256 train boundary reached S1=1.000 and the free-q
expressive probe fit 100% top-1 on all 2688 rows within the
frozen r256 basis. The r512 run is therefore reported as an
exploratory rank-sensitivity diagnostic, not as the
precommitted underfit branch.

## Basis verification (§7)

| rank | energy fraction | rel. Frobenius error | top-1 agreement on HIST-1 z | free-q top-1 |
|---|---|---|---|---|
| 256 | 0.2195 | 0.883 | 0.060 | **1.000** |
| 512 | 0.3204 | 0.824 | 0.117 | — |

The source head is not low-rank (22%/32% energy at 256/512),
and HIST-1's own decision vectors project poorly (6–12% top-1
agreement) — but the free-q probe shows the basis itself is
fully expressive for these decisions. Expressivity was never
the constraint; generalization is.

## Results (DEV 21)

| arm | S1 | Top4 | NLL | mean rank | wall | VRAM |
|---|---|---|---|---|---|---|
| HIST-1 | **10/21** | 15/21 | 4.417 | 1838.8 | 9.4 ms | 2128.9 MB |
| LRDL-0 r256 | 8/21 | 13/21 | 3.958 | 293.5 | 3.1 ms | 2261.8 MB |
| LRDL-0 r512 | 8/21 | 14/21 | 3.392 | — | 3.6 ms | 2394.1 MB |

Splice (frozen A2 + ps): both ranks mean E[A]=2.619, S1=8/21,
S4=0.143, deepest 15, C2=0.875.

## Measured findings

1. **Both ranks lose EXACTLY json_13 and json_14, gain
   nothing.** The 8 r512 hits are a strict subset of HIST-1's
   10. json_13: +0.82→−0.61 (rank 1→2); json_14: +0.04→−2.06
   (rank 1→2). All stable-hit prompts carry over (convo,
   math, longform_13 — the latter IMPROVES +1.49→+3.09).
2. **The regression is rank-INSENSITIVE.** Doubling rank
   improved bulk (NLL 3.96→3.39, Top4 0.619→0.667) and changed
   the gate not at all — the same signature as the ranking
   race: bulk calibration improves, boundary decisions do not.
3. **NOT underfitting.** Train boundary S1=1.000 at both
   ranks; free-q fits 100%. The failure is generalization of
   the retrained trunk under the basis interface, not
   capacity, not basis expressivity.
4. **First-position wall is 2.6–3× faster** (3.1–3.6 ms vs
   9.4 ms; the 4096-dim pseudo-state + full-head matmul is
   gone) — but proposer speed is not the binding constraint
   and does not offset the S1 regression.

## Confound (explicit)

LRDL-0 retrains the trunk from scratch with a new head. The
measured S1 drop therefore mixes (a) the representation
interface change with (b) trunk-retraining variance. The
DD-1-family retraining band spans 8–10/21 (DD-1 9, HIST-0M 9,
rank arms 9/9/8), so 8/21 twice is at the band's floor: a real
regression (json_13, usually stable, is lost in both), but the
experiment does not isolate WHY. A warm-start variant (frozen
HIST-1 trunk, train only the q head) would separate the
confounds but is not in the precommitted set.

## Status

- **LRDL-0 frozen-basis sub-branch: REGRESSION (8/21 at both
  tested ranks). No DEV promotion.**
- **The LRDL family is NOT closed** — the §20/§21 constrained
  adaptable basis (LRDL-1) is untested — but no §59
  escalation condition fires: 11A (LRDL-0 shows gate signal)
  NO; 11B (clear underfit) NO; 11C (succeeds, cost high) NO.
- HIST-1 + A2 + ps remains the frozen DEV comparator
  (S1 10/21, E[A] 2.809).
- Open decision point (roadmap level): (i) LRDL-1 constrained
  adapter despite no LRDL-0 gate signal; (ii) warm-start
  diagnostic to deconfound; (iii) §35 fallback (direct
  token/rank classifier, retrieval+rerank). None is
  authorized by the current roadmap's conditionals.
- Capability gate E[A] ≥ 4 remains OPEN. Engineering stays
  92–93%.

## Artifacts

```text
artifacts/d3_lrdl0/basis_r256.pt            frozen basis
artifacts/d3_lrdl0/basis_r512.pt            frozen basis
artifacts/d3_lrdl0/basis_verify_r{256,512}.json
artifacts/d3_lrdl0/free_q_probe.json        expressive probe
artifacts/d3_lrdl0/lrdl0_r{256,512}/lrdl0.pt
artifacts/d3_lrdl0/lrdl0_r{256,512}_eval.json
mosaicvm/bench/d3_lrdl.py                   trainer + verify + probe
mosaicvm/bench/d3_lrdl_eval.py              S31/S32 evaluator
```
