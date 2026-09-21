# Session Report — Hard-Negative Ranking Race: CLOSED
# (Monotone Regression; HIST-1 Remains DEV Comparator)

**Date:** 2026-09-13
**Roadmap:** HIST Attribution Complete — Ranking Is NEXT (§12–§28)

## Contract compliance (§12)

Only the training objective changed. Frozen: HIST-1 inputs
(h40 cur + prev aux), DDHistNet H1024/D3/aux256, corpus, train/
DEV partitions, AdamW lr 1e-3 wd 1e-4, cosine 600 steps,
CHUNK 512, clip 1.0, seed 104729, stored-teacher margin +
KL terms (µ=1, λ_m=1, λ_kl=0.5), evaluator, statistics harness,
A2, ps, splice semantics. Added:
L_rank = max(0, m + ℓ_c* − ℓ_y), m=1.0 frozen, c* = the
model's CURRENT hardest wrong token via full-vocab
target-masked argmax (dynamic mining, §20). Full-vocab CE
retained throughout.

## Arms derived from the training-only smoke (§17–§18)

Fresh-init calibration on the 2688 training rows (no DEV
involvement): grad-norm CE 89.04, rank 108.15 →
LOW λ_r=0.2058 (0.25×CE), MED 0.4116 (0.5×CE),
HIGH 0.8233 (1.0×CE). Active hinge at init: 1.000.

## Results (standalone + frozen-ps splice, DEV 21)

| arm | λ_r | S1 | Top4 | mean | trim20 | LOO | splice S1 | C2 |
|---|---|---|---|---|---|---|---|---|
| HIST-1 | 0 | 10/21 | 15/21 | 2.809 | 1.769 | 2.15 | 10/21 | .800 |
| LOW | .2058 | 9/21 | 16/21 | 2.762 | 1.692 | 2.10 | 9/21 | .889 |
| MED | .4116 | 9/21 | 14/21 | 2.762 | 1.692 | 2.10 | 9/21 | .889 |
| HIGH | .8233 | 8/21 | 14/21 | 2.619 | 1.538 | 1.95 | 8/21 | .875 |

S4 = 0.143, deepest = 15 on all arms (depth retained).
Weak domains (agent/coding/longform/retrieval) identical on
every arm: 1.0/1.0/1.333/1.667 — no broad movement.
Bootstrap CI95 all overlap the baseline.

## Classification

$$
\boxed{S_1: 10\rightarrow9\rightarrow9\rightarrow8\text{ as }\lambda_r\uparrow}
$$

- Primary gate S1 ≥ 11/21: **FAIL on all three arms** (§22)
- Robust splice stats: no gain on any arm (mean, trim20, LOO
  all ≤ baseline) (§26)
- Monotone degradation with ranking strength

Per §28:

$$
\boxed{\text{hard-negative full-vocab ranking}=\text{CLOSED under HIST-1}}
$$

No fine-grained λ tuning.

## Interpretation (measured, telemetry pass §21/§25)

Read-only telemetry over all four checkpoints
(`d3_rank_telemetry.py`, artifacts/d3_dd1_rank_telemetry.json):

| arm | mean rank | mean NLL | mean margin | S1 | VRAM |
|---|---|---|---|---|---|
| HIST-1 | 1838.8 | 4.417 | −3.130 | 10/21 | 2128.9 MB |
| LOW | 1289.0 | 4.360 | −3.153 | 9/21 | 2128.9 MB |
| MED | 238.0 | 3.932 | −2.674 | 9/21 | 2128.9 MB |
| HIGH | 777.0 | 4.583 | −3.372 | 8/21 | 2128.9 MB |

DD wall: ~9.4 ms/first-position min on every arm (identical
architecture; HIST-1's 13.5 ms mean is a single 88.7 ms
lazy-init outlier).

1. **The hinge improves BULK ranking but not boundary
   decisions.** MED cuts mean target rank 1839→238 and NLL
   4.42→3.93, yet S1 falls. Calibration mass moves to the
   distribution's bulk, away from the razor-thin boundary
   cases that produce Top-1 hits.
2. **The S1 losses are exactly json_14 (all arms; margin
   +0.039→−4.12/−1.98/−4.86, rank 1→4/2/5) plus json_13 at
   HIGH only (+0.82→−1.10).** Notably json_13's margin
   STRENGTHENS at LOW/MED (+1.70/+2.08) — the hinge is not
   uniformly destructive; it specifically destroys the
   narrowest boundary case.
3. **The conversion pool moves non-uniformly but does not
   convert.** agent_15 improves (−7.96→−6.06 LOW/−5.80 HIGH);
   coding_15 improves strongly at MED (−16.91→−11.61) but
   reverses badly at HIGH (−21.44); longform_14 WORSENS
   (−1.64→−2.12/−2.82/−2.90); retrieval_13 moves only
   slightly (−2.66→−2.47/−2.52/−2.57). No prompt crosses
   Top-1 on any arm.
4. **C2 "improvement" is a denominator artifact.** S2 count
   is identical (8/21) on HIST-1, LOW, MED; C2 rose .800→.889
   only because S1 fell 10→9. HIGH has S2=7 (genuinely
   lower). Absolute continuation depth is unchanged on
   LOW/MED — no continuation-calibration gain exists.
5. Train boundary reaches S1=1.000 (memorized; hinge inactive
   by step 100) — the rank gradient arrives only early, where
   the dynamically mined hardest-wrong token changes every
   step.

The conversion pool moves non-uniformly but does not
convert. Some targets improve substantially under ranking,
others change little, and some worsen. MED improves aggregate
target rank and NLL but does not improve Top-1 and loses the
razor-thin json_14 hit. Therefore this hard-negative
objective is closed under HIST-1. The experiment does not
distinguish a fundamental information deficit from other
representation/objective limitations; low-rank direct logits
is promoted because it is the precommitted next
representation family, not because ranking proved
information insufficiency.

## Consequences

- **HIST-1 (λ_r=0) remains the frozen DEV comparator**
  (S1 10/21, mean 2.809, trim20 1.769).
- Per §30, the next family is the **low-rank direct-logit
  latent**: root features → q ∈ R^r → E q (r ∈ {128, 256,
  512}), removing the requirement to produce a 4096-dim
  state compatible with the source head.
- HIST-3 remains deprioritized; BranchPack closed;
  DD-2 deferred; TCS-FD deferred.
- Capability gate E[A] ≥ 4 remains OPEN. Engineering stays
  92–93%.

## Artifacts

```text
artifacts/d3_dd1_rank/rank_LOW/dd1_hist.pt    LOW checkpoint
artifacts/d3_dd1_rank/rank_MED/dd1_hist.pt    MED checkpoint
artifacts/d3_dd1_rank/rank_HIGH/dd1_hist.pt   HIGH checkpoint
artifacts/d3_dd1_rank/rank_{LOW,MED,HIGH}_eval.json
artifacts/d3_dd1_rank_smoke.json              calibration pass
artifacts/d3_dd1_rank_telemetry.json          S21/S25 telemetry
mosaicvm/bench/d3_dd1_rank.py                 trainer + smoke
mosaicvm/bench/d3_rank_telemetry.py           read-only telemetry
```

All three checkpoints load in the unmodified
`d3_dd1_hist_eval.py` (format-compatible; only objective
differed).
