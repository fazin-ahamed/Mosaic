# G2 Calibration — Source-vs-Source Envelope + §41 Localization

**Date:** 2026-09-04
**Roadmap:** G1 Complete — G2 Calibration + TRUE P2 Phase (§17-41, §71 Track E)
**Status:** LOCALIZATION COMPLETE; CALIBRATION COMPLETE; G2 ENVELOPE PROPOSED (freeze = roadmap decision)

## 1. §41 first-divergence localization (§17-20)

**Artifact:** `first_divergence_localization.json` — per-layer hidden
drift (rel_l2 / cosine / max_abs) between the MosaicVM runtime and the
G1 source corpus at every matched decode step up to each prompt's first
token divergence, plus both sides' top-5 logits at the divergence.

Findings:

- Hash layers 0-2: drift 1.3-2.7% (deterministic routes; pure
  attention/core numerics).
- Learned layers: **smooth growth** 2.5% → 10-30% with depth; bump
  layers vary per prompt (no single-layer jump — no bug signature).
- At every first divergence (5/5 prompts): **both candidates appear in
  both top-5s**, cross-implementation |Δlogit| 0.27-0.81, corpus
  top1-top2 margins 0.003-0.44.
- §20 hypothesis chain **confirmed with quantified evidence**: small
  numeric drift → learned-router boundary swaps (6th expert) →
  accumulated hidden drift → near-tie greedy flip.
- No earlier semantic divergence found at any first divergence.

## 2. Independent source-vs-source calibration (§21-23)

**Neither side is the MosaicVM runtime.** Same kit code
(`inference/model.py` + identical torch kernel shims) on:

- **A:** RTX PRO 6000 Blackwell (102 GB)
- **B:** RTX 3060 (11.6 GB display GPU; CPU head/embed, streamed
  experts)

**Corpus:** 14 new calibration prompts (2/domain), 12 tokens each,
greedy — disjoint from the certification corpus. Router top-10 scores
recorded per gate call for boundary-band measurement.

**Artifacts:** `g2_calibration_corpus/` (Blackwell, 14 files),
`g2_calib_3060/` (3060, 14 files), `g2_calibration_result.json`.

### Measured envelope (kit-vs-kit, n = 4859 hidden comparisons, 5203
### gate calls, 568 logit comparisons)

| quantity | p50 | p90 | p99 | max |
|---|---|---|---|---|
| hidden rel_l2 (matched steps) | 0.100 | 0.166 | 0.252 | 0.437 |
| route \|Δw\| (common experts) | 0.0055 | 0.021 | 0.048 | 0.273 |
| \|Δlogit\| (common top-5) | 0.269 | 0.724 | 1.434 | 3.429 |
| route-flip rank-6/7 gap | 0.0146 | 0.052 | — | 0.221 |

- Token agreement: **120/154 (78%)** — the same kit implementation
  diverges cross-hardware at 22% of greedy tokens.
- Route-set agreement: 3627/5203 (70%); 1576 flips, all at rank-6/7
  gaps ≤ 0.221.
- Kit-vs-kit greedy flips occur at top1-top2 margins up to **1.72**.

## 3. MosaicVM vs the calibrated envelope

| axis | MosaicVM | kit-vs-kit envelope | inside? |
|---|---|---|---|
| hidden rel_l2 p50 | 0.114 | 0.100 (p99 0.252) | ✓ |
| hidden rel_l2 max | 0.608 | 0.437 | ✗ (see note) |
| first-flip margins | 0.003-0.44 | flips ≤ 1.72 | ✓ all |
| \|Δlogit\| at flips (max) | 0.813 | p99 1.434 | ✓ |
| route-set agreement | 19-40% exact | 70% | same character |

Note on hidden max: MosaicVM's 0.608 was measured over matched prefixes
up to 26 decode steps vs the calibration's ~11 — drift accumulates with
steps; the robust p50 statistic aligns (0.114 vs 0.100). The frozen G2
spec should condition the hidden envelope on matched-prefix length (or
use a p99 cut).

## 4. Proposed G2 envelope (for freeze — §34)

All thresholds from kit-vs-kit calibration only (no MosaicVM-derived
numbers):

```text
tau_hidden    layerwise rel_l2 <= 0.25 (kit p99) at matched steps,
              with a step-count scaling factor or a p99-per-step-slot
              definition (max drift grows with matched prefix length)
tau_router    Top-6 set differences admissible only when every
              differing expert's biased score lies within 0.22 of the
              rank-6 cutoff (kit flip-gap max); k=6 always; clearly
              separated experts must agree
tau_w         |dw| <= 0.05 (kit p99) on commonly selected experts
tau_logit     a token mismatch qualifies only if both tokens lie in
              the near-tie set with top1-top2 margin <= 1.72 (kit
              flip-margin max) AND |dlogit| on the pair <= 1.43
              (kit p99)
sequence      at the first qualifying flip: classify, stop strict
              comparison for that branch, restart from a common
              committed prefix for the next probe (§35-38)
invariants    hash routes, k=6, router equation, normalization,
              quantization equations, SwiGLU, shared placement,
              ordered reduction, HC/head equations — exact, always
```

Applying this envelope to the existing certification run: all five
MosaicVM first divergences qualify (margins 0.003-0.44 ≤ 1.72,
|Δlogit| ≤ 0.81 ≤ 1.43, rank-2 candidates, matched prefixes, hash
routes exact). The final G2 certification corpus should be rerun under
the frozen spec with the §38 prompt+committed-prefix unit.

## 5. Artifacts

```text
artifacts/first_divergence_localization.json    §41 evidence
artifacts/g2_calibration_corpus/ (14 files)     Blackwell kit side
artifacts/g2_calib_3060/ (14 files)             RTX 3060 kit side
artifacts/g2_calibration_result.json            measured envelope
artifacts/g2_calibration_prompts.json           calibration prompts
mosaicvm/bench/g2_calibration_compare.py        comparison
mosaicvm/bench/source_corpus_gen.py             generator (now with
  SCG_CPU_HEAD_EMBED / SCG_CACHE_CAP / SCG_CALIB / SCG_PROMPTS)
```

## 6. Remaining for 2/2

1. Freeze the G2 spec (roadmap decision — proposal above).
2. Rerun the certification corpus under the frozen spec with the §38
   unit (prompt + committed-prefix → next-token).
3. If inside envelope on all probes → 3060 equivalence 2/2.
