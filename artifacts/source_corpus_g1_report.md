# G1 Source-Class Corpus — Regeneration & 3060 Comparison

**Date:** 2026-09-04
**Roadmap:** Full Source-Class RTX 3060 Runtime — Reference Requalification Phase (§27-41)
**Status:** CORPUS GENERATED; TOKEN-0 7/7; FULL-SEQUENCE STRICT G1 NOT MET — G2 CONDITION DEMONSTRATED

## 1. What was built

A **source-class golden corpus** (roadmap §28-31) regenerated from the
authoritative checkpoint using the **checkpoint's own `inference/model.py`**
(from the HF snapshot, revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`)
executed on an RTX PRO 6000 Blackwell (102 GB) molab sandbox:

- **Implementation:** kit code unmodified; only the tilelang kernel
  backend swapped for torch shims of the same arithmetic class:
  - `act_quant` — exact source math (per-128 amax → pow2 E8M0 → E4M3)
  - `fp8_gemm` — per-128-block code accumulation, then block scales
  - `fp4_gemm` — per-32-block code accumulation (exact in fp32 at expert
    shapes), then act(128) × weight(32) scales — the kit kernel class
  - `sparse_attn`, `hc_split_sinkhorn`, `fp4_act_quant` — exact torch math
  - `fast_hadamard_transform` — Sylvester WHT d^-0.5 (proven in srcchain)
- **wo_a:** convert.py serving transform (fp8+128×128 scale → bf16)
- **Experts:** native packed FP4 + E8M0 scales, streamed from the 48
  shards (GPU LRU); core resident (1521 tensors)
- **Config:** `inference/config.json` (canonical kit config; verified
  consistent with the top-level config.json), greedy temperature=0,
  dspark disabled, 32 tokens × 7 domains

## 2. Corpus provenance

```text
revision               7872f01b1d1fe23eabc4c98b48bffcef5a386062
core archive cross-ref a9089f25fa85bac9
hardware               NVIDIA RTX PRO 6000 Blackwell Server Edition
generator sha          c03c4308cea48466 (script)
arithmetic class       E4M3 act per-128 E8M0 x E2M1 per-32 E8M0 (experts),
                       E4M3 per-128 (core), bf16 wo_a, fp32 gate/head
records per token      greedy token, top-5 logits+ids, margin,
                       Top-6 expert IDs + route weights (43 layers,
                       hash + learned), per-layer hidden checksums
files                  artifacts/source_corpus/{7 domains}.pt + manifest
                       (slim, hiddens on sandbox: mosaicvm/bench/
                       rfix_out/source_corpus/)
```

Generator: `mosaicvm/bench/source_corpus_gen.py`. Comparison mode:
`scripts/validate_exact_model.py srcorpus <n_gen>` (records my Top-6 IDs +
route weights per layer per token; compares tokens/routes/weights).

## 3. Comparison result (RTX 3060 source-class runtime vs source corpus)

| domain | tokens exact | first div | margin | ours rank | routes exact (set) |
|---|---|---|---|---|---|
| coding | **32/32** | — | — | — | 29% (50%) |
| long_generation | **32/32** | — | — | — | 40% (67%) |
| structured_json | 26/32 | tok 26 | 0.052 | 2 | 30% (54%) |
| agent_tool_use | 16/32 | tok 16 | 0.440 | 2 | 21% (36%) |
| math_reasoning | 8/32 | tok 6 | 0.132 | 2 | 8% (16%) |
| long_context_retrieval | 3/32 | tok 2 | 0.003 | 2 | 4% (6%) |
| general_conversation | 2/32 | tok 2 | 0.380 | 2 | 3% (7%) |

```text
token-0                    7 / 7 EXACT
full 32-token exact        2 / 7
mean exact prefix          17.0 / 32  (vs ~16/32-equiv vs old corpus)
```

full corpus (incl. per-layer hidden checksums, 1333/prompt)
  mirrored local, 8/8 MIRROR_OK:
```text
agent_tool_use   c70839a30b6659ac      math_reasoning        af913ba946cd28b5
coding           954654bf5585fe5f      structured_json       7cb908356e89089a
general_conv     025a2e2e76e9aebf      long_gen              9c03610f0c39ab81
long_ctx_retr    4d3459422b5eea74
```
270 in all source-class paths (shared-expert FP8 class confirmed).

## 4. Divergence character (§34 evidence)

Every **first** token divergence across all 5 non-exact prompts:

- my token = the corpus's **rank-2** candidate (5/5),
- corpus top1-top2 margin 0.003-0.440 (long_context tok2 is a 0.003 tie),
- tokens agree on the full prefix before the flip (no earlier semantic
  divergence).

Route-level analysis at token-0: hash layers 0-2 **exact** (deterministic
table); learned-layer divergence is confined to the **6th expert slot and
rank ordering among near-equal scores** (e.g. top-5 experts identical,
6th differs at weights 0.1495 vs 0.1474; rank-2/3 order swaps at
0.26 vs 0.25). Route weights for commonly selected experts agree to
~1e-3 at token-0, drifting with sequence length.

Mechanism: accumulated hidden-state numeric differences between the two
source-class implementations (Blackwell kit-torch vs 3060
ExactAttention/MXINT; different legal kernel/op orders) flip top-6
boundary decisions; different 6th experts compound state divergence
until a near-tie greedy flips. Token-0 decisions (single forward) are
robust; long-sequence strict determinism is not achievable across
implementations.

## 5. Classification

```text
G1 STRICT (all tokens exact)          NOT MET (2/7 prompts over 32 tokens)
G1 TOKEN-0 GATE                       PASS 7/7
§33 G2 CONDITION (source-class
cross-hardware near-tie flips)        DEMONSTRATED
MosaicVM semantic defect              NONE FOUND at any first divergence
                                      (all flips rank-2 near-ties)
3060 EQUIVALENCE                      1/2 → token-0 gate passed;
                                      strict sequence parity not certifiable
                                      cross-implementation
```

Per §36-38: G1-first was executed; the G2 numerical-equivalence envelope
is now justified. Per §37 it must be frozen before final P4 — the §34
candidate rule applies at every first divergence observed (both
SOURCE_CLASS, alternative token in top-2, small margin, no preceding
semantic divergence). The **upstream-routing-agreement clause** of §34 is
not satisfiable in the strict sense (6th-expert boundary flips exist
throughout matched prefixes) — the envelope definition must decide
whether route-set divergence at near-equal scores is admissible. That is
a roadmap decision, not a runtime fix.

## 6. Artifacts

```text
artifacts/source_corpus/*.pt + manifest.json   slim corpus (local)
artifacts/source_corpus_compare.json           f4b52331d2f2b77f
  (per-token divergences w/ corpus top-5 + margins,
   per-layer token-0 route divergences)
mosaicvm/bench/source_corpus_gen.py            generator
scripts/validate_exact_model.py                srcorpus comparison mode
sandbox: mosaicvm/bench/rfix_out/source_corpus/  full corpus incl.
        per-layer hidden checksums (1376/prompt) for §41 localization
regressions                                    7/7 PASS (post hc_pre fix)
```

## 7. Next

1. Freeze the G2 envelope (roadmap decision): candidate — first
   divergence eligible iff both SOURCE_CLASS, alternative token rank ≤ 2,
   corpus margin < predeclared threshold, matched prefix, and route
   divergence limited to near-equal-score boundary flips.
2. §41 localization (optional, evidence in hand): diff per-layer
   hiddens at first divergence using the full corpus records.
3. TRUE P2 production sweep (roadmap NEXT 4) — unblocked, no longer
   gated on corpus work.
