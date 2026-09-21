# Session Report — A1 Parity Audit: OFF-BY-ONE ROLLOUT BUG FOUND AND FIXED

**Date:** 2026-09-05
**Roadmap:** T0 Audit Complete — Fix Free-Roll Equivalence Before T1

## Root cause found: the free-roll rollout had an off-by-one — every
## FR metric from the T0 and audit runs was computed on a lagged
## trajectory

The model's forward consumes `t[:, :-1]` as input tokens (position j
predicts token j). The original rollout fed the growing `toks` list
(length L) directly — so `t[:, :-1]` = `toks[0..L-2]`, and the output
taken (`lg[0, -1]`) was position L-1: **a re-prediction of the last
already-drafted token**, not the next one. Each appended draft token
was conditioned on a context missing the most recent token.

**The stage-trace proves it**: feeding TF and FR the same inputs
yields agreement 1.0000 with every intermediate tensor diff exactly
0.0 (embeddings → blocks → logits). The model was always correct;
the evaluator was wrong. The prior "85% agreement" and "5×
conditional collapse" were the bug's signature, not architecture
failure.

## Corrected T0 free-roll (same checkpoint, fixed rollout)

| metric | buggy rollout | **corrected** | TF reference |
|---|---|---|---|
| S1 | 0.4491 | 0.4351* | 0.4351 (bypass) |
| S2 | 0.0105 | **0.0526** | — |
| S3 | 0.0 | **0.0105** | — |
| S4 | 0.0 | 0.0 | — |
| E[A] | 1.4596 | **1.4351** | — |
| P(h2 FR \| h1 correct) | 0.023 | **0.121** | **0.125 TF** |

(*S1 differs because the buggy run's first FR token was the model's
own h1 re-prediction on 96/285 examples where it happened to differ
from the bypass; the corrected rollout starts from the bypass
prediction — S1 = 0.4351 exactly matches the bypass.)

**The corrected FR h2 conditional (0.121) now matches the TF
conditional (0.125) to within 0.4pt** — exactly what §20 demanded.
TF/FR path equivalence: **PASS**.

## What the corrected T0 baseline actually is

- S2 = 0.0526 (5× better than the buggy 0.0105)
- P(h2|h1) = 0.121 — the honest acceptance-critical conditional
- Conditional chain: P(h2|prefix)=0.121, P(h3|prefix2)=0.20,
  P(h4|prefix3)=0.0 — h3-conditional on surviving prefixes is
  actually 20%, but the sample shrinks to 3 examples
- E[A] = 1.4351: the chain dies at h2 for 88% of h1-correct examples
- KV rollout latency: **28.0ms** (below the 30ms gate); naive
  corrected rollout: 40.7ms (needs the KV cache)

## Diagnostic ceiling check (§37)

Roadmap predicted: if FR ≡ TF on h1-correct, S2 ≈ 0.4491 × 0.125 =
0.0561. Measured corrected S2 = 0.0526 ✓ (within sampling noise of
the 124-example h1-correct subset: 0.4351 × 0.121 = 0.0527 exact).

## Remaining honest state of the proposer

- The T0 architecture is now correctly measured: E[A] = 1.44,
  P(h2|h1) = 0.121. Still far from production (~24-28 needed).
- The causal mechanism's REAL conditional quality is 12% at the
  acceptance-critical position — this is the number T1 (multi-state +
  deltas) must move, now from a trustworthy baseline.
- KV-cache rollout at 28.0ms passes the latency gate.

## Artifacts

```text
artifacts/a1_parity.json             stage-trace + corrected metrics
mosaicvm/bench/a1_parity.py          parity harness + fixed rollout
/tmp/tiny_causal_t0.pt               T0 checkpoint (unchanged)
```
