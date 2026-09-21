# A1.5 Complete — E[A] Invariant + KV Parity Both PASS

**Date:** 2026-09-05
**Roadmap:** Off-by-One Root Cause Solved — Fix KV/E[A] Reporting
Before T1

## E[A] three-way invariant: **PASS** (all three methods agree to 6
## decimals)

| method | value |
|---|---|
| direct accepted-count | 1.498246 |
| 1 + Σ S_k (cumulative) | 1.498246 |
| prefix histogram | 1.498246 |

**Corrected T0 E[A] = 1.4982** (the prior stored 1.4351 summed
exact-k counts instead of cumulative ≥k counts — a pure aggregation
bug, now fixed with a permanent three-way assert).

## KV rollout parity: **PASS — token-identical on all 285 held-out
## examples**

| metric | value |
|---|---|
| full K=32 sequence match | **285/285 (100%)** |
| per-token agreement | **1.0** |
| KV latency (K=32) | **22.5ms** |
| naive corrected latency | 35.6ms |

The clean single-pass KV rollout (seed at position 0, each drafted
token at position len(toks), per-block incremental K/V caches)
produces exactly the same tokens as the naive corrected rollout on
every example. The prior 3.33% agreement was the two-phase cache
construction bug — completely rewritten, not patched.

**KV latency 22.5ms is now CERTIFIED** — well inside the 15-25ms
preferred band.

## The frozen corrected T0 baseline

```text
S1                    0.4351  (raw L40 bypass)
S2                    0.0526
S3                    0.0105
S4+                   0
E[A]                  1.4982
P(h2 | h1)            0.121
P(h3 | prefix2)       0.20   (3 examples — hint only)
KV rollout latency    22.5ms (certified)
```

## Classification

The tiny causal proposer is now:
- seed-preserving ✓ (by construction, bypass)
- TF/FR path-equivalent ✓ (stage-trace 0.0 diff)
- KV-certified ✓ (token-identical, 285/285)
- E[A] invariant-enforced ✓ (three-way, permanent)
- latency-gated ✓ (22.5ms < 30ms)

**Mechanism validated at h2; chain depth insufficient (E[A] 1.50 vs
~24-28 needed).** T1 (temporal multi-state + deltas) is now fully
unblocked against a trustworthy baseline.

## Artifacts

```text
artifacts/a15_kv_parity.json     certified results
mosaicvm/bench/a15_kv_parity.py  invariant + KV parity harness
```
