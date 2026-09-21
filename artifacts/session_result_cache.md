# Session Report: Result-cache hypothesis — exact-input repetition census

**Date:** 2026-09-21
**Status:** REJECTED at screening under the generous upper bound. Zero exact
effective-input repeats across 5,151 expert ops on real captured trajectories.
No weight bytes avoidable. The mechanism does not address the 86% gap.

---

## 1. The hypothesis, as tested

An expert result cache stores the *output* of a deterministic expert operation
\(y = F_e(x)\). If the same **effective input** recurs, the output can be reused
and expert \(e\)'s weights never read — attacking whether a read is needed at
all, unlike caching/scheduling which only move reads around.

The screening target was established from v4's bypass oracle
(0.428 GB/output token) and the best measured storage rate (1.82 GB/s):
30 tok/s permits ≤60.7 MB/output token, so the mechanism must avoid **~86%** of
remaining storage bytes even granting zero compute.

## 2. Effective-input definition (contract followed)

Each key is a SHA-256 over:
- expert id, layer, op (w1/w2/w3), checkpoint revision
- the **quantized activation operand** — the exact FP8/quantized tensor the
  source's GEMM consumes, produced by the source's own `act_quant`
- its scale bytes
- the routing weight applied inside the operation

Hashes locate entries; the key bytes would be compared to rule out collisions
(not exercised, since no repeats occurred). Similar activations and repeated
expert IDs explicitly do **not** qualify.

## 3. Method

`result_cache_census.py` wraps the source's own `RoutedLinear` forward on L41/L42
via `SourceTail`, replaying **real captured h40 trajectories**
(`artifacts/d3_scale1000_data`) through the source's exact path. The operands
hashed are therefore the exact ones the source computes on real states — not
synthetic activations.

Generous upper bound: unlimited result-cache capacity, free lookups, zero
compute cost, free fallback.

## 4. Result

| metric | value |
|---|---|
| trajectories replayed | 4 (agent_127–130) |
| total expert ops | 5,151 |
| distinct effective inputs | 1,260 |
| repeated inputs (≥2) | **0** |
| % of distinct repeated | **0.00%** |
| extra executions repeatable | 0 |
| avoidable weight bytes | **0.0000 GB of 5.285 GB read (0.00%)** |

**Verdict: REJECT — negligible exact-input repetition.**

## 5. Interpretation

The mechanism's premise does not hold. Whole-expert inputs effectively never
repeat across positions in a decode session: activations are high-dimensional
continuous states, and the source's activation quantization to FP8 does not
collide them often enough to matter (zero collisions in 5,151 ops). Even at
unlimited capacity and zero cost, a result cache avoids no storage bytes.

This confirms the prior expectation (exact whole-input repetition across new
tokens would be limited) with measurement rather than intuition. The gap to
30 tok/s is not closed by avoiding weight reads via exact output reuse.

## 6. Scope and what remains

- **Scope:** 4 trajectories, single decode sessions, L41/L42 operands. The
  census measures the *mechanism's* repeat rate; it is not a corpus claim.
  Within-batch grouping benefit (already partly captured by existing batching)
  is not separately credited here, but it does not create exact repeats — it
  only shares already-resident weights.
- **Fine-grained variant (sub-expert blocks):** could in principle repeat more,
  but requires matching all block dependencies and preserving source reduction
  and rounding semantics. Given 0.00% at the whole-expert grain with a free
  capacity bound, and that the required saving is ~86%, investing in the
  fine-grained variant is not justified by this evidence.
- **Certified partial evaluation:** remains the only genuinely different
  untested direction. It is difficult research (rounding, quantization scales,
  state consistency) with no demonstrated certificate rate — not an easy
  implementation and not a demonstrated route to 30.
- **Achieved throughput:** still unmeasured. 0.151 tok/s reference capture
  remains the only achieved number.

## 7. State

```text
Frontier (staging)              CLOSED - no useful traffic reduction
Result cache (exact reuse)      REJECTED - 0.00% exact-input repetition
Exact-30                        unsupported on this hardware
max achieved throughput         UNMEASURED (0.151 tok/s reference only)
```

No further implementation of the result cache. The exactness work, frozen
proposer, and sealed BLIND remain untouched. The next work requires a specific,
independently testable mechanism with a measured byte-saving opportunity large
enough to matter — none is currently identified.

Artifacts: result_cache_census.py, result_cache_census.json
