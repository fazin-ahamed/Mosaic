# Session Report — D1-R: Exact Tail Replay PASSES (Bit-Exact)

**Date:** 2026-09-06
**Roadmap:** L42 FP32 Ceiling = 100% — State Prediction + Deep-Tail
Replay Become the Primary Proposer Program

## D1-R verdict: **PASS — source L41/L42 replay from true h40 is
## architecturally exact**

Method: true h40 trajectory (captured from the full kit model) →
sequential Block 41 forward (building L41's KV/DSA state position by
position from h40 only) → sequential Block 42 forward → compare
against the model's own h42.

| prompt | positions | h42 cosine p50 | h42 max_abs | replay head Top-1 |
|---|---|---|---|---|
| coding_17 | 29 | 1.000001 | **0.0** | 78.6% |
| coding_18 | 27 | 1.0 | **0.0** | 76.9% |

**h42 replay is BIT-EXACT** (max_abs = 0.0, cosine = 1.0). The L41/
L42 tail, when given the true h40 trajectory and building KV/DSA
state sequentially from h40 alone, reproduces the model's own h42
without any numerical difference.

The ~76-79% head Top-1 average is diluted by PROMPT positions (where
the prompt's next token ≠ the model's greedy choice — expected). At
DECODE positions, the replayed h42 produces the model's own greedy
token (by construction, since h42 is exact).

## What this proves

**The deep-tail corrector architecture is architecturally valid:**

```text
true future h40 → real L41 → real L42 → fp32 head → perfect prediction
```

The h40→L41→L42 chain works when:
- L41/L42 KV/DSA state is built sequentially from h40 inputs
- No additional state beyond h40 is needed for the tail
- The replay is deterministic and exact

## What remains unproven

1. **D1-B — approximate-state robustness**: does the tail still work
   when h40 is APPROXIMATE (from a student)? The tail's KV/DSA/indexer
   may be sensitive to h40 errors.
2. **D2 — tail cost**: actual wall time of L41/L42 execution (the
   tail expert bank ~6.86 GB fits in host RAM; VNNI v4-B can
   execute it).
3. **Student learnability**: can a practical-size model predict
   future h40 states well enough?

## The critical next experiment: D1-B

The D1-B curve — **state prediction error → tail output Top-k** —
is now THE decisive measurement for the proposer:

- If the tail is robust to moderate h40 errors: the student doesn't
  need perfect state prediction → the architecture works
- If the tail is chaotic: the student must predict h40 extremely
  accurately → consider direct h42/logit distillation instead

## Artifacts

```text
artifacts/d1r_tail_replay.json    D1-R results
mosaicvm/bench/d1r_tail_replay.py  replay harness
```
