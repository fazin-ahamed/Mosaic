# Session Report: ProofCache-0/1 — PC0 boundary pin, PC1 exact-reuse census, PC2 output-equality oracle

**Date:** 2026-09-21
**Status:** Executed the document's first three gates against real source
arithmetic, with PC2 corrected mid-review from a distance heuristic to a
direct output-equality oracle. **Path A (exact reuse) is closed: zero repeats
in 5,151 ops.** **Complete-output reuse shows zero opportunity: 0 of 1,029
eligible pairs produce identical BF16 outputs.** Certificate feasibility at
other boundaries or finer granularities is **not established** — neither
claimed nor closed. The ~86-94% byte avoidance that 30 tok/s requires is not
delivered by any measured reuse path.

---

## 0. A correction to my own previous experiment

`result_cache_census.py` v1 had an accumulation bug: `tail.census` was cleared
per trajectory, so the repetition analysis ran on the **last trajectory only**
(1,260 ops) while the report summed all four (5,151). The zero-repeat verdict
survived, but the scope claim was wrong. `proofcache01.py` accumulates across
all trajectories and reports per-trajectory counts during the run. Corrected
numbers below.

## 1. PC0 — pinned canonical operation boundary

- **Boundary:** source `RoutedLinear` on L41/L42, ops `w1` / `w2` / `w3` — the
  project's already-parity-certified exact expert path.
- **Effective input:** the source's own `act_quant(x)` → FP8 operand `q`
  (128-element blocks) + block scales `sa`. The GEMM consumes `q` and `sa`;
  raw `x` alone is **not** the operand.
- **Weights:** FP4 E2M1 codes + per-block scales, keyed by (layer, expert, op).
- **Output:** BF16 `[.,2048]` (w1/w3) or `[.,4096]` (w2), consumed by SwiGLU →
  w2 → routing-weight multiply (outside this boundary) → expert sum → residual.
- **Routing weight** is applied outside the boundary, so it is not part of this
  boundary's effective input — it belongs to the expert-sum boundary.

Route IDs alone cannot establish repetition; the census hashes the quantized
operand and scale bytes, exactly as the contract requires.

## 2. PC1 — exact-reuse census (generous upper bound)

Unlimited capacity, free lookups, zero compute, accumulated across all
trajectories:

| metric | value |
|---|---|
| trajectories (agent_127–130) | 4 |
| total expert ops | 5,151 |
| **distinct effective inputs** | **5,151** |
| repeated inputs (≥2) | **0** |
| within-trajectory repeats | 0 |
| across-trajectory repeats | 0 |
| avoidable cold bytes | **0.0000 GB of 21.605 GB read (0.00%)** |

**Gate: FAIL.** Zero exact repetition. The required avoidance is ~86% (at
1.82 GB/s, zero other work) to ~94% (with other time). Achieved: 0%.

The mechanism's premise does not hold at this boundary: inputs to a given
expert op are high-dimensional continuous states, and the source's FP8
activation quantization does not collide them. **Path A is closed as a major
## 3. PC2 — direct BF16 output-equality oracle (corrected test)

The original PC2 was a **distance heuristic, not a certificate test**: it
compared relative BF16 output distances against an FP8 `act_quant` step,
mixing two different boundaries, and a "distance below half a step" criterion
is not sufficient for cell equality in any case (values 0.49 and 0.51 are
0.02 apart but round to different integers). It has been **replaced by a
direct output-equality oracle** — the decisive test on already-captured outputs:

- Calls grouped by (layer, expert, op) and compatible shape.
- Complete BF16 outputs hashed to locate; `torch.equal` confirms (hash never
  authorizes).
- Chronological pairing: an earlier identical output would be reusable.
- Identical outputs from different inputs are an **opportunity upper bound
  only** — discovering equality after execution supplies no cheap
  pre-execution certificate.

Result, the 6 most-invoked groups in trajectory 0:

| group | invocations | eligible pairs | identical outputs |
|---|---|---|---|
| 42_159_w1 | 19 | 153 | **0** |
| 42_159_w3 | 19 | 153 | **0** |
| 42_159_w2 | 19 | 153 | **0** |
| 41_62_w1 | 20 | 190 | **0** |
| 41_62_w3 | 20 | 190 | **0** |
| 41_62_w2 | 20 | 190 | **0** |

**Total: 0 identical output pairs of 1,029 eligible.**

No pair of distinct calls at this boundary produces the same complete BF16
output. A sound checker therefore cannot authorize reuse of earlier complete
outputs for any sampled call. **The narrow negative result is now
established directly**, without any distance heuristic.

Scope limits, stated explicitly:
- This tests **complete-output** reuse at the pinned RoutedLinear boundary
  only. **Post-`act_quant` equality is a different boundary and remains
  untested.**
- **Partial-output / row / block reuse is untested.** Zero whole-call repeats
  (PC1) does not establish zero row or block repeats, particularly when calls
  contain multiple rows.
- **Residual disclosure (Path C) is untested.**
- Coverage: 6 groups, trajectory 0, 1,029 pairs.


## 4. Interpretation

- **Path A (exact reuse): closed.** Zero repetition under the generous bound
  (0/5,151).
- **Path B (certified reuse): not established by the current test.** The
  corrected test is a **direct equality oracle**, which establishes a narrow
  negative: no complete-output reuse opportunity exists in this sample
  (0/1,029 pairs). This does **not** prove certificate feasibility either way —
  a cheap pre-execution certificate is a different question from post-hoc
  output equality, and partial-output and post-quantization boundaries remain
  untested. The mechanism is neither defended with optimism nor closed by a
  distance heuristic.
- **Path C (adaptive residual disclosure):** the toy showed savings only under
  artificially coarse quantization (step 512) and sparse inputs; dense
  fine-quantization cases read ~100% of bits. Real FP4 has no demonstrated
  certificate rate, and the storage layout question (whether residual bytes are
  physically separable at I/O granularity) remains open and unaddressed by
  this run.
- **The gap is unchanged:** 30 tok/s needs ~86-94% byte avoidance; all measured
  reuse paths deliver 0%.

- **Establishes:** at the pinned RoutedLinear boundary, no exact input reuse
  (0/5,151) and no complete-output reuse opportunity (0/1,029 eligible pairs,
  direct byte equality). Path A is closed; complete-output reuse at this
  boundary has no opportunity.
- **Does NOT establish:** that no other boundary or finer granularity could
  work. Partial-output/row/block reuse, post-quantization boundaries, integer
  accumulators with proved range, and residual disclosure were not tested.
  Each candidate must carry its own soundness argument and a measured rate —
  none is currently demonstrated. Coverage: 6 groups, 1,029 pairs, 4
  trajectories at L41/L42 only.
- **Max achieved throughput remains unmeasured.** 0.151 tok/s reference capture
  is still the only achieved number.


## 6. State

```text
Frontier (staging)              CLOSED - no useful traffic reduction
Result cache (exact reuse)      CLOSED  - 0.00% exact repetition (corrected scope)
ProofCache Path A               CLOSED  - 0/5151 repeats, 0% cold bytes avoided
ProofCache Path B (certified)   NOT ESTABLISHED - 0/1029 complete-output equal
                                pairs (direct oracle); certificate feasibility
                                open; partial/post-Q boundaries untested
ProofCache Path C (residual)    untested; toy showed savings only with coarse Q
Exact-30                        unsupported on this hardware
max achieved throughput         UNMEASURED (0.151 tok/s reference only)
```

No ProofCache implementation. The frozen proposer, exactness work, sealed BLIND,
and frozen FRONTIER-0 v4 remain untouched.

The corrected status, stated exactly:

> **PC1:** zero exact whole-call input reuse in the reported sample.
> **PC2:** complete-output equality oracle found 0/1,029; certificate
> feasibility is not established by the current test.
> **Exact-30:** unsupported; maximum achievable throughput remains unmeasured.

The next work needs a specific, independently testable mechanism with a
measured byte-saving opportunity large enough to matter. ProofCache was tested
on its own gates; Path A failed PC1, and complete-output reuse showed no
opportunity — but neither this test nor any prior one closes the underlying
idea at every boundary, and none should be read as a universal maximum.

Artifacts: proofcache01.py, proofcache01.json, result_cache_census.py (v1,
superseded — accumulation bug documented), result_cache_census.json.
