# Session Report: ProofCache-0/1 — PC0 boundary pin, PC1 exact-reuse census, PC2 certificate screen

**Date:** 2026-09-21
**Status:** Executed the document's first three gates against real source
arithmetic. **Path A (exact reuse) is closed: zero repeats in 5,151 ops.**
**Path B (certified reuse) is screened out at this boundary: closest-pair output
distance is 3-10× the quantization step**, so proximity certificates cannot
authorise reuse here. The ~86-94% byte avoidance that 30 tok/s requires is not
reachable by either path at this boundary.

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
lever** — consistent with the architecture document's own warning.

## 3. PC2 — nearest-neighbour certificate screen

Path B needs *different* inputs to yield the *same* required output. Measured
on the 6 most-invoked (layer, expert, op) groups in trajectory 0, pairing only
same-shape calls (prefill `[S,4,4096]` vs decode `[1,4,4096]`):

| group | invocations | closest input cosine | relative output dist | act_quant step |
|---|---|---|---|---|
| 42_159_w1 | 19 | 0.9787 | 0.1570 | 0.0335 |
| 42_159_w3 | 19 | 0.9787 | 0.1528 | 0.0294 |
| 42_159_w2 | 19 | 0.9672 | 0.3003 | 0.0330 |
| 41_62_w1 | 20 | 0.9731 | 0.1529 | 0.0497 |
| 41_62_w3 | 20 | 0.9731 | 0.1136 | 0.0423 |
| 41_62_w2 | 20 | 0.9657 | 0.2267 | 0.0440 |

- Best input cosine across all groups: **0.979** — the closest inputs are
  genuinely nearby, confirming the interesting-part-of-the-design premise.
- But the resulting **relative output distance is 0.114–0.300**, versus an
  `act_quant` relative step of ~0.029–0.050.
- **Minimum ratio: 0.1136 / 0.0440 ≈ 2.6×; worst: ~9×.**

Certified cell reuse requires the output distance to fall **below roughly half a
quantization step** (identical output code), or to be exactly 0 (bitwise
reuse). At 2.6-9×, the closest input pairs still produce outputs separated by
several quantization cells. **Proximity-based certificates cannot authorise
reuse at this boundary.**

## 4. Interpretation

- **Path A (exact reuse): closed.** Zero repetition under the generous bound.
- **Path B (certified reuse): screened out at this boundary.** The mechanism is
  mathematically coherent, and the toy integer experiment in the architecture
  document is genuinely interesting — but on real source arithmetic, the
  closest naturally-occurring input pairs are far too distant for
  quantization-cell certificates at the FP8 boundary. This is the document's
  own predicted failure mode ("bounds cross quantization cells almost always"),
  now measured rather than assumed.
- **Path C (adaptive residual disclosure):** the toy showed savings only under
  artificially coarse quantization (step 512) and sparse inputs; dense
  fine-quantization cases read ~100% of bits. Real FP4 has no demonstrated
  certificate rate, and the storage layout question (whether residual bytes are
  physically separable at I/O granularity) remains open and unaddressed by
  this run.
- **The gap is unchanged:** 30 tok/s needs ~86-94% byte avoidance; all measured
  reuse/certificate paths deliver 0%.

## 5. What this does and does not settle

- **Establishes:** at the pinned RoutedLinear boundary on real captured
  trajectories, neither exact reuse nor proximity certification can remove
  weight reads. The architecture's first three gates are answered negatively.
- **Does NOT establish:** that no other boundary could. The screen covered the
  6 most-invoked groups of the L41/L42 path on 4 trajectories; other
  boundaries (e.g. integer accumulators with proved range, or coarser existing
  source boundaries) were not tested. But each candidate must carry its own
  soundness argument and measured rate — none is currently demonstrated.
- **Max achieved throughput remains unmeasured.** 0.151 tok/s reference capture
  is still the only achieved number.

## 6. State

```text
Frontier (staging)              CLOSED - no useful traffic reduction
Result cache (exact reuse)      CLOSED  - 0.00% exact repetition (corrected scope)
ProofCache Path A               CLOSED  - 0/5151 repeats, 0% cold bytes avoided
ProofCache Path B (certified)   SCREENED OUT at this boundary - 2.6-9x step gap
ProofCache Path C (residual)    untested; toy showed savings only with coarse Q
Exact-30                        unsupported on this hardware
max achieved throughput         UNMEASURED (0.151 tok/s reference only)
```

No ProofCache implementation. The frozen proposer, exactness work, sealed BLIND,
and frozen FRONTIER-0 v4 remain untouched.

The standing requirement is unchanged: the next work needs a specific,
independently testable mechanism with a measured byte-saving opportunity large
enough to matter. ProofCache was tested on its own gates and did not clear them.

Artifacts: proofcache01.py, proofcache01.json, result_cache_census.py (v1,
superseded — accumulation bug documented), result_cache_census.json.
