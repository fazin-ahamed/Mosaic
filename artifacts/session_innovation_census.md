# Mosaic Innovation Execution — input-side exact decomposition census

**Status: NEGATIVE (kill signal).** Innovation Execution, as pinned at the
source RoutedLinear L41/L42 input boundary, does not avoid storage. At the
basis budget tested it achieves **0.09% avoidance against a required
86–94%**, and its zero-residual probability is **exactly zero**. This closes
the input-side variant of the hypothesis for the captured trajectories.

Evidence class: **SIMULATED** (replay of captured source operands on a CPU
reference; no kernel was built, no throughput is claimed anywhere below).

## Arithmetic boundary precondition

Innovation Execution is only *valid* if the source arithmetic exposes an
exact integer boundary. The source's own reference path
(`source_corpus_gen.fp4_gemm`) accumulates the FP4×FP8 dot product in
**FP32**, not integers. A mathematically exact boundary does exist:

- `act_quant` uses `s = 2^ceil(log2(amax/448))` — a **power-of-two** scale
  per 128-column block (E8M0-like).
- FP4 weight codes decode to {0, ±0.5, …, ±6} and FP8 E4M3 activation codes
  are dyadic, so every per-32-block product is exact in FP32 at expert
  shapes (the shim's own contract note).
- The project's MXINT K32 identity (`qA = 512·A_E4M3`, exact integer,
  |qA| ≤ 229376; `qW = 2·W_E2M1` ∈ [−12, 12]; per-K32 |Σ qA·qW| < 2³¹)
  therefore stands as an exact integer boundary, and scales are applied
  once after the integer dot — the decomposition never has to reason about
  a non-integer value.

This census used that integer boundary. The decomposition itself is exact
by construction (int64 reconstruction asserted per call), so the negative
result is about *sparsity*, not about arithmetic validity. What remains
unproved is the **output-side equivalence obligation**: no reconstructed
GEMM was executed and compared bit-for-bit to the source's FP32
accumulator + BF16 cast. That obligation is now moot for performance
purposes — even a perfectly valid reconstruction saves nothing (below).

## Result — 4 trajectories, 82880 K32 groups, 770,240 calls

| metric | value |
|---|---|
| row-block calls | 770,240 |
| groups tested (≥2 calls) | 82,880 |
| exact reconstruction failures | 0 (all 770,240 decomposed exactly) |
| **P(residual = 0)** | **0.0000** (0 / 770,240) |
| mean residual nnz | **31.54 of 32** (98.6%) |
| P(nnz ≤ 8) | 0.0014 |
| **P(nnz ≥ 28)** | **0.9905** |
| calls served (nnz ≤ 8) | 1,082 (0.14%) |
| basis adds | 769,158 (711 per serve) |
| avoidance, served-only best case | 77.8% |
| **avoidance, all calls** | **0.0939%** |
| required avoidance | **86–94%** |
| pages avoided | ~0 (see below) |

Residual nnz distribution (nnz : count): the mass piles at the top.

```
3:109  4:76  5:234  6:114  7:246  8:303  9:227  10:207  11:178  12:206
13:187 14:229 15:179 16:251 17:202 18:309 19:233 20:199 21:253 22:246
23:354 24:260 25:296 26:682 27:1508 28:4339 29:14044 30:44462
31:105073  32:595034
```

595,034 of 770,240 calls (77%) have residual nnz = 32 — the *full width of
the group*. Zero columns avoided, zero pages avoided. The 1,082 "served"
calls are 0.14% of traffic and each cost ~711 basis-vector reads to build
the cache that served them.

## Physical page read model

A K32 group under a column-major layout is 32 columns × `n_rows` × 17 bytes
(16 packed FP4 + 1 scale). At 4 KiB pages a residual of nnz columns touches
nnz disjoint page ranges; nnz = 32 touches the entire group. The 0.09%
all-calls figure is the honest number: it counts every call, charges served
calls for residual pages plus the int32 action cache, and charges admitted
calls the full group read that built the basis.

Storage gate, per K32 group: the int32 action cache beats the group's own
weight bytes only when `r_b < k/8 = 4`. The observed basis saturates the cap
of 3 immediately (`basis_hist`: 604,140 of 770,240 calls sit at the cap),
so even the 0.14% served slice is at the gate's edge — and it is 0.14% of
traffic.

## Why this is decisive, and what it does NOT decide

This decides the **input-side** hypothesis at the tested parameters:
captured L41/L42 operands do not lie near a small common lattice. With basis
cap 3 and serve threshold nnz ≤ 8 the served rate is 0.14% and the
avoidance 0.09%; the requirement is 86–94%. The gap is ~920×.

It does not decide:
- **Larger basis budgets.** This run used `MAX_BASIS = 3` (under the K32
  storage gate `r_b < 4`). A budget at or above the gate cannot be justified
  by storage; it would need a different argument (e.g. compute reuse) and
  would change the arithmetic-charge model.
- **The output side.** `Wx = Zc + Wr` was not tested; `x = Bc + r` was.
  The input side failing at 0.09% makes an output-side success much less
  likely — the residual that defeats input reuse is exactly the residual
  that must be multiplied by the weights — but it is a separate experiment.
- **Other layers or full-precision operands.** Only L41/L42 w1/w2/w3, only
  the MXINT integer view, only 4 trajectories (agent_127–130).
- **Any throughput claim.** No kernel was built; nothing here is tok/s.

## Scope and honesty caveats

- Input-side only; L41/L42; 4 of the captured trajectories; K32 grain.
- The page model assumes column-major within each K32 group. The resident
  packed-FP4 bank is row-major, where skipping a column is impossible —
  so the modeled layout is the *favorable* one and the real number is worse.
- The 86–94% bar comes from `B_oracle ≈ 0.428 GB/token` vs 1.82 GB/s
  measured storage (60.7 MB/token at zero other time), already recorded in
  the FRONTIER-0 session.
- The exactness contract held throughout: 770,240/770,240 exact int64
  reconstructions, 0 approximations, 0 failures.

## Recommendation

Do not build an Innovation Execution kernel. Do not spend kernel time on
output-side `Wx = Zc + Wr` unless a new argument appears for why operands
should lie near a small lattice — this census says they do not, by 920×.
Record the exact-output-equivalence obligation as closed-moot: even a
bit-exact reconstruction would save 0.09% of expert weight bytes.

The capacity bottleneck stays where FRONTIER-0/EK0b put it: expert weight
movement, not input reuse. Return to H2/P3 — real (layer, expert) diversity,
measured CPU/H2D traffic, strategic residency — which is where the 30 tok/s
budget actually is.

## Artifacts

- `mosaicvm/bench/innovation_census.py` — the census (already the MXINT K32
  rewrite; the pre-rewrite version with the broken E4M3 decode and raw-code
  residual was discarded and is not cited anywhere).
- `artifacts/innovation_census.json` — full results, 82,880 group records.
