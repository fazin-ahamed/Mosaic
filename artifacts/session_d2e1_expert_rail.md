# Session Report — D2-E1: CPU VNNI Expert Rail vs Source GPU Expert

**Date:** 2026-09-09
**Roadmap:** D3-v1 Breaks the Real Acceptance Floor (S46-S51)

## E1 result: PARTIAL — expert outputs at the bf16-ULP class, 4 mismatching cases

Live kit model on GPU (the source-class path used by the recorder),
forward hooks on all L41/L42 routed-expert StreamLinear modules,
real routing traffic from 3 validation prompts (convo/math/coding),
prefill + 8 decode steps. 684 distinct (layer, expert, matrix)
cases, M in {1..8, 14}:

| comparison | bitwise | max_abs |
|---|---|---|
| act E4M3 codes + E8M0 scales (GPU vs CPU) | 684/684 | 0 |
| GPU expert output vs CPU VNNI output | 680/684 | 1.22e-4 |
| GPU expert output vs CPU torch reference | 683/684 | sub-5e-7* |

*The 683/684 figure and the starred max_abs carry a harness caveat:
`cmp()` rounded max_abs to 6 decimals, so the one non-bitwise
reference case reports 0.0 while being a genuine non-equal output
(sub-5e-7 difference, likely a rounding-boundary or signed-zero
case). The harness is fixed to store unrounded values for future
runs; this JSON's rounded figures stand as recorded.

The 4 non-bitwise VNNI cases are bf16-ULP-boundary effects on
small-magnitude elements (rel <= 0.006; case magnitudes 1.6-8.9).
Cross-device fp32 accumulation order differs; after the bf16 cast
this is bit-identity in 680/684 cases.

## Classification (frozen discipline)

```text
D2-E1 = PARTIAL
E4M3/E8M0 ACTIVATION ADAPTER = BITWISE (684/684)
CPU VNNI EXPERT OUTPUTS = 680/684 BITWISE, 4 ULP-BOUNDARY CASES
NATIVE-SOURCE BITWISE CERTIFICATION = REMAINS OPEN
```

No tolerance promotion. The 4 cases are explicitly enumerated in
`d2e1_gpu_cpu_expert.json`. E1 does NOT freeze
"CPU EXPERT RAIL = CERTIFIED" at the bitwise bar.

## Localization evidence — and its limit

GPU-vs-CPU-torch-reference is bitwise in 683/684 cases with
sub-rounding-scale residuals, while the known full-tail CPU-vs-GPU
h42 mismatch is max_abs 4.0. This is strong EVIDENCE that the
dominant contribution to the tail mismatch lies outside the routed
experts — but it is NOT proof that experts are exonerated:
ULP-scale local differences can cross subsequent act_quant/router
boundaries and amplify. Whether the expert rail contributes
causally to the 4.0 tail drift is exactly what D2-E2 (GPU core +
CPU VNNI experts hybrid-tail replay vs source GPU tail) must test.

**CLOSURE (same day, per-case ablation, scoped to tested
traffic): D2-E2b substituted each of these four deltas ALONE into
the GPU reference tail (one call at a time, triggered exactly
once each) — all four runs produced bit-identical h42. Each delta
provably washes out in isolation ON THESE TRAFFIC CLASSES. For
the HISTORICAL all-CPU max_abs=4.0 probe, causality remains
UNLOCALIZED until that exact probe is replayed through the hybrid
topology (optional D2-E2c); the E2 prompts are not that probe,
and ~900/1536 expert combos remain un-enumerated. See
session_d2e2_hybrid.md and artifacts/d2e2b_per_case_ablation.json.**

## Coverage note

w1/w3 see raw routed rows; w2 sees route-weight-multiplied
(signed/fractional) activations — both real-activation classes
covered. M values 1-14 from natural routing (decode M=1, prefill
up to 14 rows per expert per call).

## Artifacts

```text
artifacts/d2e1_gpu_cpu_expert.json     684 cases + per-case detail
mosaicvm/bench/d2e1_gpu_cpu_expert.py  harness (cmp rounding fixed
                                        post-run; stored figures as
                                        recorded)
```

Production timing deliberately NOT claimed (D2-P after E2).
