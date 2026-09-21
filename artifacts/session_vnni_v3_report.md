# Session Report — Exact VNNI Kernel: Parity-Proven, First Real Numbers

**Date:** 2026-09-05
**Roadmap:** Belady Closed + CPU Dual-Rail Narrowed to Exact VNNI

## Track V — the exact VNNI packed-FP4 kernel exists and is exact

**Formulation** (MXIE class, VNNI-realized, source-exact):
- qW = 2·LUT[nibble] ∈ {0,±1..±12} — int8
- activation = mA·2^eA: mA = 4-bit e4m3 mantissa ∈ 0..15 (int8),
  eA = per-128-block pow2 exponent
- per-32-weight-block integer dots via `vpdpbusd` (products ≤ 180,
  block sums ≤ 5760 — exact int32)
- y = sA · Σ_wb sW·dot32 / 2 — only fp32 ops are pow2 scale multiplies

**PARITY: bit-exact** (max_abs = 0.0 vs the fp32 reference across
random weights/activations at all block boundaries). The source
exactness contract holds on the CPU rail — no BF16 shortcut, packed
codes + scales throughout.

Two real bugs were caught by the hand-check + parity discipline:
1. lane-semantics bug (vpdpbusd's 8 lanes are K-groups of one output
   row, not 8 output rows — the v2 kernel computed 1/8 the work at
   8× impossible speed),
2. a factor-2 convention error in the REFERENCE (the kernel was
   right; the qW=2W factoring).

## Measured performance (v3, correct, 12 threads)

| rows/expert | ms/expert | packed-equiv GB/s | TFLOPS-eq |
|---|---|---|---|
| 3 (realistic mean 3.3) | 0.83 | 15.1 | 0.18 |
| 8 | 1.84 | 6.8 | 0.22 |
| 16 | 3.33 | 3.8 | 0.24 |

Thread scaling: 1T 0.043 → 6T 0.14-0.27 → 12T 0.18-0.24 TFLOPS-eq.

## Hybrid projection (measured, v3 kernel)

All-CPU N32 block (2537 objects): **1.99-2.12s**. Balanced GPU/CPU
split: f ≈ 0.55-0.57, **Tcrit ≈ 1.13-1.18s → perfect-proposer
ceiling ≈ 27-28 tok/s.**

vs the roadmap's 0.4 TFLOPS model (Tcrit 0.75s → 42.5): the v3
kernel is ~1.6-2× short. Identified optimization headroom (v4):

1. **Packed-nibble reads** — v3 reads int8-expanded weights (2× RAM
   traffic: 25MB vs 13.4MB packed). On-the-fly nibble unpack ≈ 2 ops
   per 32 bytes vs 16 VNNI MACs → halves traffic toward the 15→30+
   GB/s packed target.
2. **Tiled VNNI layout** (8 outputs per vector, activation broadcast)
   — eliminates the per-block horizontal sum (~50% of v3's
   instruction count) → ~2× arithmetic throughput.
3. Both together target the ~0.4 TFLOPS-eq class where the model's
   30-35 tok/s practical band lives.

## Status against the VNNI gate (§29)

Not yet passed (0.24 < 0.4 measured; ceiling analysis says v4
optimizations plausibly close the gap). Not failed. The kernel is
EXACT and the economics are now measured rather than modeled:

```text
v3 measured:      Tcrit 1.13-1.18s -> 27-28 tok/s perfect-proposer
v4 target:        Tcrit ~0.75-0.9s -> 35-42 tok/s perfect-proposer
practical (roadmap §34): 30-35 tok/s if overheads behave
```

## Artifacts

```text
mosaicvm/bench/vnni_kernel.c       exact AVX-VNNI kernel (cb6e4aeb)
mosaicvm/bench/libvnni_kernel.so   compiled (gcc -O3 -mavxvnni -fopenmp)
mosaicvm/bench/vnni_bench.py       parity + benchmark (d18e7991)
artifacts/vnni_kernel_result.json  measured results (17702801)
```
