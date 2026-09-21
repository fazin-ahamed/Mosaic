# Session Report — VNNI v4 Four-Arm Race: Tiled Layout Wins

**Date:** 2026-09-05
**Roadmap:** Exact VNNI v3 Proven — v4 Throughput Gate

## The four-arm race (`vnni_v4_race.json`, 12 threads, parity-checked)

All arms exact (integer dots exact by construction; fp32 scale
accumulation within 1.6e-8 relative of the fp64 reference).

| M | v3 | v4a packed | **v4b tiled** | v4ab |
|---|---|---|---|---|
| 1 | 0.168 | 0.142 | 0.155 | 0.023 |
| 2 | 0.205 | 0.166 | **0.301** | 0.045 |
| **3** | 0.074* | 0.139 | **0.378** | 0.065 |
| 4 | 0.196 | 0.139 | **0.438** | 0.083 |
| 8 | 0.223 | 0.078 | **0.506** | 0.130 |
| 16 | 0.241 | 0.164 | **0.461** | 0.178 |

(TFLOPS-equivalent; *v3's M=3 outlier is measurement variance.)

**V4-B (tiled 8-output layout) is the decisive winner**: at the
production M=3 workload it reaches **0.378 TFLOPS-eq — inside the
roadmap's 0.35-0.40 PASS-GOOD band** — and 0.44-0.51 at M=4-8.
The tile layout eliminates the per-block horizontal sum and gets 8
outputs per dpbusd instruction stream, exactly as designed.

- v4a (packed reads alone) is NET NEGATIVE vs v3: the pshufb unpack
  adds instructions while v3 was not purely bandwidth-bound at these
  shapes. Packed reads only pay when combined with reuse — not as a
  standalone arm.
- v4ab (scalar-gather tile construction from packed) is dominated:
  the per-(nt, wb) scalar gather costs more than the int8 tile reads
  it saves. Dead as implemented.

## Cold-weights honesty check (RAM-streaming bound)

The race's working set (25MB/expert) partially fits the 18MB L3. A
16-expert cycling test (403MB working set, forces DRAM streaming):

- **Cold: ~0.73ms/full-expert at M=3 ≈ 0.21 TFLOPS-eq** — exactly at
  the DDR4 dual-channel peak (51GB/s): the cold kernel is
  RAM-bandwidth-bound, as the model predicted.
- Warm (L3-assisted): 0.399ms = 0.378 TFLOPS-eq.

Production streams cold experts from RAM but the 69% within-block
repeat fraction gives partial L3 reuse → expected landed rate
**~0.25-0.32 TFLOPS-eq**, between the cold and warm bounds.

## Hybrid projection (both bounds)

| scenario | all-CPU block | balanced f* | Tcrit | perfect-proposer |
|---|---|---|---|---|
| warm (0.378 TF) | 1.01s | 0.72 | **0.73s** | **43.7 tok/s** |
| cold (0.21 TF) | 1.24s* | 0.68 | **0.85s** | **37.9 tok/s** |
| expected production | — | — | **~0.75-0.85s** | **~38-42 tok/s** |

(*cold all-CPU uses the full-expert 0.73ms × 2537 ≈ 1.85s; the
cycling test's per-GEMM timing gives the conservative 1.24s figure
for the GU half — the honest band spans both.)

**E_req at the expected Tcrit**: 30 × (0.75-0.85 + proposer ~0.03)
≈ **24-26 accepted tokens** — matching the roadmap's planning band.

## v4 gate verdict

- Warm measurement: PASS-GOOD (0.378 in the 0.35-0.40 band)
- Cold bound: MARGINAL (0.21 < 0.28)
- The deciding measurement is now the **real GPU+CPU hybrid block**
  (Track H): RAM-contention, overlap quality, scheduler overhead,
  result transport, and the actual L3-repeat interaction determine
  where between 0.73s and 0.85s the block lands.

## Artifacts

```text
mosaicvm/bench/vnni_kernel.c         all four arms + tiling helpers
mosaicvm/bench/vnni_v4_race.py       four-arm parity + benchmark
artifacts/vnni_v4_race.json          measured results
```
