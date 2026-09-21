# D2 CPU real-expert correctness probe — 2026-09-07

**D2 production execution remains open. Existing VNNI v4-B cannot directly consume source activations.** This probe executed the complete pinned checkpoint `Expert.forward` AST unchanged on CPU for L41 expert 0, with its actual six packed-weight/scale tensors (13,369,344 source bytes). Linear calls use the existing ordered per-K32 reference, then the existing corpus shim in a separate pass. Both include dynamic block-128 E4M3 activation quantization, E8M0 scales, BF16 linear outputs, source FP32 clamp/SwiGLU, route-weight multiplication before DOWN, and the source BF16 DOWN input cast.

The input is one seeded **synthetic** BF16 row and declared route weight 0.25. These are not captured h40, routed activations, or held-out student residuals. All three BF16 linear outputs and the final output matched bitwise between these two CPU references for this single case. The two activation quantizers matched codes/scales exactly. This is not a comparison against the native source GPU kernels or a global bitwise certification. No tolerance was used to promote a mismatch; per-stage exact equality/error counts are recorded in `d2_cpu_expert.json`, with checkpoint model and tensor SHA256 values. No timing claim is made.

## Concrete VNNI blocker

`vnni_gemm_tiled` uses unsigned-byte × signed-byte `_mm256_dpbusd_epi32`. `tile_acts` copies signed int8 into uint8 without sign compensation. A runnable minimal check supplies 32 activation values of -1 and source weights +1: the expected dot is **-32**, while the existing kernel produces **8160** (32 × 255). The raw unsigned-byte kernel is correct for its declared integer input domain; this finding identifies the missing source activation adapter, not invalid integer multiplication. Historical v4 race activations are restricted to 0..15 and cannot expose this issue.

The real-expert probe's actual quantized input contains 2,057 negative, 135 fractional, and 85 greater-than-255 E4M3 codes. The source DOWN activation contains 1,009 negative, 641 fractional, and 14 greater-than-255 codes. A direct int8/uint8 cast cannot preserve them. Exact MXIE qA=512A spans ±229,376; integrating it requires a demonstrated exact decomposition/sign treatment and preserved block-scale/reduction semantics. Existing v4-B warm/cold synthetic timings do not include that work and cannot establish production CPU expert latency. `mosaicvm/kernels/mxint.py:qint_encode_a` already supplies exact carry-propagating signed limbs for GPU MXINT and can be reused as an encoding reference. Inspection of every `vnni_gemm_tiled`/`tile_acts` call site found no corresponding CPU adapter: race/H1 use synthetic inputs, and `hybrid_block.py` forwards caller-supplied tiles. Signed-limb reuse still needs unsigned-byte compensation in VNNI and reconstruction before each K32 scale/reduction, rather than summing independently rounded full GEMMs.

## Reproduce

```bash
OMP_NUM_THREADS=2 .venv-cu/bin/python -m mosaicvm.bench.d2_cpu_expert --self-test
OMP_NUM_THREADS=2 .venv-cu/bin/python -m mosaicvm.bench.d2_cpu_expert
```

Both checks passed locally. Work was CPU-only, two Torch/OpenMP threads, one expert, with no checkpoint rewrite or GPU allocation. Scales 0/255 deliberately fail closed because those reference-helper edge semantics are not established by this probe. The next CPU step is exact source-activation support before benchmarking real expert diversity, full routed/shared reduction, or CPU/H2D economics. No historical artifact was modified.
