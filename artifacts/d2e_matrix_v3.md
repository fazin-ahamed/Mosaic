# D2-E CPU source adapter matrix — 2026-09-08

Evidence: `d2e_matrix_v3.json` and `d2e_matrix_v3.log`.

Run:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-cu/bin/python -m mosaicvm.bench.d2_cpu_expert --matrix
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-cu/bin/python -m mosaicvm.bench.vnni_source
```

The 24-case matrix covers checkpoint L41/L42 experts 0 and 1 at M=1,2,3,4,8,16.
All 72 linear-stage BF16 results and all 24 complete Expert.forward BF16 results
match the ordered CPU reference bit-for-bit. The existing torch source shim
also matches these tested stage and final results. No tolerance is used.
The checkpoint Expert AST runs unchanged, preserving clamp, FP32 SwiGLU,
route-weight multiplication before the DOWN-input BF16 cast, and DOWN.
Each case records source/tensor hashes and expanded weight bytes.

Inputs are seeded synthetic BF16 rows, not captured tail activations. Declared
synthetic route weights span 0.05–1.5 for multiple rows; M1 uses 0.05. Neither
real router coverage nor held-out student residual coverage is established.
One expert's three matrices are prepared at a time; closure cycles are collected
between cases. Execution is CPU-only, bounded to two torch/OpenMP threads.

Adapter checks cover every finite E4M3 encoding (254), signed/fractional extrema,
an exact int64 K32 oracle, all admitted E8M0 scale bytes 1–254 with a bounded
one-term dot including subnormal FP32 output, and rejection of E8M0 0/255 and
both E4M3 NaN encodings. The scale sweep is a separate bounded test, not a claim
that every arbitrary scale/activation product stays finite. The three-plane
kernel is unchanged. Exact integer representation does not certify arbitrary
native floating-point reduction order.

Classification: `LOCAL_CPU_SYNTHETIC_REAL_EXPERT_MATRIX_REFERENCE_ONLY`.
This is bitwise agreement on the tested ordered CPU-reference cases, not
`SOURCE_EXACT`, full-tail/native-source certification, or production latency.
The prepared representation is 31,457,280 bytes per routed expert, including
int8 weights, FP32 scales, and int32 compensation sums; activation planes and
other temporary allocations are additional.

Full-tail replay remains gated. The existing source replay and StreamLinear
loader explicitly move core/state/weights to CUDA. A CPU adaptation would also
need CPU-versus-captured-source evidence for FP8 core reductions, HC, attention/
DSA, router, shared expert, and ordered routed accumulation before h41/h42/head
could certify this adapter. None of those CPU source-core comparisons exists
in this matrix. No CPU full-tail or production timing was run or inferred;
old v4-B timings do not apply. Actual captured expert-input/route replay is the
next parity extension once those inputs are available.
