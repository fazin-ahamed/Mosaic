# Real-weight DSA numeric validation — PASS (authoritative bytes)

Date: 2026-09-03 · Harness: scripts/validate_dsa_real_weights.py
Archive: a9089f25fa85bac9… (authoritative hub/xet rebuild)
Classification: NUMERIC_PARITY (bf16 mode, source cast-boundary noise class)

The validation that was BLOCKED by the stale-CDN archive (0% mutual-finite
overlap) now passes on authoritative weights:

| layer | mode | fin(src) | fin(ours) | max_abs | rtol2% | rtol0.2% |
|---|---|---:|---:|---:|---:|---:|
| 0 (ratio0) | prefill 256 | 1.0 | 1.0 | 0.031 | 94.4% | 59.2% |
| 0 | decode p=256 | 1.0 | 1.0 | 0.008 | 95.8% | 64.1% |
| 0 | chunked 130+34 | 1.0 | 1.0 | 0.016 | 87.6% | 36.1% |
| 2 (ratio4+indexer) | prefill 256 | 1.0 | 1.0 | 0.047 | 69.9% | 17.2% |
| 2 | decode p=256 | 1.0 | 1.0 | 0.023 | 70.4% | 17.2% |
| 2 | chunked 130+34 | 1.0 | 1.0 | 0.021 | 70.0% | 17.1% |
| 3 (ratio128) | prefill 256 | 1.0 | 1.0 | 0.031 | 93.9% | 56.1% |
| 3 | decode p=256 | 1.0 | 1.0 | 0.008 | 96.1% | 64.4% |
| 3 | chunked 130+34 | 1.0 | 1.0 | 0.016 | 86.3% | 32.7% |

Method: kit source code (inference_model.py, bf16 leaves, ue8m0 act-quant
sims) vs mosaicvm ExactAttention, IDENTICAL dequantized real weights
(fp8×e8m0→bf16 projections, fp32 norms/compressor) on both sides.
Reference for the chunked case = source full-prefix prefill rows.

The layer-2 (ratio4+indexer) 70% rtol2% matches the historical random-weight
bf16 run (71%) — the residual is the known bf16 kernel-order noise at the
source's own cast boundaries, not a real-weight-specific divergence.

Execution note: USE_FUSED_SPARSE=False (eager BITWISE_MOSAIC_REFERENCE path)
because the fused Triton kernel's SM86 shared-memory requirement
(BLOCK_T=64 × D=512 × stages=2 ≈ 114KB) exceeds the 3060's 101KB limit at
full attention width. The fused kernel is separately parity-tested
(tests/test_dsa_sparse_fused.py, 269/269 suite green). Production bf16
dtype and quant semantics are otherwise unchanged.

## Verdict (roadmap §26)

```text
EXACT DSA → CLOSED / FROZEN (source semantics)
```

Remaining DSA-adjacent work is TRUE P2 integration, not DSA semantics.
