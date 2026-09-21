# Exact single-layer parity vs kit source — PASS (rung 1 of §43 ladder)

Date: 2026-09-03 · Harness: scripts/validate_exact_layer.py
Archive: a9089f25fa85bac9… (authoritative)

Reference: the checkpoint's own Block code (inference_model.py, real
authoritative weights, bf16 leaves, boundary-free kernel shims — fp4_gemm
shim = dequant-fp4 + F.linear, matching the runtime-under-test's expert
semantics).

Runtime-under-test: ExactAttention (dsa_layers) + mosaicvm Router/
HashRouter + dequant_fp4 routed experts + FP8-dequant shared expert,
source hc_pre/hc_post/norms shared (not under test).

| layer | class | finite | max_abs | rtol2% | rtol0.2% | router |
|---|---|---|---:|---:|---:|---|
| 5 | routed (learned gate) | both ✓ | 0.031 | 95.5% | 65.0% | idx equal, Δw=0 |
| 0 | hash (tid2eid) | both ✓ | 0.012 | 98.7% | 87.2% | idx equal, Δw=0 |

The full-layer parity matches the DSA-only validation class (95-99%
rtol2% for non-indexer layers) — composing attention + HC + router +
FP4 experts + shared expert amplifies nothing beyond the known bf16
cast-boundary noise.

## Two runtime findings (fixed in-harness, production-relevant)

1. **Router bias dtype**: mosaicvm Router's e_score_correction_bias
   defaults to the ambient dtype (bf16 under bf16 defaults) while the
   source Gate uses fp32 — a bf16 bias flips near-tie top-6 selections
   (observed: 4/256 experts mismatched on a 64-token probe). Production
   V4Model must force the fp32 bias. (Same class as regression bug 6.)
2. **Hash-router input shape contract**: the router expects the
   collapsed [s, HIDDEN] activation with matching-length token ids.

## Next rungs

multi-layer → 43-layer + head golden → one-token greedy → short greedy
→ 3060 overlap corpus (the BLOCKED_ON_EXACT_3060_RUNTIME item).
