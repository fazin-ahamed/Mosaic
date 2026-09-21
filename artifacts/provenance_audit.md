# Provenance audit — stale-CDN archive consumers (roadmap §104 NEXT 1)

Date: 2026-09-03 · Authoritative archive: `a9089f25fa85bac924a7c159dd67960aa901fd3d7126babcf5e202d950597983`
Stale archive (forensic): `.mosaicvm/core/core.stale-cdn.safetensors`

## Scope of contamination

Tensor-by-tensor comparison (`artifacts/provenance_tensor_diff.json`):
**ALL 1661/1661 tensors differ** between stale and authoritative (shapes and
dtypes identical for every tensor; every value differs). The stale CDN copy
was a wholesale different weight dump — not a partial corruption. Any
VALUE-dependent result produced from the local archive before the
2026-09-02 rebuild consumed stale bytes.

## Audit table

| artifact / harness | bytes consumed | class | action |
|---|---|---|---|
| core_classification.json | stale | shape/dtype classification only | **RETAIN** (metadata identical 1661/1661) |
| phase29_report.json (LUT/ratio, mxint-vs-qpa) | **synthetic random experts — no archive dependency** | value-insensitive throughput | **RETAIN** (reclassified 2026-09-03: phase29r.py uses torch.randn experts/activations, never reads the archive) |
| gemm_race parity results | stale values (both sides of each comparison) | kernel-vs-kernel parity | **REQUALIFY** on authoritative bytes — internally consistent but production NUMERIC_PARITY must be re-established; the "NaN-aware 0xFF-scale" handling exercised stale-only paths now dormant |
| dsa_bench.json | stale values | P1 timing + finite_frac | timing **RETAIN**; finiteness fractions obsolete (clean data ⇒ ~100% finite; trivial recheck) |
| routes_overlap_corpus_3060.json | Blackwell hub model (authoritative) | generative | **RETAIN** (corpus is clean — reclassified 2026-09-03) |
| routes_overlap_3060_result.json | — | generative | **BLOCKED_ON_EXACT_3060_RUNTIME** — the streaming V4Model is a simplified non-exact implementation (no RoPE/compressor/indexer/act-quant); exact 3060 execution needs the ExactAttention integration (TRUE P2 scope). See result artifact for the six runtime fixes applied this session |
| compute_floor_3060.json | timing | value-insensitive | **RETAIN** |
| true_p2.json / true_p2_mixed.json | timing | value-insensitive (NaN computes at full speed) | **RETAIN** timing; E_A fields derive from psc_audit (sandbox runtime — unaffected) |
| p2_exact_dsa.json | timing (delta method) | value-insensitive | **RETAIN** timing |
| verifier_floor_sweep.json | timing (ran pre-rebuild on stale bytes) | value-insensitive | **RETAIN** — note: measured 2026-09-02 pre-rebuild; shapes/layouts/dtypes identical; NaN/Inf full-speed on GPU |
| Case-A kill gate (k_cycle_3060) | timing | value-insensitive | **RETAIN** |
| mem_audit / single_stack_fit / transport | memory + timing | shape-based | **RETAIN** |
| fetch_core.py | acquisition method | — | **SUPERSEDED** — never reuse; hub/Xet only |
| rebuild_core_hub.py | acquisition | — | canonical re-acquisition path |

## Explicitly unaffected (sandbox runtime / hub bytes)

golden_routes.json · dspark_stage1_inspection.json · all DSpark/B3/B3-R/
B3-S-RFIX/B4-O artifacts · nanospark datasets · AUTHORITATIVE_BLACKWELL_
ROUTING corpus · psc_audit (branchpack oracle inputs).

All were produced from the sandbox's `snapshot_download` model (xet-backed,
authoritative) — verified: runtime q_b(L2) sha16 65d1cf8c02b74b16 ==
authoritative archive tensor.

## Test suite

269/269 green post-rebuild — tests use random weights, no archive
dependency.

## Standing rule

`fp8_core.CORE_PATH` now points at the authoritative archive; all future
local runs consume authoritative bytes automatically. Requalification
priority: (1) 3060 overlap corpus regen, (2) gemm parity re-establish,
(3) phase29 LUT/ratio recheck.
