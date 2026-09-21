# Exact-DSA memory/performance census (§70 NEXT 1 / §39–43)

Date: 2026-08-29 · Level: **P1 module census**, RTX 3060, thermal **UNKNOWN**
Artifacts: `artifacts/dsa_state_census.json` (raw), this file (findings)
Harness: `mosaicvm/bench/dsa_state_census.py` (rerunnable)

State bytes are exact per-buffer accounting validated against closed-form
formulas (`analytic_match: true` for every component after the cache-dtype
fix). Decode latency = median of 50 single-token steps per point, caches
populated at context size (timing is value-insensitive); the fill-based
method validated at 4K against a real prefill (ladder 1.84 ms vs real
2.16 ms — same order, method representative).

## 1. The preliminary 30 MB/layer figure was a misreading

The ~30 MB figure was **~30–46 MB TOTAL across all 43 layers at 4K**, not
per layer. Measured semantic state (bf16 caches, fp32 states):

| context | semantic total (43 layers) | per-layer r4 / r128 / r0 |
|---:|---:|---|
| 4K | **46.0 MB** | 1.52 / 0.69 / 0.13 MB |
| 16K | 130.6 MB | |
| 64K | 468.7 MB | 21.2 / 1.18 / 0.13 MB |
| 256K | 1.82 GB | |
| 512K | 3.62 GB | |
| 1M | **7.23 GB** | |

The §40 concern (1.29 GB at 4K) does not exist. **The real pressure is
context scaling**: at 1M, semantic state is 7.23 GB — 97% of it the
ratio-4 layers (21 × compressed KV 268 MB + 21 × indexer cache 67 MB).
Bounded components (window rings 128×512, kv/score states) total ~13 MB
at ANY context — they are structure-bounded exactly as §41 suspected.

Note: caches were found registered fp32 (2× the source's bf16 design) and
fixed to bf16 (source semantics) during this census; all 19 parity tests
remain bitwise-green after the change.

## 2. Freqs tables are a 28 GB implementation artifact at 1M

Current implementation materializes per-layer RoPE tables (3 per ratio-4
layer, 2 per ratio-128, 1 per ratio-0 = 105 tables):

| context | freqs artifact (43 layers) | shareable (3 unique tables) |
|---:|---:|---:|
| 4K | 110 MB | 3.1 MB |
| 64K | 1.76 GB | 50 MB |
| 1M | **28.19 GB** | 805 MB |

The tables are value-identical across layers of the same theta class, so
deduplication (or on-the-fly computation) is exactness-preserving and
**mandatory** before any long-context run. Not semantically per-layer
state; excluded from the semantic column above.

## 3. Decode latency (reference path, module level)

Median single-token decode, bf16 caches, reference torch implementation:

| context | ratio-0 | ratio-4 (+indexer) | ratio-128 |
|---:|---:|---:|---:|
| 4K | 1.19 ms | 1.84 ms | 1.31 ms |
| 64K | 1.22 ms | 1.84 ms | 1.27 ms |
| 256K | 1.18 ms | 2.47 ms | 1.32 ms |
| 512K | 1.18 ms | 3.18 ms | 1.32 ms |
| 1M | 1.18 ms | **4.69 ms** | 1.55 ms |

Compression-event steps add ~+1.0 ms (r4: 2.85 vs 1.84) / ~+0.3 ms
(r128: 1.64 vs 1.31).

Findings:
* ratio-0 and ratio-128 decode are effectively **context-independent**
  (window-bounded / C÷128-row gather of 8.4 MB at 1M).
* ratio-4 is the only context-scaling term: the indexer scores
  q[1,64,128] × compressed[C/4,128] in **fp32** + top-k over C/4 —
  ≈2.1 GFLOP/token/layer at 1M. The einsum currently upcasts to fp32
  (`.float()`); a bf16 einsum is an obvious 2×+ production lever.
* Model-wide reference-path attention at 1M ≈ 2×1.18 + 21×4.69 +
  20×1.55 ≈ **135 ms/token**. These are REFERENCE-path numbers (no
  fusion, fp32 math, per-loop heads) — they establish scaling laws, not
  production costs. Production kernels must close a large constant factor
  (the fused sparse-attn/indexer kernel work is future §63 scope).

## 4. Transients

* Decode-step peak transient: ratio-4 @1M = **132 MB** (indexer fp32
  scores + softmax intermediates), ratio-128 @1M = 25 MB, ratio-0 =
  0.65 MB constant. Fits alongside state at every context.
* **Prefill gather is O(S·(win+S/ratio))**: `sparse_attn_ref`
  materializes an S×(128+S/4)×512 fp32 gathered tensor — 9.6 GB at
  S=4090 (OOMed the 12 GB card during this census). Long prefill needs
  **chunked prefill**, and the chunked path (start_pos>0 with s>1) is
  currently unimplemented — the same gap the P2 verification blocks need
  (N-token chunks at a position). This is now a named prerequisite for
  both long-context prefill and the post-DSA P2 rerun.

## 5. Placement verdict (§40 answer)

* **≤64K**: full-GPU semantic state (46–469 MB) plausibly coexists with
  the ~11.5 GB bank+core (tight at 64K once workspace is counted).
* **≥256K**: 1.8 GB+ semantic state cannot be fully GPU-resident beside
  bank+core → the §44 placement race is required. Structure favors:
  - ratio-4 compressed history is read **only via top-512 gather**
    (0.5 MB/layer/token) → RAM residency costs ~10 MB H2D/token total.
  - ratio-128 history is read in full (C/128 rows = 8.4 MB/layer at 1M)
    → 20 layers = 168 MB/token → ~13 ms/token at 12.9 GB/s H2D —
    significant but bounded; GPU residency or layer-local staging better.
  - Layers execute sequentially → layer-local H2D staging bounds
    GPU-resident history to one layer (~335 MB at 1M) at movement cost.
* Freqs dedupe (finding 2) is mandatory in every scenario.

## 6. Census-corrected next actions

1. Dedupe freqs tables (exactness-preserving; required for any ≥64K run).
2. Chunked prefill path (start_pos>0, s>1) — prerequisite for long
   prefill AND P2 verification blocks.
3. bf16 indexer scoring einsum (production lever, 2×+ on the only
   context-scaling decode term) — after P2 rerun priorities are settled.
4. P2 rerun (§48) now unblocked by this census: at benchmark contexts
   (≤4K effective) DSA state is 46 MB and decode/placement is not a
   constraint — the block is compute-valid AND memory-compatible there.
