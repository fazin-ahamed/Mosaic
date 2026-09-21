# Session Report — D2-COMPLETE-TAIL: Chunked Hybrid Tail Measured

**Date:** 2026-09-10
**Roadmap:** D2-P Complete (§44–§46)

## Complete L41/L42 wall (captured real h40 trajectories,
## genuine routes, p50 over 10 reps × 4 prompts)

Wall time in ms; includes the committed-prefix prefill (S
positions, batched) in every number:

| placement | geometry | K1 | K4 | K8 | K16 | K32 |
|---|---|---|---|---|---|---|
| GPU (reference shim) | serial | — | — | 1063 | 1674 | 2740 |
| GPU (reference shim) | chunk | 489 | 566 | 623 | 723 | 922 |
| **Hybrid (CPU-VNNI experts)** | serial | 407 | 571 | 755 | 1124 | 1931 |
| **Hybrid (CPU-VNNI experts)** | **chunk** | **351** | **402** | **479** | **584** | **768** |

## Headline findings

1. **Chunked routing dominates serial everywhere** — K32: 768ms
   vs 1931ms (hybrid), 2.5×. CENSUS CORRECTION (sidecar
   `d2_complete_tail_census.json`): the original JSON's per-replay
   call counts divided a per-rep-zeroed census by 40 and were
   ~40× too small. Correct per-replay traffic at K32: **chunk =
   508 expert calls (maxM 34) vs serial = 1402 calls (maxM 15)**
   — 2.76× fewer calls via row aggregation. The walls in the
   original JSON are unaffected (timing was correct; only counts
   were wrong). A census-only rerun confirmed the counts but ran
   under CPU contention from a concurrent training job, so ITS
   walls are invalid and were discarded — the original
   uncontended walls remain the certified D2 numbers.
2. **The hybrid CPU-expert rail is the fastest measured
   production-candidate tail** — hybrid_chunk beats the
   reference-shim GPU path at every K (768 vs 922ms at K32).
   CAVEAT: the GPU column is the kit's reference torch shim
   (einsum fp4_gemm), not a production CUDA kernel — it bounds
   nothing about optimized GPU execution. The comparison that
   matters is hybrid-serial vs hybrid-chunk and the CPU rail's
   absolute cost.
3. **Marginal speculative cost ≈ 13.5 ms/position** ((768−351)/31)
   at chunk geometry. The ~350ms K1 baseline is dominated by the
   committed-prefix PREFILL — in production, the tail's committed
   state persists across cycles (boundary reuse, D0): the
   per-cycle tail cost approaches K×13.5ms (K8 ≈ 108ms).
4. RAM: the 1200-expert prepared cache (~12.6GB) fit host RAM
   during the hybrid run (peak 22GB system). The measured full
   512-expert bank is 16.11GB — partial-residency with LRU is
   what actually ran.

## Implications for the cycle budget

With prefill reuse, a K8 hybrid-chunk tail ≈ 108ms marginal. At
the R ≥ 30 target: E[A]=8 needs the full cycle ≤ 267ms — tail
108ms + student + verifier + H2. The verifier (TRUE-P2 N32 =
565ms alone) remains the dominant term; tail economics are now
credible, verifier economics are not yet integrated (H2 open).

## Harness notes

Three OOM causes fixed en route: expert-cache caps per placement
(GPU 128 / CPU 1200), SourceTail max_seq 512→64 (captures need
≤47 positions), and the decisive one — the chunk path ran
WITHOUT inference_mode, so autograd retained every expert's
[M,N,K/32] einsum intermediate (~10GB on a K32 chunk).

## Artifacts

```text
artifacts/d2_complete_tail.json     full matrix + census
mosaicvm/bench/d2_complete_tail.py  harness
```
