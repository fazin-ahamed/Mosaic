# Session Report — imma4 Certified + P3 Deadline Answer + Stage-1 Resumed

**Date:** 2026-09-04
**Roadmap:** Stage-1 Recording + GU-imma4 + JIT Expert Streaming Phase

## Track D — imma4 full TRUE-P2 certification: **PASS**

Recorded by pausing the Stage-1 recorder (restartable; 12 prompts
preserved), running the full sweep with imma_k=4, then resuming.

**New canonical fit** (`true_p2_imma4.json`):

$$
T_{\rm verify}(N) = 0.297\,s + 8.36\,ms \cdot N
$$

vs the previous 0.305 + 9.52 — **slope -12.2%, intercept -8ms**. The
full-block wins exceed the isolated-kernel projection:

| N | T (imma8) | T (imma4) | delta |
|---|---|---|---|
| 16 | 0.4818s | 0.4277s | -54ms (-11.2%) |
| 32 | 0.7039s | 0.5997s | -104ms (-14.8%) |
| 56 | 0.9538s | 0.8000s | -154ms (-16.1%) |

New verifier-only E_req: **N16 = 12.92, N32 = 16.94** (fit-based).
Verify-only capacity: N32 ≈ 56.7 candidates/s. The practical proposer
target relaxes to ~14-18. Intercept decomposition (imma4): routed
0.128s, swap 0.058, attn 0.044, hc 0.025, dsa 0.021, shared 0.011.

## Track P3-A — deadline simulator: the movement answer is quantitative

`p3_deadline_sim.json` — real golden routes + TRUE-P2 layer timing,
warm-persistent-cache steady state (cold-start affects only block 1):

**Per-layer demand reality** (corrects the 3.43GB mental model): a
32-token block needs **~62 unique experts per layer** (32×top-6 with
partial overlap), 93 new at L0, 48 at L1, saturating to the full
256-expert union by ~L10. Cross-layer reuse is broad-front, not
layered-local.

| cache C | VRAM | misses/block | moved | H2D | unhidden bubble |
|---|---|---|---|---|---|
| 64 | 0.86GB | 192 | 2.57GB | 199ms | 863ms |
| 128 | 1.72GB | 128 | 1.72GB | 133ms | 288ms |
| 160 | 2.14GB | 96 | 1.29GB | 100ms | 84-94ms |
| **192** | **2.57GB** | **64** | **0.86GB** | **67ms** | **18-27ms (3-4%)** |
| 256 | 3.43GB | 0 | 0 | 0 | 0 (doesn't fit VRAM) |

Findings:

1. **Pure JIT / no cache is dead** — cold per-layer demand (833MB at
   L0) cannot meet any deadline; LRU below C≈192 thrashes 3-7× the
   minimum movement.
2. **C=192 (2.57GB) is the minimum deadline-feasible staging**: 64
   misses/block, 67ms H2D, 18-27ms bubble (3-4% of the block).
3. **The serialized H2D stream at demand peaks is the binding
   constraint, not the prefetch window** — oracle 1-layer prefetch
   (O1) barely improves over reactive JIT (286 vs 288ms at C=128).
   Route predictors help only by pre-spreading the stream.
4. **Memory Pareto emerges**: core 7.8 + C192 2.57 + workspace 0.5 ≈
   10.9GB → ~0.7GB proposer on GPU; with CPU embed/head (+43ms tax):
   ~2.8GB free → C192 + 2.1GB proposer co-fit. Alternatively C160
   (+84-94ms bubble) buys back VRAM. These are the P3 geometry
   choices, now priced in ms and MB.

## Track S — Stage-1: recording resumed, race ready

Recorder resumed with skip-done (12/126 done, ~6.5h remaining,
unattended). A/B/C architecture race code validated end-to-end. One
batch's worth of TRAIN prompts already on disk.

## Updated Exact-30 arithmetic (imma4)

```text
T(N) = 0.297 + 0.00836N
E_req(16) = 12.92   E_req(20) = 13.93   E_req(24) = 14.93
E_req(28) = 15.94   E_req(32) = 16.94
+ P3 overheads at N32: C192 bubble ~20ms, CPU-head tax 43ms (if used)
```

## Artifacts

```text
true_p2_imma4.json        imma4-certified sweep + fit
p3_deadline_sim.json      warm-cache deadline analysis
gu_sweep.json             parameter race (imma4 winner)
stage1_record.py          recording (resumed, restartable)
stage1_train.py           A/B/C race (validated)
```
