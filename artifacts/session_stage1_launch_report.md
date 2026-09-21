# Session Report — Stage-1 Launch + GU Race + P3 Movement Insight

**Date:** 2026-09-04
**Roadmap:** G2 Certified — Stage-1 Strong-Proposer Phase

## Track S — Stage-1 (PRIMARY): infrastructure complete, recording in flight

- **Dataset**: `stage1_prompts.json` — 126 disjoint prompts (7 domains ×
  18; 12 train / 3 val / 3 test per domain, prompt-disjoint splits,
  excludes all G2 certification prompts). 40 tokens per prompt →
  ~5,000 state windows at K=32 horizons.
- **Recorder** (`stage1_record.py`): batched (12/batch) kit-model
  generation with the kit generate.py masking semantics; captures full
  [4, 4096] fp16 states at exits 36/38/40/42 per generated position.
  Includes a recording-speed expert path (FastStreamLinear: pre-scaled
  bf16 cached weights + source act-quant + tensor-core GEMM; class
  equivalence verified — differences are bf16-output-quantization
  scale, ~0.6% mean, immaterial for proposer data vs the 10% drift
  envelope).
- **Status**: batch 1/11 recorded (12 prompts, 2368s); ~7h to full
  completion, running unattended and restartable (per-batch saves +
  skip-done). Bottleneck is Python-level per-expert launch structure
  (36% GPU util) — a fused-batch MoE recorder would be the next
  speedup if more datasets are needed.
- **Architecture race** (`stage1_train.py`): A independent horizon
  mixings (2.1M), B shared trunk + horizon bias (67M), C state-chain
  decoder (46M) — all read ONE L40 state, emit K=32 horizon proposals
  through the frozen lm_head in ONE batched head GEMM. Loss: per-horizon
  CE. Eval: top-k by horizon, path survival, greedy-block E[A]
  (prefix+1 semantics, math validated), proposal latency. Code smoke-
  PASSED end-to-end on batch-1 data (CPU); full training runs when the
  GPU frees.

## Track D — GU race at Exact-30 widths (`gu_sweep.json`)

**imma_k=4 wins at every target width** (vs current default imma8):

| N | imma8 | imma4 | delta |
|---|---|---|---|
| 16 | 5.59ms | 4.81ms | -14% |
| 24 | 8.13ms | 7.01ms | -14% |
| 32 | 10.06ms | 9.31ms | -7.5% |
| 112 | 21.69ms | 16.62ms | -23% |

Projected block impact at the target widths: ~0.75-0.8ms/call × 43
layers ≈ 32-33ms/block (5-7% of T(N)). **Pending: full TRUE-P2 delta
rerun with imma4 to certify the block-level win (roadmap §21 gate).**

## Track P3 — hot-bank study (`hotbank_study.json`): the movement insight

From real Blackwell golden routes (7 prompts × 43 layers × 63 tokens):

- **Every block of N≥16 tokens touches ALL 256 experts** (6 experts ×
  43 layers × N tokens — union saturates). Static frequency hot-banks
  CANNOT reduce within-block misses: bank128 = 50% hit, 128 missing
  experts, 133ms unhidden H2D per block.
- **Per-layer just-in-time streaming IS feasible**: total block
  movement 256 × 13.4MB = 3.43GB at steady 12.9GB/s = 266ms, vs
  T_verify(32) = 610ms — an average of only 5.6GB/s is needed.
  **Expert movement can be entirely hidden by pipelined per-layer H2D**
  (double-buffered: layer L+1's experts stream while L computes).
- This REPLACES the hot-subset-first P3 shape with **full-bank RAM
  warm tier + pipelined streaming** as the primary design; static
  hot-banks remain only a latency-miss safety net. Route prediction
  for prefetch is unnecessary at N≥16 (the router for block b+1 can be
  computed early — routes depend on cheap early layers).

## Status

```text
G2 certification          PASS (last session) — 2/2 stands
GU imma4                  measured winner at N16-32 (TRUE-P2 delta pending)
P3 movement design        JIT streaming viable; static banks closed
Stage-1 dataset           recording (batch 1/11 done, ~7h, restartable)
Stage-1 race code         smoke-passed, ready to train on completion
```

## Next (when the GPU frees)

1. `python mosaicvm/bench/stage1_train.py` — full A/B/C race on the
   126-prompt dataset (train/val/test), E[A] + latency + params report.
2. TRUE-P2 rerun with imma_k=4 (validate the GU block-level win).
3. If E[A] ≥ 10-12 credible → B4-O integration + multi-token head
   variants; if not → data-scale or architecture iteration.
