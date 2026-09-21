# Session Report — C192 Real Streaming Implementation + Auto-Chain

**Date:** 2026-09-04
**Roadmap:** imma4 Certified + C192 P3 Residency Solved — Stage-1
Single Gating Unknown

## Delivered this block

### 1. C192 real streaming implementation (`c192_stream_bench.py`)

The roadmap §52/§79 design, implemented:

- **Pinned warm tier**: all 256 experts (real layer-5 weights, packed4
  layout) in pinned CPU RAM — 3.43 GB.
- **C192 slot-mapped VRAM working set**: `MXINTPacked4Bank` over
  mutable slot arrays (2.57 GB) — the bank object stays valid while
  slots are loaded/evicted in place.
- **Async H2D**: dedicated copy stream + per-slot CUDA events; LRU
  eviction with in-flight-transfer guarding; free-list slot
  management.
- **Per-layer pipeline**: `ensure(ids)` at layer start (transfers
  overlap the attention half) → `wait(ids)` before the MoE → slot
  remap (one D2H + dict map + one H2D per layer — no per-element GPU
  syncs) → `fused_moe_mxint` on the working-set bank.
- **Policies**: reactive JIT and oracle-1-layer lookahead (golden
  routes known ahead — upper bound on any predictor).
- **Block**: the exact TRUE-P2 full block (imma4 GU, real HC/router/
  shared/head, DSA via reconstruction is NOT included in this bench —
  it measures the streaming overlay against the static-bank block
  times).
- Warm start = top-192 by trace frequency (steady state, matching
  `p3_deadline_sim`).

Two performance bugs caught pre-run: broken free-list allocation
(post-eviction slot collisions) and a per-element `.item()` remap that
would have added ~8k GPU syncs/block.

### 2. Auto-chain (unattended overnight execution)

`/tmp/autochain.sh` armed:

```text
Stage-1 recording (running, 114 prompts left)
  -> stage1_train.py  (A/B/C race, held-out E[A], latency, params)
  -> c192_stream_bench.py  (real streaming vs static-bank block times)
```

All results land in `artifacts/` with logs in /tmp. The single most
consequential benchmark (Stage-1 held-out E[A]) executes first when
the recording completes.

## Decision table ready (from the roadmap)

| held-out E[A] | action |
|---|---|
| ≥19-20 | freeze proposer → integrate C192 P3 immediately |
| 18-19 | near-viable: geometry/latency race at N16-32 |
| 15-18 | strong signal, needs another gain |
| 10-15 | escalate architecture |
| <10 | close A/B/C family |

All three candidates (2.1M / 67M / 46M params) are P-SMALL class —
the 43ms CPU-head tax likely avoidable for the first race.

## Artifacts

```text
mosaicvm/bench/c192_stream_bench.py     real streaming implementation
/tmp/autochain.sh                       unattended execution chain
artifacts/c192_stream_bench.json        (pending — auto-chain)
artifacts/stage1_race_result.json       (pending — auto-chain)
```
