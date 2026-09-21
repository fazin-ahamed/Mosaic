# Session Report — Stage-1 Closed + P3 Movement Correction (Major)

**Date:** 2026-09-05
**Roadmap:** C192 Implemented — Stage-1 + Real Streaming Certification
Pending

This session produced three decisive findings. One closes a research
branch; two correct the P3 movement architecture with real data.

## 1. Stage-1 held-out E[A]: **1.2–1.5 → one-state A/B/C family CLOSED**

Dataset: 126 prompts recorded (all batches), 1076/267/285
train/val/test full-K windows, prompt-disjoint, G2-cert prompts
excluded. Frozen decision table applied without reinterpretation:

| arch | E[A] | top1@1 | top1@2 | top1@32 | survival@16 | latency | params |
|---|---|---|---|---|---|---|---|
| A horizon mixings | 1.495 | 0.460 | 0.084 | 0.004 | 0.0 | 3.5ms | 2.1M |
| B shared trunk | 1.081 | 0.081* | — | 0.032 | 0.0 | 4.2ms | 67M |
| C state-chain | 1.200 | 0.179 | — | 0.021 | 0.0 | 9.2ms | 46M |

*B failed to train (optimization artifact) — but A trained cleanly to
the raw-head baseline and still decays to 0.08 by horizon 2.

**The representational finding**: the L40 state contains the NEXT
token (46-55% top-1) and essentially nothing about token 2+. Horizon-2
top-1 = 0.084 vs 0.46 at horizon-1; path survival at 16 = 0.0. This is
structural, not underfitting — one deep state is not a multi-token
oracle. Per the precommitted table (E[A] < 10): **family closed as a
direct Exact-30 route**. Data-scale caveat (1076 full-K windows, not
the ~5000 designed) does not plausibly bridge a 12× gap.

## 2. The C192/C256 analysis was dimensionally wrong — and the golden
## route corpus is corrupt

Two compounding errors discovered while running the real streaming
bench:

- **Expert pools are PER-LAYER**: the universe is 43 layers × 256
  experts × 13.4MB = **147.6 GB**, not 3.43GB. The C192/C256
  single-pool framing, the "3.43GB total block movement", and the
  deadline-sim's feasibility all treated a global 256-expert pool.
- **The deadline sim v2 never enforced capacity eviction** — it
  modeled a warm-set size, not a hard cache cap. Real C192-LRU with
  eviction thrashes: 465 loads/block (6.2GB) per layer-pool scale,
  272-383ms unhidden — vs the sim's 64 loads / 18-27ms.
- **`golden_routes.json` captures are identical across prompts** (all
  7 MD5 = a548194e) — one trajectory replicated. Every route statistic
  derived from it (hot-bank hit rates, deadline sim inputs, frequency
  warm sets) used n=1 route data.

## 3. The REAL movement economics (from genuine G1-corpus routes)

Rederived from the 7-prompt G1 corpus gates (real kit-model routes,
43 layers × 32 tokens each):

```text
per-layer unique experts (N=32):  59 mean -> 34.1 GB/block
H2D at 12.9 GB/s steady:          2.65 s  vs 0.61 s compute
movement/compute ratio:           4.3x
adjacent-token top-6 overlap:     0.24-0.44 (mean 0.37)
cross-prompt expert overlap:      0.13-0.30 (static hot banks dead)
within-prompt 16-window reuse:    0.33-0.57 (mean 0.47)
4-token-recency coverage:         0.46-0.59 (mean 0.56)
```

**Exact-30 movement requirement at N=32**: cycle budget 1.067s −
0.61s compute = 0.457s movement = 5.9GB → **83% of expert bytes must
be resident/reused**. Available instantaneous residency (VRAM ~3.4GB +
RAM ~20GB) = **16% of the universe**. Even at N=1 (sequential):
3.45GB/token → 268ms H2D vs 305ms compute — already movement-bound.

**Consequence: TRUE-P2 T(N) = 0.297+0.00836N is a perfect-residency
upper bound, not an achievable P3 verifier time.** The real P3
throughput is movement-dominated unless the residency/reuse structure
changes the game. The gap between 16% available and 83% required
residency is the new central P3 problem — temporal locality
(within-prompt reuse 0.47-0.56) is the only lever visible in the data.

## Verified C256 single-layer bench (infrastructure works)

The full streaming machinery ran end-to-end (pinned pool, slot cache,
async H2D, pinning): C256 with CPU head = **zero movement, zero
stall**; block = static ref + CPU-head tax (bf16 head measured slower
than fp32 on this CPU: 594ms/919ms at N16/N32 vs 528ms fp32 N16 —
CPU-head at block widths costs 100-320ms, not the 43ms batch-1 census
figure). The infrastructure is correct and reusable; the single-pool
scenario it tested was not the real problem.

## Corrected open questions

1. Can conversation-adaptive residency (per-layer LRU over recent
   blocks + route-prediction prefetch) reach ~83% byte coverage? The
   data says 0.47-0.56 within-prompt reuse — promising but unproven at
   the required scale. Needs a real multi-block route corpus (the
   golden corpus must be re-captured or G1-extended).
2. Does CPU Dual-Rail (experts compute in RAM, skip H2D) beat H2D
   streaming at 4.3× movement/compute? CPU expert GEMM pricing is now
   first-order.
3. Width interaction: movement grows with N (union saturates per
   layer) but amortizes per token — the optimal N under real movement
   needs the joint model.
4. Proposer: the one-state family is closed; escalation (L38+L40
   fusion, recurrent, multi-state, tiny transformer) per §67/74 — now
   against the MOVEMENT-corrected economics, which raise E_req
   substantially.

## Artifacts

```text
stage1_race_result.json        A/B/C held-out results (family closed)
c192_stream_bench.json         C256 zero-movement single-pool bench
stage1_prompts.json / stage1_record.py / stage1_train.py
c192_stream_bench.py           real streaming infra (reusable)
G1-corpus route analysis       (in this report; golden_routes.json
                               marked CORRUPT — do not use)
```
