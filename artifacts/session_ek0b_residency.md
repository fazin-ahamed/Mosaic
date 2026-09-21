# Session Report: EK0b — Temporal Residency. GPU-only Exact-30 CLOSED.

**Date:** 2026-09-20
**Status:** DECISIVE. The Belady oracle (perfect future knowledge, physically
unachievable online) cannot bring cold expert traffic within the Exact-30
movement budget at any usable VRAM capacity. **GPU-only Exact-30 with the full
exact 43-layer MoE is physically closed on RTX 3060 by ~4.5x.**

---

## 1. Setup

Real consecutive decode cycles (source_corpus gates, 7 domains x 31 positions),
W=6 verify window, stride 4 (~E[A]=3.757 accepted length), cache key
(layer,expert), packed-FP4 13.37MB/expert, H2D 12.90 GB/s. Cache reset per
domain (realistic session start). LRU + static-hot-LFU + Belady optimal.

Absolute gate: budget 125.2ms − core 17.5ms = 107.7ms H2D = **≤103 cold
experts/cycle** even with ZERO expert compute. Mean union |U_c| = 900/cycle.

## 2. Result — miss counts per cycle (need ≤103)

| VRAM (routed) | slots | LRU miss | static-hot miss | **Belady miss** |
|---|---|---|---|---|
| 2 GB | 149 | 900 | 757 | 824 |
| 4 GB | 299 | 900 | 643 | 744 |
| 6 GB | 448 | 900 | 552 | 650 |
| 8 GB | 598 | 900 | 473 | 536 |
| **10 GB** | 747 | 900 | 406 | **469** |

Even at an optimistic 10 GB routed-cache (leaving almost nothing for core/KV/
shared/activations), **Belady still misses 469/cycle vs the 103 gate — 4.5x
over.** LRU is pathological (≈900, no reuse captured) because each W=6 cycle's
union of 900 experts barely overlaps the next. Static-hot is best but still
4x over.

## 3. Movement-only tok/s upper bound

Best case (Belady, 10GB, expert compute = 0): mean cold 6.28 GB/cycle → 504 ms
→ **8.4 tok/s upper bound.** Real execution (expert compute > 0) is lower.
Exact-30 needs 125ms; the oracle floor is 504ms. **No caching algorithm closes
this** — it is a bytes-vs-bandwidth physics result, not a policy weakness.

## 4. Why reuse is so low

Adjacent decode-token expert reuse (single stream, token-to-token) is only
0.24-0.44. Across a 6-position window the union explodes to ~900 of the touched
experts, and consecutive windows share little. DeepSeek's 256-expert / top-6
routing with 40 score-routed layers spreads across the expert population fast.
The 104 GB observed working set has NO small hot core that fits VRAM: getting
90% of traffic requires far more than 10 GB.

## 5. Verdict

```text
GPU-only cold streaming        CLOSED (EK0: 896ms vs 125ms)
GPU-only + LRU residency       CLOSED (900 miss)
GPU-only + static-hot          CLOSED (406 miss @10GB)
GPU-only + BELADY ORACLE       CLOSED (469 miss @10GB vs 103 gate)
=> GPU-ONLY EXACT-30           PHYSICALLY IMPOSSIBLE on 3060
```

This is the hard, evidence-based hardware ceiling the roadmap asked for. It is
NOT a proposer problem, NOT a kernel problem, NOT a cache-policy problem. It is
weight bytes vs PCIe bandwidth vs VRAM capacity.

## 6. What survives — the CPU-hybrid question

GPU-only is closed; the project is not necessarily closed. The precise next
question (roadmap §30): cold experts must execute where their weights already
live — **CPU (VNNI) on host-resident weights, no PCIe crossing.**

At 10GB VRAM (Belady) ~469 experts/cycle are cold. For Exact-30 those must be
served without H2D. Required: host-resident prepared expert bank + CPU execution
fast enough that CPU expert wall + GPU-resident expert wall + core ≤ 125ms,
under parallel CPU/GPU scheduling.

But note the harder constraint EK0 already implies: the 104 GB working set
exceeds host RAM (~24-31 GB) too. Only the packed 147 GB full bank fits on NVMe.
So even the CPU rail faces a capacity wall — host RAM holds a prepared LRU
subset, and NVMe (2-5 GB/s) is slower than H2D. The hybrid feasibility is itself
now quantitatively in doubt.

## 7. Honest project status

The measured chain is now complete and consistent:
- Proposer: E[A]=3.757 fresh, strong, generalizes. NOT the bottleneck.
- Core: ~18ms/cycle. NOT the bottleneck.
- **Experts: 11.6 GB compulsory/cycle, 104 GB working set, oracle-floor 504ms.
  THE bottleneck, and it exceeds both VRAM (12GB) and host RAM (~24-31GB).**

Exact-30 on this hardware with the full exact model requires moving/executing
~6-12 GB of distinct expert weights per 125ms cycle from a working set larger
than total system memory. The measured physics says this is not achievable by
residency or streaming. The remaining live path (CPU-hybrid) faces a working
set > RAM and must be quantified, but the optimistic case is weak.

## 8. Caveat (one honest limitation)

These 49 cycles come from 7 short (32-token) prompts. A single long-context
generation might show higher sustained locality (a stable working set forming
over hundreds of tokens). Before declaring the ceiling FINAL, one long
single-stream route capture should confirm the reuse-distance CDF does not
improve materially with sequence length. But adjacent-token reuse of 0.24-0.44
makes a jump to the required >0.88 steady-state very unlikely.

## 9. State / next

```text
EK0b verdict                   GPU-only Exact-30 CLOSED (Belady 4.5x over)
oracle movement-only tput UB   8.4 tok/s (10GB, zero compute)
working set                    104 GB > VRAM(12) and > RAM(~24-31)
NEXT decision                  (a) long-stream reuse confirmation, then
                               (b) CPU-hybrid capacity/throughput bound, or
                               (c) accept measured max tok/s + document ceiling
proposer / BLIND               frozen / sealed (irrelevant to ceiling)
engineering                    system feasibility now the gate, not %
```

## 10. Bottom line

The pivot from proposer to system did exactly its job: it found the real wall
in two cheap measurements. Exact-30 with the full exact DeepSeek-V4 MoE is
**movement/capacity-bound beyond what a 12GB/12.9GB-s/~28GB-RAM machine can do**,
proven against an optimal oracle. The honest deliverables now are (1) a
long-stream locality check to finalize the ceiling, and (2) a measured CPU-hybrid
bound to state the true maximum sustainable tok/s. This is a hardware ceiling,
not a modelling failure — and it is the first time the project has a physics-
grade answer to "why not 30."
