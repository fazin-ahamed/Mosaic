# Session Report: EK0 — Route/M Census + Movement Roofline

**Date:** 2026-09-20
**Status:** EK0 COMPLETE. The two decisive numbers reframe the whole Exact-30
problem. Verdict: **GPU expert streaming is physically impossible for Exact-30;
residency is mandatory and the working set does not fit VRAM.**

Inputs: real per-layer top-6 routes from `artifacts/source_corpus/*.pt` (7
domains x 32 decode positions), 43 MoE blocks, 256 routed + 1 shared/layer,
packed-FP4 = 13.37 MB/routed-expert, measured H2D 12.90 GB/s.

---

## 1. P(M) — batching opportunity (weak)

Rows-per-expert per layer over a W=6 verify window:

```
M=1  0.647    M=2  0.178   M=3  0.075
M=4  0.040    M=5  0.028   M=6  0.031     mean M = 1.72
```

**65% of expert invocations are M=1** — single-row. Batching (IMMA large-M)
helps a minority; the small-M/DP4A path dominates. Even across 6 verified
positions, experts rarely coincide. This kills any "big GEMM batching saves us"
hope: the workload is fundamentally many tiny expert calls.

## 2. Compulsory movement roofline — the killer

Per verify cycle (W=6 window), the union of distinct routed experts touched:
- **865 expert-slots/cycle** (~20 of 256 per layer x 43 layers)
- packed-FP4 bytes: **11.56 GB/cycle** if cold
- **T_move_min = 11.56 GB / 12.90 GB/s = 896 ms/cycle**

Budget is **125 ms**. Cold GPU streaming is **7.2x over budget** on movement
alone, before any compute. B4 vs B1 is identical here (branches reuse the same
per-layer expert union — the earlier "21 nodes" fear was unfounded; branching
adds ~0 movement).

**Conclusion: streaming experts H2D per cycle cannot reach Exact-30. Not close.**

## 3. Residency — the harder wall

- Full routed bank: 43 x 256 x 13.37MB = **147 GB** (vs 12 GB VRAM).
- Observed working set (7 domains, 32 pos): 7772 slots = **104 GB packed**.

So the experts actually used across realistic traffic are **104 GB** — ~8.7x
VRAM. You cannot pin the working set. Even a perfect LRU faces a working set an
order of magnitude larger than VRAM. This is the real Exact-30 obstacle, and it
was invisible until the route union was measured.

## 4. What this means

The non-expert core is 2.92ms/pos (~18ms/cycle) — trivial. The entire Exact-30
problem is: **move/execute ~11.6 GB of distinct expert weights per cycle from a
104 GB working set through a 12 GB VRAM / 12.9 GB/s pipe in 125 ms.**

Movement floor alone (896ms) already forbids it on the GPU-stream model. The
only physics that can close a 7.2x movement gap:

1. **Massive cross-cycle reuse** — but the working set is 104GB and each cycle
   touches 11.6GB fairly spread; temporal locality across cycles is the ONLY
   lever and must be measured (not yet done).
2. **CPU-side execution** of cold experts (no H2D) — the VNNI rail. If experts
   live in host RAM (104GB fits in ~24-31GB? NO — 104GB > host RAM too, only
   the packed 147GB full bank on NVMe). Host RAM holds a prepared LRU subset.
3. **NVMe streaming** at ~2-5 GB/s is even slower than H2D — worse, not better.

## 5. Hard conclusion — reframed project status

The measured route union says Exact-30 with the full exact 43-layer MoE at
E[A]=3.757 (or any E[A]) is **movement-bound by ~7x on GPU and the working set
is ~8.7x VRAM / likely > host RAM**. No proposer improvement (the entire
BAPM/AAL/OFDM/Hydra/ExposureBank axis) affects this by even 1%: E[A] changes the
budget linearly (125ms -> at E[A]=6, 200ms) but movement is 896ms.

To reach 125ms you need the per-cycle compulsory cold bytes down from 11.6GB to
~1.6GB — a **7x reduction via cross-cycle residency reuse**, which requires the
inter-cycle working set to be far smaller than the 104GB aggregate. That is the
single measurement that now decides feasibility:

**EK0b (next): cross-cycle expert reuse. Over a real multi-cycle generation, how
many of each cycle's 865 experts were already resident from recent cycles?**
If reuse is high (few hot experts dominate), a VRAM+RAM LRU makes it feasible.
If reuse is low (routes spread), Exact-30 with the full exact model on this
hardware is likely physically out of reach and the honest deliverable is a
measured max tok/s < 30 with a documented memory-bandwidth floor.

## 6. State

```text
EK0 P(M)                    mean 1.72, 65% M=1 (batching weak)
EK0 union                   865 experts/cycle, 11.56 GB compulsory
EK0 roofline                T_move_min 896ms vs 125ms budget = 7.2x over
EK0 residency               working set 104GB vs 12GB VRAM = 8.7x over
branching movement cost     ~0 (B4==B1 union) — d3 branch is nearly free
core cost                   ~18ms/cycle (negligible)
NEXT                        EK0b cross-cycle reuse (decides feasibility)
proposer                    FROZEN, irrelevant to the movement floor
BLIND                       SEALED
```

## 7. Bottom line

EK0 delivered exactly what the pivot promised: the real bottleneck is measured
and it is **expert weight movement / residency, not the proposer and not
compute.** The gap is 7x on movement and 8.7x on VRAM working set. Whether
Exact-30 is reachable now hinges entirely on **cross-cycle expert reuse** (EK0b)
— if a small hot set dominates real generation, residency closes it; if not,
the exact model on a 12GB/12.9GB-s box has a hard sub-30 tok/s ceiling that no
proposer work can lift. That is the next measurement, and it is cheap (reuses
the same route data over consecutive cycles).
