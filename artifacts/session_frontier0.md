# Session Report: FRONTIER-0 tier-1 screen (real routes, warm cache)

**Date:** 2026-09-20
**Status:** Executed the first, cheapest tier of FRONTIER-0 that existing
artifacts allow: staged-vs-eager verification demand on the REAL persisted route
trace against an EXPLICIT warm LRU cache (no compulsory-miss subtraction), plus
a measured async storage queue-depth sweep. Result: Mosaic Frontier's staged
verification reduces cold demand **less** on real routes than the synthetic
model predicted, and the honest warm-cache traffic is **higher** than the
capacity-only estimate. The stop decision stands; the accounting is now correct.

---

## 1. What was run (and its evidence class)

- **SIMULATED (real routes + warm LRU + measured acceptance):**
  `frontier0_screen.py` replays the 256-position captured trace
  (`ek0c_routes.pt`), draws acceptance from the measured E[A]=3.757
  (truncated-geometric, W≤6), and compares eager-6 / staged / sequential
  verification cold demand against an evolving LRU cache at 30/35/38 GB.
- **MEASURED (this SSD):** async read queue-depth sweep, expert-size (13.37 MB)
  random reads with `fadvise DONTNEED`.
- **CONDITIONAL:** tok/s from warm-cache GB/cycle × measured storage range.

## 2. Real-route warm-cache demand (the key correction)

At a 38 GB store, warm LRU, cold experts/cycle (INCLUDING first touches +
evictions — no compulsory subtraction):

| policy | cold/cyc | GB/cyc | reduction vs eager |
|---|---|---|---|
| eager6 | 361.8 | 4.84 | 0% |
| frontier2_4 | 333.8 | 4.46 | 7.7% |
| frontier1_2_3 | 314.0 | 4.20 | 13.2% |
| sequential | 275.9 | 3.69 | 23.7% |

**Two findings:**
1. On REAL routes the staged-verification demand reduction (8-24%) is
   **smaller** than the synthetic experiment's (21-35%). Real routes have less
   exploitable stopping structure than the synthetic random model assumed.
2. The honest warm-cache traffic (3.7-4.8 GB/cycle) is **much higher** than the
   earlier capacity-only figure (0.35 GB/cycle). That earlier number subtracted
   compulsory misses; a real warm run with eviction does not get to.

**Caveat (optimistic bias):** window rows reuse consecutive accepted-path
routes as a proxy for speculative candidates. Real rejected candidates route to
DIFFERENT experts and would raise cold demand further. So these numbers are a
lower bound on real speculative demand.

## 3. Measured async storage queue-depth sweep

Expert-size (13.37 MB) random reads, cache-defeated:

| queue depth | aggregate GB/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|
| 1 | 1.05 | 11.6 | 18.3 | 22.6 |
| 4 | 0.83 | 28.7 | 212.9 | 437.6 |
| 16 | 1.82 | 104.3 | 152.1 | 225.6 |

Storage path delivers ~**0.5-1.8 GB/s** depending on pattern/depth, tail-latency
heavy at moderate queue depths. Coalesced 64 MB reads earlier hit 1.61 GB/s.
Defensible range for conditional bounds: **0.55-1.82 GB/s**. Not a proven
hardware maximum, but the observed envelope on this SSD.

## 4. CONDITIONAL tok/s (warm demand × measured storage, I/O + core only)

38 GB store, best policy (sequential, 3.69 GB/cycle):

| storage | cycle | tok/s |
|---|---|---|
| 0.55 GB/s | 6723 ms | 0.6 |
| 1.05 GB/s | 3530 ms | 1.1 |
| 1.82 GB/s | 2044 ms | 1.8 |

**For 30 tok/s at 38 GB warm sequential, storage must reach ~34 GB/s** — before
any expert compute. That is ~19-62× the measured range and above PCIe Gen3 x16
H2D (12.84 GB/s) as well. Frontier's 24% demand cut does not change the order of
magnitude.

## 5. What FRONTIER-0 establishes about Mosaic Frontier

- **Direction is sound, magnitude is insufficient.** Prioritizing the earliest
  token and limiting speculative cold reads genuinely reduces demand, but on
  real routes only ~8-24%. The gap to 30 tok/s is 19-62× in storage terms.
- **Early rejection alone cannot supply the missing factor** — confirms the
  proposal's own conclusion, now on real routes rather than synthetic.
- Frontier would need to combine with a **materially faster storage tier**
  (34 GB/s class — not this NVMe, not PCIe Gen3) to matter. No such tier exists
  on this box.

## 6. What still cannot be measured without NEW capture

- **Real speculative candidate routes with rejected branches.** The current
  trace is the accepted autoregressive path only. A true FRONTIER-0 needs the
  frozen proposer + source verifier wired to emit W-wide candidate routes and
  their per-position acceptance, so rejected-branch demand is real (this screen
  is optimistic without it). This is the one substantial new artifact required.
- **Exact packed-VNNI kernel throughput** (CPU tier) — still unbuilt; only a
  bf16→fp32 reference upper bound exists.
- **Integrated end-to-end runtime** — not built; achieved throughput remains the
  0.151 tok/s reference capture.

## 7. Recommendation

The bounded experiment was worth running and is now answered at tier 1:
**Mosaic Frontier does not close Exact-30 on this hardware, and the honest
warm-cache accounting is worse than the prior capacity-only estimate.** The
storage requirement (~34 GB/s) exceeds even PCIe H2D.

Two honest options:
1. **Stop the Exact-30 push here** (recommended). The evidence across GPU-stream,
   residency-oracle, compression, CPU-hybrid, and now real-route warm-cache +
   Frontier scheduling consistently shows a movement wall 1-2 orders of
   magnitude from 30 tok/s. The reproducible measured baseline is 0.151 tok/s.
2. **If one more experiment is authorized:** capture real speculative candidate
   routes (proposer + verifier) so the rejected-branch demand is real, and
   confirm whether it changes the order of magnitude. It will not lower demand
   below the accepted-path proxy — it can only raise it — so this mainly tightens
   the negative result rather than opening a path.

Proposer frozen, BLIND sealed, exactness intact. No new VNNI / grouped-GEMM /
scheduler development is justified by this evidence.

Artifacts: frontier0_screen.json, h0a_compressed_replay.json,
h0a_longtrace.json, ek0c_routes.pt, session_h0_final.md.
