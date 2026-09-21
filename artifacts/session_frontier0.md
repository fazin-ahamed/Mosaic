# Session Report: FRONTIER-0 tier-1 screen (v2, corrected)

**Date:** 2026-09-20
**Status:** The v1 screen had five methodology bugs (flagged in code review) that
inflated Mosaic Frontier's apparent benefit. The corrected v2 screen — shared
acceptance ledger, chronological all-loads cache, per-emitted-token accounting,
and an offline Belady lower bound — shows **staged verification gives ~0% demand
reduction on real routes**, and that the real lever is the residency policy
(LRU is ~2× worse than optimal), not staging. Even optimal caching plus the best
plausible storage on this box falls ~7× short of 30 tok/s. This is strong
negative evidence for Frontier specifically; it is NOT a universal impossibility
proof.

---

## 1. Bugs fixed vs v1

1. **Shared acceptance ledger.** v1 advanced one RNG across policies, so eager
   ran 80 cycles, staged 60, sequential 71 — not a paired comparison. v2 draws
   one acceptance sequence and replays it identically across all policies
   (51 cycles, 257 emitted tokens, for every policy).
2. **Per-emitted-token reporting.** v1's headline GB/cycle was a whole-run
   average mislabelled "warm." v2 reports **GB per emitted token** with cold
   (first third) and later intervals separated explicitly.
3. **Chronological all-loads cache.** v1 counted misses at window start and only
   committed accepted rows — speculative loads never occupied capacity or
   evicted. v2 replays a layer-major access stream (for layer: for row: for
   expert) through one LRU; EVERY load counts and can evict.
4. **No unproven lower-bound claim.** Window rows still use accepted-path routes
   as a proxy for speculative candidates, but the demand DIRECTION vs real
   candidates is **UNKNOWN** — a rejected candidate may select resident OR
   colder experts. The prior "new capture can only worsen the result" claim is
   **withdrawn**.
5. **Belady lower bound.** v1 implied LRU demand was the floor. v2 computes the
   offline optimal (Belady) miss count on the same request stream, exposing the
   LRU-to-optimal gap.

## 2. Corrected result (38 GB store, paired, per emitted token)

| policy | LRU miss | GB/tok | cold GB/tok | later GB/tok | Belady miss | LRU/Belady | vs eager |
|---|---|---|---|---|---|---|---|
| eager6 | 15917 | 0.828 | 0.793 | 0.964 | 8002 | 1.99× | 0% |
| frontier2_4 | 16000 | 0.832 | 0.767 | 0.962 | 7977 | 2.01× | −0.5% |
| frontier1_2_3 | 15988 | 0.832 | 0.768 | 0.963 | 7941 | 2.01× | −0.4% |
| sequential | 15848 | 0.824 | 0.764 | 0.956 | 7887 | 2.01× | +0.4% |

**Two corrected findings:**
1. **Staged verification (Mosaic Frontier's core mechanism) gives ~0% demand
   reduction** (−0.5% to +0.4%) under correct paired chronological accounting.
   v1's 8-24% was an artifact of the bugs. Deferring speculative rows does not
   eliminate their loads — the experts are needed anyway as the prefix advances,
   and within-layer reuse already shares them.
2. **LRU is ~2.0× the Belady optimal.** A better residency policy — not staging
   — is the real lever, with ~2× headroom.

## 3. Conditional throughput (corrected, per emitted token)

Core per emitted token: 17.5 ms/cycle ÷ 5.04 emitted/cycle = 3.47 ms.

| policy / cache | GB/tok | @0.55 | @1.05 | @1.82 | @13.0 GB/s |
|---|---|---|---|---|---|
| LRU sequential | 0.824 | 0.7 | 1.3 | 2.2 | 15.0 tok/s |
| Belady optimal | 0.410 | 1.3 | 2.5 | 4.4 | 28.5 tok/s |

- At the **measured storage envelope (0.55-1.82 GB/s)**, even optimal caching
  yields **1.3-4.4 tok/s**.
- **For 30 tok/s at Belady optimal, storage must reach ~13.7 GB/s** — above this
  NVMe (0.55-1.82) and essentially at PCIe Gen3 x16 H2D (12.84 GB/s). No such
  expert-storage tier exists on this box.

## 4. What this does and does not establish

- **Establishes:** Mosaic Frontier's staged-verification mechanism does not
  reduce demand on real routes; the residency policy has ~2× headroom to
  optimal; and even optimal caching plus best plausible storage is ~7× short of
  30 at the measured storage speeds.
- **Does NOT establish:** a universal impossibility. Belady@13 GB/s reaches
  28.5 tok/s — so if a ≥13.7 GB/s exact expert-storage tier existed AND
  near-optimal caching were realized AND the accepted-path proxy holds, 30 is
  arithmetically conceivable. None of those three holds on this hardware today.
- **LRU is not a lower bound.** The Belady comparison is the correct floor for a
  fixed request stream; it is still ~7× short at measured storage.

## 5. Open items (unchanged by this correction)

- **Real speculative candidate routes with rejected branches** — the trace is
  accepted-path only; direction of the proxy error is unknown, so a real capture
  could move demand either way. This is the one artifact that would remove the
  proxy caveat.
- **Exact packed-VNNI kernel** — unbuilt; only a bf16→fp32 reference exists.
- **Near-optimal online residency policy** — the 2× LRU/Belady gap is real
  headroom, but closing it plus a 13.7 GB/s tier is what 30 would require.

## 6. Recommendation

The corrected screen makes the case for Mosaic Frontier's *scheduling* mechanism
**weaker**, not stronger: staging is ~neutral. The residency policy is the only
software lever with headroom (2×), and even the offline optimal needs ~13.7 GB/s
storage — hardware this box lacks. Combined with the earlier rails, no measured
or optimal-cache configuration reaches 30 tok/s here.

Defensible status, restated exactly:
> 30 tok/s is unsupported on this hardware; a universal maximum is NOT
> established. The measured achieved baseline is 0.151 tok/s (reference capture).
> Conditional bounds (measured demand × measured/optimal cache × measured
> storage) span ~0.7-4.4 tok/s; reaching 30 requires ~13.7 GB/s exact expert
> storage plus near-optimal caching, none available on this box.

No new VNNI / grouped-GEMM / scheduler build is justified. The one experiment
that would add information is a real speculative-candidate capture — worth it
only if the residency-policy headroom is separately shown to matter.

Artifacts: frontier0_screen.py (v2), frontier0_screen.json,
h0a_compressed_replay.json, h0a_longtrace.json, ek0c_routes.pt.
