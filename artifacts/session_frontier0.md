# Session Report: FRONTIER-0 tier-1 screen (v3, canonical acceptance)

**Date:** 2026-09-20
**Status:** v3 reconciles the screen to the CANONICAL acceptance definition
(`d1b_tail_robustness.prefix_metrics`) and fixes the four v2 defects found in
review. Substantive result is unchanged and now defensible on the accepted-path
proxy: **staged verification (Mosaic Frontier's mechanism) gives ~0% demand
reduction** (0.1-0.5%), LRU is ~2.0× the offline optimal, bypass caching adds
nothing here, and even optimal caching needs **~14.3 GB/s** exact expert storage
for 30 tok/s. The one open caveat is the accepted-path proxy, whose error
direction is unknown.

---

## 1. Acceptance semantics reconciled (finding 1)

Read the canonical evaluator, not a new definition:
`prefix_metrics` sets `depth` = consecutive accepted draft decisions (**0..W**,
excluding the exact root, stop at first miss) and **`E_A = depth + 1`**. The
project mean **3.757 is E_A**, so mean `depth` = 2.757. v2 wrongly fit accepted
drafts to 3.757 and floored `depth ≥ 1` (excluding immediate rejection).

v3 draws `depth` (iid hit q=0.781, may be 0), emits `depth+1`, advances the
cursor `depth+1`. Realized draw over the 256-position trace: 66 cycles, mean
depth 2.864, **mean E_A 3.864** (finite-sample, slightly above the 3.757 target;
honest realized value). The **+1 bonus is a free emission** (argmax of the last
verified position's output — no extra forward pass), so full-accept cycles
advancing navail+1 while accessing navail positions is correct, not a skipped
forward. 20 full-accept cycles; committed positions reconcile to emitted tokens.

## 2. Real staged access order (finding 2)

Each verification STAGE is now its own layer-major forward sweep over its
positions; stages run in sequence:
- eager = one sweep over all W positions (depth unknown until verified),
- frontier1_2_3 = up to three sweeps (1, then 2, then 3 positions),
- sequential = up to W single-position sweeps.

The cache persists across stages, so a staged policy re-traverses layers per
stage rather than getting v2's advance-knowledge single batched order.

## 3. Continuous cache; interval binning (finding 3)

ONE LRU runs over the whole stream; per-cycle misses are binned during that
single simulation. Cold (first third) and later intervals are slices of the
continuous run, not cold restarts.

## 4. Two oracles (finding 4)

On the same stream: mandatory-admission Belady AND bypass-capable Belady (may
serve a miss from a staging buffer without admitting). Bypass is **not** claimed
as a universal floor and its transient staging-buffer cost is noted.

## 5. Result (38 GB store, paired, per emitted token)

| policy | LRU miss | GB/tok | cold GB/tok | later GB/tok | Bel-mand | Bel-bypass | LRU/opt | vs eager |
|---|---|---|---|---|---|---|---|---|
| eager6 | 15905 | 0.834 | 0.796 | 0.855 | 8103 | 8103 | 1.96× | 0% |
| frontier2_4 | 15833 | 0.830 | 0.755 | 0.872 | 7888 | 7887 | 2.01× | 0.5% |
| frontier1_2_3 | 15822 | 0.830 | 0.749 | 0.874 | 7878 | 7877 | 2.01× | 0.5% |
| sequential | 15897 | 0.834 | 0.760 | 0.874 | 7844 | 7843 | 2.03× | 0.1% |

**Findings:**
1. **Staging gives ~0% reduction (0.1-0.5%).** Mechanistic reason: the per-cycle
   speculative over-fetch that staging avoids is a small fraction of total
   demand, which is dominated by the committed-path working-set churn (89 GB
   footprint ≫ 38 GB cache) that every policy must load. Deferring speculative
   rows defers, it does not remove.
2. **LRU is ~2.0× the offline optimal.** Residency policy is the only software
   lever with headroom.
3. **Bypass caching ≈ mandatory admission here** (differ by ≤1 miss). Bypass
   does not help on this stream, so the ~2× LRU/optimal gap is the real headroom.

## 6. Conditional throughput (per emitted token; core 17.5 ms/cycle ÷ 3.864 = 4.53 ms/tok)

| cache | GB/tok | @0.55 | @1.05 | @1.82 | @13.0 GB/s | 30 tok/s needs |
|---|---|---|---|---|---|---|
| LRU sequential | 0.834 | 0.7 | 1.3 | 2.2 | 14.6 | 28.9 GB/s |
| Belady optimal | 0.411 | 1.3 | 2.5 | 4.3 | 27.7 | **14.3 GB/s** |

At the measured storage envelope (0.55-1.82 GB/s), even optimal caching yields
1.3-4.3 tok/s. For 30 at optimal, storage must reach ~14.3 GB/s — above this
NVMe and above PCIe Gen3 x16 H2D (12.84 GB/s). No such expert-storage tier
exists on this box.

## 7. What v3 establishes vs does not

- **Establishes (on the accepted-path proxy, canonical acceptance, real staged
  order):** Mosaic Frontier's staged-verification mechanism does not reduce
  demand; the residency policy has ~2× headroom to optimal; bypass adds nothing;
  even optimal caching needs ~14.3 GB/s storage for 30.
- **Does NOT establish:** a universal impossibility. Belady@13 GB/s reaches
  27.7 tok/s; if a ≥14.3 GB/s exact expert-storage tier existed AND near-optimal
  online caching were realized AND the proxy holds, 30 is arithmetically
  conceivable. None holds on this hardware today.
- **Open caveat (unchanged):** window rows use accepted-path routes as a
  speculative-candidate proxy; the demand-direction error vs real candidates is
  **unknown** (withdrawn any "capture can only worsen" claim). A real
  speculative capture is the only artifact that removes this caveat.

## 8. Recommendation

Consistent with the review: **pause new architecture and expensive captures;
this validated replay is the deliverable.** It now meets the stated invariants —
explicit root/draft/bonus/next-cycle accounting reconciled to `prefix_metrics`,
every committed source position accounted for (free bonus noted), real
stage-by-stage access order, one shared ledger, persistent cache across
intervals, and both mandatory-admission and bypass-capable oracles.

Defensible status, restated exactly:
> Mosaic Frontier has no demonstrated useful speedup (staging ~0%). 30 tok/s is
> unsupported on this hardware; a universal maximum is NOT established. Measured
> achieved baseline 0.151 tok/s (reference capture). Conditional bounds
> (measured demand × LRU-or-optimal cache × measured storage) span ~0.7-4.3
> tok/s; reaching 30 needs ~14.3 GB/s exact expert storage plus near-optimal
> caching, neither available here. The residency-policy 2× gap is the only
> software headroom; the accepted-path proxy is the only open measurement.

Artifacts: frontier0_screen.py (v3), frontier0_screen.json.
