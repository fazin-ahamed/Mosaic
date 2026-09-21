# Session Report: FRONTIER-0 tier-1 screen (v4, boundary accounting corrected)

**Date:** 2026-09-20
**Status:** v4 fixes the last concrete accounting defect found in review: on
FULL acceptance the cycle emits `navail+1` tokens but v3 charged only `navail`
positions, leaving the bonus position's forward pass — its hidden state and KV
entries — uncharged. Boundary forwards are now explicitly charged, a
deterministic check proves the charged set is gap-free and cursor-consistent,
and the gate refuses to emit results if it fails. Corrected demand is ~7% higher
and the conclusions are reworded to match what the replay supports.

---

## 1. The defect, confirmed and quantified

Reproduced the reviewer's numbers exactly: v3 charged **235 positions → 60,630
accesses** (235×43×6) while the ledger emitted 255 tokens over 66 cycles with
**20 full-acceptance cycles**. Each full acceptance advanced the cursor
`navail+1` while charging `navail` sweeps, so 20 interior bonus positions were
never processed:

## 2. The narrow fix (no architecture change)

- `cur` semantics made explicit: the position whose forward pass is still owed.
- On **full acceptance** (`depth == navail`), a **boundary forward** on position
  `cur + navail` is charged as a single-position sweep appended to the same
  cycle. It uses the same cache and access ordering as any other forward.
- Deterministic `verify_boundary_accounting()` runs **before any result**, now
  for **every policy's schedule** (not eager only):
  - charged set is contiguous from 0 — **no interior gaps**,
  - each cycle's `cur` equals the previous charged end + 1 (cursor consistent),
  - explicit two-cycle full-acceptance inspection proving the boundary position
    is charged and the next `cur` is exactly boundary+1.
- The script raises `SystemExit` if any policy fails.

Qualification (from review): the guard is a **useful check**, not a proof of
total protection. It reconstructs expected positions by the same scheduling
rules as the demand builder, so an error shared between the two would pass it.
It cannot silently undercount *again* by the mechanism that caused the last
defect, but it does not certify the replay against every possible accounting
error.

## 3. Gate result (deterministic, runs first, all policies)

```
charged positions=255 max=254 contiguous=True cursor_ok=True two_cycle=True
all policies pass: True
```
Two-cycle full-acceptance inspection (identical for every policy, since the
ledger and position set are policy-independent): cycles 1 and 4 — boundary
positions 10 and 20 charged, next-cycle cursors 11 and 21 = boundary+1.

## 4. Corrected result (38 GB store, paired, per emitted token)

| policy | LRU miss | GB/tok | cold GB/tok | later GB/tok | Bel-mand | Bel-byp | LRU/opt | vs eager |
|---|---|---|---|---|---|---|---|---|
| eager6 | 17118 | **0.898** | 0.895 | 0.899 | 8512 | 8512 | 2.01× | 0% |
| frontier2_4 | 17371 | 0.911 | 0.882 | 0.927 | 8320 | 8319 | 2.09× | −1.5% |
| frontier1_2_3 | 17250 | 0.904 | 0.864 | 0.927 | 8242 | 8241 | 2.09× | −0.8% |
| sequential | 17284 | 0.906 | 0.872 | 0.925 | 8162 | 8161 | 2.12× | −1.0% |

Demand is **~7% higher** than v3 (eager 0.8975 vs v3's 0.834 GB/tok) because the
5,160 uncharged accesses are now included. **Eager is the best LRU policy**
(0.8975 GB/tok); all staged policies are slightly WORSE (−1.5% to −0.8%), since
charging the boundary forward removes the only free position staged policies
were deferring. Findings, with limited wording:

- **Staging: no useful traffic reduction demonstrated on this replay** — staged
  policies are slightly worse than eager. **No demonstrated runtime speedup.**
- **Residency: ~2× miss-count headroom within the current replay** (LRU/optimal
  2.0-2.1×). This is miss-count headroom, **not necessarily 2× end-to-end
  speed**, and it is **not the only possible software improvement**.
- **Belady (mandatory and bypass) is an optimal-cache figure for the admission
  models implemented**, not a universal lower bound over every legal design.
- **Bypass ≈ mandatory** again (≤1 miss difference) — bypass adds nothing here.

## 5. Conditional throughput (recalculated after the fix)

Core per emitted token: 17.5 ms/cycle ÷ 3.864 = 4.53 ms.

| cache | GB/tok | @0.55 | @1.05 | @1.82 | @13.0 GB/s | 30 tok/s needs |
|---|---|---|---|---|---|---|
| LRU best (eager) | 0.898 | 0.6 | 1.2 | 2.0 | 13.5 | 31.4 GB/s |
| Belady optimal | 0.432 | 1.3 | 2.4 | 4.1 | 26.5 | **15.0 GB/s** |

The **~15.0 GB/s** figure is a **conditional estimate** (measured demand under
the implemented admission models × measured core), not a measured hardware
requirement. It moves from v3's 14.3 to 15.0 because the boundary forwards are
now charged.

## 6. Decision (review accepted)

> **v4 is FROZEN as a proxy experiment.** The Frontier review item is closed.
>
> - **Frontier staging:** no useful traffic reduction demonstrated on this
>   replay. Stop investing in Frontier as the proposed breakthrough.
> - **Residency:** ~2× miss-count headroom vs the offline oracle within this
>   replay — not a promised runtime speedup.
> - **Exact-30:** unsupported on this hardware.
> - **Maximum achievable throughput: STILL UNMEASURED.** Neither the
>   conditional estimates here nor the 0.151 tok/s reference run establish it.
>
> Keep MosaicVM and its exactness work. **Pause the Exact-30 optimization
> campaign on this configuration.** A real speculative capture would improve
> confidence, but the current evidence gives no reason to expect it to close
> the large gap. Further work requires a **specific, independently testable
> mechanism for reducing the gap** before another expensive experiment is
> justified.

Artifacts: frontier0_screen.py (v4, frozen), frontier0_screen.json
(includes per-policy `boundary_check_by_policy` records).
