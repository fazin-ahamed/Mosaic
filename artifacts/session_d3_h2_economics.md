# Session Report: H2/P3 Cycle Economics — first assembly from measured parts

**Date:** 2026-09-20
**Status:** Proposer frozen (ps1001+d3-B4, fresh E[A]=3.757). Pivoted to H2.
Assembled a first-order cycle estimate from EXISTING measured artifacts and
identified the one blocking unmeasured quantity.

---

## 1. Budget

Exact-30 at E[A]=3.757 needs **T_cycle <= 3.757/30 = 125.2 ms**.

A cycle = draft a short window (W ~ 5-6 to accept 3.757) + exact-verify it
through the full 43-layer source model + emit accepted prefix.

## 2. Measured components (on disk, real GPU)

**Core (43 layers, experts EXCLUDED), `core_timing.json`, idle, thermal UNKNOWN:**
- per position: **2.92 ms** (K32 window 93.5ms / 32)
- verify a W-position window (core only): W=5 -> 14.6ms, W=6 -> 17.5ms

So the non-expert core of a real speculative cycle is cheap: **~15-18 ms**.
Well inside budget. The core is NOT the problem.

**Expert tail — this is where it breaks.** `d2_complete_tail.json` (L41/L42
hybrid, gpu_serial, captured real routes, includes prefill):
- K=1: 581ms, K=8: 1063ms, K=16: 1674ms, K=32: 2740ms
- marginal ~66 ms/position for **just the L41/L42 two-layer tail**

## 3. The blocking finding

Two things make the current numbers unusable as a production cycle wall:

1. **Coverage:** `d2_complete_tail` measured only the L41/L42 tail (2 layers).
   The source model routes experts in ~58 MoE layers, not 2. There is no
   measured production expert cost for the full stack per position.
2. **Provenance (per AGENTS.md contract):** the D2 tail uses the source
   **torch-shim** reference implementation — explicitly "slow reference
   implementations, NOT production D2/P4 benchmarks." `d2_readiness.md`
   already states: "actual runtime footprint and production kernel latency
   remain open."

So the memo's clean "if H2 lands at 118ms" scenario cannot be evaluated: the
**production expert-kernel latency is genuinely unmeasured**, and the only
expert timings on disk are (a) 2-layer-only and (b) reference-shim, which the
project's own exactness/《evidence contract forbids promoting to production wall.

## 4. Honest cycle estimate (bounded)

- Core (real): ~15-18 ms/cycle for a W~5-6 window. Not the bottleneck.
- Experts (full stack, production kernel): UNMEASURED. Lower bound from
  reference shim on 2 layers alone (~66ms marginal/pos) already blows the
  125ms budget many times over IF the shim reflected production — but it does
  not; a real fused/quantized expert kernel with residency could be far lower.

**Verdict: the cycle wall is dominated by expert execution, which has never
been measured with a production kernel across the full layer stack. That
measurement — not proposer E[A] — is the true remaining unknown for Exact-30.**

## 5. What this means for the E[A] debate

The entire BAPM->AAL->OFDM->Hydra->ExposureBank chase over +0.243 E[A] was
optimizing a term (proposal/acceptance) whose sibling term (verification cost)
is unmeasured and expert-dominated. Whether E[A]=3.757 or 4.0 or 4.5 is
required is **undefined until a production expert kernel latency exists**,
because required E[A] = 30 * T_cycle and T_cycle is set by expert execution.

The operator's pivot is correct and the priority is now unambiguous.

## 6. The real NEXT (blocking, and it is a build not a tweak)

A production expert-execution path is required — the thing D2-readiness flagged
as open. Concretely:
- A fused/quantized GPU expert kernel (source FP4/FP8) OR the VNNI CPU path,
  measured per position across the real routed layer set, with the three-tier
  residency (VRAM hot / prepared-RAM LRU / packed NVMe) actually populated.
- Then B1/B2/B4 timed cycles -> R_B = E[A]_B / T_B -> production B*.

This is substantial systems work (the P3/H2 track), not another eval. It is
also the only path that can answer whether the proposer is already good enough.

## 7. State

```text
proposer                 FROZEN ps1001+d3-B4, fresh E[A] 3.757
core verify cost         MEASURED ~2.92ms/pos (43L, experts excl) — cheap
expert tail cost         UNMEASURED at production (shim=slow, 2-layer only)
T_cycle                  UNDETERMINED -> required E[A] UNDETERMINED
Exact-30 feasibility     BLOCKED on production expert kernel latency
BLIND                    SEALED
ExposureBank             CONDITIONAL (needs T_cycle to justify)
engineering              proposer ~done; system/verifier track is the frontier
```

## 8. Bottom line

The proposer is not the bottleneck and never provably was — the expert
verifier kernel is. Every measured component says core is cheap (~3ms/pos) and
experts are the entire cost, but no production expert kernel has been timed
across the full stack. That build is the true P3 gate.
