# Session Report — All Four Tracks (E/S/D/M) + G2 v1 Freeze

**Date:** 2026-09-04
**Roadmap:** TRUE P2 Complete + G2 Calibrated — Strong-Proposer Phase

## Roadmap corrections applied

1. **E_req relabeled** (`true_p2_sweep.json`, 192dada2): mathematical
   verifier-only from the fit — N16 = **13.72**, N32 = **18.29**
   (30×T_fit). The 14.4/20.3 values (30×T_raw) are labeled
   raw-measurement operating points; the 15-20 band is the
   practical/headroom-adjusted target.
2. **Swap floor classified** (in artifact): 0.060s = internal resident
   fp8→bf16 weight decode into the double-buffered workspace
   (GPU-resident codes → GPU-resident workspace). **No host/device
   expert movement in the timed region → TRUE_P2 classification
   remains valid.** 128-expert bank note documented (per-expert timing
   identical; full 256-bank cannot co-fit with fp32 HC intermediates).

## Track E — τ_h(s) frozen, G2 v1 spec written

The Blackwell sandbox died mid-session; all artifacts were already
mirrored (nothing lost). Full-horizon pair rebuilt locally-available
data: **G1 Blackwell corpus vs a fresh 3060-kit 32-token run on the
same prompts** (both kit, neither MosaicVM; `g2_calib_3060_cert32/`).

- Kit-vs-kit prefixes at 32 tokens: 32, 2, 5, 2, 9, 32, 27 — coding
  and math fully agree cross-hardware; mean 15.6/32 (MosaicVM's own:
  17.0/32 — statistically the same population).
- **τ_h(s)** (`tau_h_prefix.json`, 7e194e6a): p50 flat 0.10 at ALL
  steps; monotone p99 envelope 0.24 (s 0-3) → 0.43 (4-15) → 0.49
  (16-23) → 0.53 (24-31).
- MosaicVM vs envelope: p50 = 0.11 in every bucket (kit level); p99
  inside the ×1.25-guard-band envelope at ALL buckets.
- 32-token horizon validation of the other τ: flip margins ≤ 0.706 ≤
  τ_margin 1.72 ✓; τ_w p99 0.060 → frozen 0.075 with guard ✓.
- **G2 spec v1 frozen** (`g2_spec_v1.md`, cf7314da): full threshold
  table, eligibility rules (§30), post-flip/§38-unit procedure,
  deviations documented. Certification run decides 2/2.

## Track D — routed-expert decomposition (`routed_expert_decomposition.json`, a5e7dd19)

Per layer-call at N=112: **GU 15.77ms vs DOWN 8.52ms (1.85:1 per
FLOP-equivalent)**, epilogue 1.10ms, dispatch (layout+quant+decode)
flat 0.66-0.78ms across all N. Intercept view (N=1×43): GU ≈ 93ms,
DOWN ≈ 36ms, dispatch ≈ 28ms, epilogue ≈ 6ms of the 138ms routed
intercept. **GU is the kernel target; dispatch is the CUDA-graph
target.**

## Track S — early-exit Stage-0 (`early_exit_scan.json`, c10b15e4)

Zero-training oracle (real head on intermediate layer states, L=42
validates at 1.0):

| L | top-1 | top-16 | d2 | d4 |
|---|---|---|---|---|
| 40 | 0.545 | 0.938 | 0.311 | 0.076 |
| 38 | 0.353 | 0.763 | 0.107 | 0.005 |
| 36 | 0.272 | 0.562 | 0.061 | 0.005 |
| ≤20 | ≤0.10 | ≤0.23 | ~0 | 0 |

Deep-exit signal is real (L=40: 54.5% top-1 raw-head, 93.8% top-16
coverage); shallow exits need trained heads. Next: Stage-0.5 — train
lightweight exit heads on recorded states (the scan provides the
dataset recipe), then multi-token heads + B4-O E[A] evaluation.

## Track M — VRAM census (`vram_census.json`, db5d9b0d)

- **7.79 vs 7.0 resolved**: 7.80 GB = fp8 codes (5.68) + small (0.02) +
  embed (1.06) + head (1.06) — the historical figure included embed;
  the 7.0 "executable" number excluded it. Both correct for scope.
- Totals: core+bank128+workspace+KV+HC+reserve = **11.09 GB** (matches
  TRUE P2 measured residency); bank256-packed4 = 12.80 GB — does not
  fit → production requires hot-subset bank + RAM warm tier (per §87).
- **Proposer budget ≈ 0 GB at full target residency; ≈ 2.1 GB with CPU
  embed/head.** Proposer architectures must be sized accordingly.

## Status vs roadmap

```text
τ_h(prefix) frozen           DONE (G2 v1 written)
GU/DOWN split                DONE (GU 1.85:1, dispatch flat)
VRAM census                  DONE (proposer budget ~2.1 GB max)
early-exit Stage-0           DONE (L=40 54.5% raw-head)
G2 §38-unit certification    NEXT (spec frozen, run it)
sandbox                      DEAD (all data local; a fresh one would
                             let the disjoint 32-token Blackwell side
                             supersede the cert-prompt pair)
```
