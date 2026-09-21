# Session Report — Warm-Start Deconfound: Outcome B
# (WS-0 3/21, WS-1 Recovers 10/21 at 3.3ms; LRDL-1 Authorized)

**Date:** 2026-09-13
**Roadmap:** LRDL-0 REGRESSED — Warm-Start Interface Deconfound
Is NEXT (§8–§24, §54)

## Question

Does the LRDL-0 regression (8/21) come from the low-rank
interface itself, or from retraining the trunk from scratch?

## Setup

Frozen HIST-1 trunk. **S16 integrity gate (bitwise,
`d3_ws_integrity.py`, artifacts/d3_ws/ws_integrity.json):**
(1) trunk state_dict and RAW pre-head z are torch.equal
between the canonical evaluator path and the WS harness on
all 21 DEV inputs; (2) the WS-1 artifact contains only Q +
references and the HIST-1 checkpoint and basis files are
byte-identical (SHA-256) to their WS-1-save-time values —
training never wrote through; (3) behavioral replay: 10/21
hits and per-prompt margins match the frozen telemetry to
4.9e-5, with both canonical-path forms `W_head(w_norm·z_rms)`
and `(W_head·w_norm)z_rms` cross-checked equal. The original
in-run check was behavioral replay, not the bitwise gate; the
separate persisted bitwise gate is the authoritative record.
Frozen rank-256 basis E_r from randomized SVD of
W_eff = W_head·diag(w_norm).

**Bug fix (pre-run):** the original WS harness and the LRDL
basis verification applied w_norm twice (projecting through
W_eff a z that already contained w_norm). Fixed to RMS-only
projection everywhere. Corrected basis telemetry: r256 top-1
agreement 5.84% (was mis-reported 6.03%), r512 11.01% (was
11.72%); r512 loading also fixed to honor --rank (previously
silently loaded the r256 basis file). LRDL-0 TRAINING/EVAL
RESULTS ARE UNAFFECTED (they never used the projection path).

## WS-0: zero-training factorization — S1 = 3/21

Pure rank-256 projection of the frozen HIST-1 decision states
(q = V_r^T z_rms; logits = E_r q):

| metric | HIST-1 (full head) | WS-0 (r256) |
|---|---|---|
| S1 | 10/21 | **3/21** |
| Top16 | 17/21 | 6/21 |
| NLL | 4.417 | 7.563 |
| hits | 10 prompts | coding_14, longform_14, retrieval_14 |

The raw SVD projection is catastrophically lossy for greedy
decisions — it loses even the stable convo/math hits, not
just the fragile json boundaries.

## WS-1: q-head-only adaptation — S1 = 10/21 (FULL RECOVERY)

Same frozen trunk + frozen E_r; only Q: z_rms → q ∈ R^256
trained (init Q = V_r^T, 600 steps, CE+0.5KL, 1.05M
trainable params, ‖Q−P‖ drift 20.8):

| metric | HIST-1 | WS-1 | LRDL-0 r256 |
|---|---|---|---|
| S1 | 10/21 | **10/21** | 8/21 |
| hits | (10 prompts) | **identical set** | −json_13,14 |
| Top4 | 15/21 | **16/21** | 13/21 |
| Top8 | 15/21 | **17/21** | — |
| splice mean | 2.809 | **2.809** | 2.619 |
| trim20 | 1.769 | **1.769** | — |
| LOO | 2.150 | **2.150** | — |
| C2 | .800 | **.800** | .875 |
| first-position wall | 9.4 ms | **3.3 ms** | 3.1 ms |

json probes: json_13 −2.68 (rank 10) → +4.48 (rank 1);
json_14 −4.12 (rank 16) → +1.96 (rank 1). The fragile
razor-thin margins become ROBUST.

## Interpretation — roadmap Outcome B (§22)

$$
WS0 = 3/21 \ll 10/21,\qquad WS1 = 10/21
$$

1. **The LRDL-0 8/21 regression was a retraining artifact.**
   With the proven HIST-1 trunk retained, the low-rank
   interface costs NOTHING after adaptation: identical hit
   set, identical splice statistics, identical C2.
2. **The raw SVD basis is decision-lossy (WS-0 3/21) but a
   1M-param linear re-mapping fully repairs it (WS-1).** The
   information the trunk extracts is not rank-256-reducible
   through V_r^T, but the adapted Q finds an equivalent
   rank-256 encoding of the same decisions.
3. **The 11/21 genuine-signal gate is NOT met.** S1 stays at
   10/21; WS-1 matches but does not exceed the comparator.
   Top4/Top8 improve (16/21, 17/21) — latent breadth exists,
   but no additional Top-1 conversion.
4. **System economics improve materially at zero acceptance
   cost:** 9.4 → 3.3 ms first position (2.9×), and the head
   matmul is [256] @ [129280×256] instead of [4096] @ full.

## Consequences per roadmap

- **Outcome B (§22): authorize LRDL-1 constrained vocabulary
  adapter** — WS recovered baseline while frozen-basis
  geometry remains a plausible limitation (WS-0's 3/21 shows
  the basis itself is decision-lossy; WS-1's Top4 16/21 with
  S1 10/21 shows unconverted breadth).
- WS-1 is NOT a new DEV comparator (no gate advance) but is
  the fastest known equal-quality first-position
  representation: frozen for LRDL-1 as the starting point.
- Per §20/§23: splice already run (mandatory at S1 ≥ 10) —
  identical to baseline; no BLIND-ADJ (E[A] 2.809 < 4).
- HIST-1 + A2 + ps remains the frozen DEV comparator pending
  LRDL-1.
- Capability gate E[A] ≥ 4 OPEN. Engineering 92–93%.

## Artifacts

```text
artifacts/d3_ws/ws_integrity.json   S16 bitwise gate (PASS)
artifacts/d3_ws/ws0_eval.json       WS-0 telemetry (3/21)
artifacts/d3_ws/ws1.pt              trained Q + integrity refs
artifacts/d3_ws/ws1_eval.json       WS-1 telemetry + splice
artifacts/d3_lrdl0/basis_verify_r{256,512}.json  (corrected)
mosaicvm/bench/d3_ws.py             WS harness (fixed)
mosaicvm/bench/d3_lrdl.py           basis loader fixed
```
