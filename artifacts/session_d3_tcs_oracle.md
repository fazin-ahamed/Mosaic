# Session Report — TCS Projection Oracle: FAIL; TCS CLOSED;
# Direct Decision Representation Is NEXT

**Date:** 2026-09-12
**Roadmap:** R3 Fully Closed — Exact-Tail Sensitivity / TCS Is
NEXT (§28–§30, §40)

## Oracle result: low-rank error correction does NOT improve
## exact-tail decisions

Baseline (uncorrected A2 one-step through exact tail):
mean E[A] = 2.905, S1 = 0.429.

| rank | λ | E[A] | S1 | ΔS1 | ΔE[A] |
|---|---|---|---|---|---|
| 8 | 0.25 | 2.905 | 0.429 | **+0.000** | +0.000 |
| 8 | 0.50 | 2.905 | 0.429 | **+0.000** | +0.000 |
| 8 | 1.00 | 2.667 | 0.381 | −0.048 | −0.238 |
| 16 | 0.25 | 2.905 | 0.429 | **+0.000** | +0.000 |
| 16 | 0.50 | 2.905 | 0.429 | **+0.000** | +0.000 |
| 16 | 1.00 | 2.524 | 0.381 | −0.048 | −0.381 |
| 32 | 0.25 | 2.905 | 0.429 | **+0.000** | +0.000 |
| 32 | 0.50 | 2.524 | 0.381 | −0.048 | −0.381 |
| 32 | 1.00 | 2.714 | 0.381 | −0.048 | −0.190 |

**Best ΔS1 = +0.000 (kill gate ≥ +0.10: FAILED).** Not a single
configuration improved a single metric. Aggressive corrections
(λ=1.0) actively hurt.

## Interpretation

The oracle removed the top-r SVD components of A2's h40
prediction errors — the directions where the errors have the
most VARIANCE — using gold information (an upper bound on what
any TCS-trained student could achieve). **Zero improvement.**

This proves: **the principal error-variance directions are NOT
the decision-critical directions.** The exact tail's decision
is insensitive to the subspace where A2's errors concentrate.
This is the anisotropy result from D1-B32 confirmed at the
oracle level: what makes state error large and what makes
decisions wrong are nearly orthogonal concerns.

A sensitivity-derived basis (finite-difference through the tail)
could in principle find different directions — but the oracle's
complete zero (not even marginal improvement from ANY rank/λ)
lowers the prior substantially, and the roadmap's kill gate is
decisive.

## Classification (CORRECTED: error-SVD ≠ FD sensitivity)

```text
TCS-error-SVD basis = CLOSED
  (error-variance directions are not decision-critical)
TCS-FD sensitivity  = OPEN / NEXT
  (the oracle did NOT test finite-difference sensitivity
   directions through the exact tail — only SVD-of-error
   directions; the FD scan remains the open experiment)
```

The SVD basis captures directions of LARGEST ERROR VARIANCE.
The FD scan measures directions of LARGEST DECISION SENSITIVITY.
These are different objects. The oracle's zero result proves only
the former is not decision-critical.

## What this means for the architecture

The entire h40 → (tail or bridge) → token pipeline has now been
exhaustively characterized:
- h40 recovery/exposure training: CLOSED (7 variants)
- bridge mixture/position/capacity/unfreeze: CLOSED (R3)
- error-subspace correction: CLOSED (this oracle)
- The remaining gap (S1 0.381-0.429 vs target) is NOT in the
  h40 representation's correctable subspace

## FD alpha-sweep oracle: TCS-FD CLOSED

Interpolating from A2's prediction toward gold:
h40_corrected = pred + alpha * (gold - pred).

| alpha | E[A] | recovery |
|---|---|---|
| 0.000 | 2.905 | 0% |
| 0.125 | 2.952 | 0.2% |
| 0.250 | 3.667 | **2.5%** |
| 0.500 | 5.238 | 7.8% |
| 0.750 | 12.238 | 31.0% |
| 1.000 | 33.000 | 100% |

**Gate (alpha=0.25 recovery ≥ 50%): FAIL at 2.5%.**

The recovery curve is STRONGLY CONVEX. Even the ORACLE
correction — moving directly toward gold, the best possible
direction — recovers only 7.8% of the acceptance gap at 50%
correction. The decision-critical error is DIFFUSE across the
h40 space, not concentrated in any subspace.

No basis construction (SVD, FD sensitivity, or otherwise) can
extract a compact decision-correcting subspace from this error
distribution: the direct gold direction is optimal and even it
requires ~75% correction for 31% recovery.

## FD sensitivity scan: SIGNAL (delta-own = 6.64× random)

The exact-tail finite-difference scan (7 domains × 2 positions ×
8 direction types × 2 ε × 2 signs = 448 probes through the real
L41/L42):

| direction type | sensitivity vs random |
|---|---|
| **delta-own** (true h40 motion at same position) | **6.64×** |
| own-err (A2 error at same position) | 2.15× |
| delta-x (cross-prompt deltas) | 1.00× |
| x-err (cross-prompt errors) | ~1.00× |
| random (baseline) | 1.00× |

The decision-critical direction is the POSITION-SPECIFIC true
h40 state motion — not the error variance directions (SVD) and
not cross-prompt generalizations.

## FD projection oracle: FAIL on the breadth gate

Per-position delta correction (the 6.64× direction, gold-assisted):

| λ | E[A] | S1 | ΔS1 |
|---|---|---|---|
| 0.25 | 3.190 | 0.429 | **+0.000** |
| 0.50 | 3.286 | 0.429 | **+0.000** |
| 1.00 | 3.524 | 0.429 | **+0.000** |

Domain-SVD rank-8 delta basis: NO CHANGE at any λ.

**Kill gate (ΔS1 ≥ 0.10): FAIL at +0.000.**

The delta-direction correction improves E[A] (up to +0.619,
a 21% gain) but does NOT improve first-position survival (S1
stuck at 0.429 = the uncorrected baseline). The sensitivity is
real but affects DEPTH, not BREADTH — the bridge already has
depth; S1 is the bottleneck.

```text
TCS-error-SVD         = CLOSED (variance directions not critical)
TCS-full-error interp = CLOSED (full-vector recovery is diffuse)
TCS-FD sensitivity   = SIGNAL CONFIRMED (delta-own 6.64x)
TCS-FD projection    = FAIL on breadth gate (dS1 = 0.000)
  (E[A] improves +0.619 but S1 does not move)
TCS                   = CLOSED for the S1/breadth objective
NEXT                  = direct decision representation
```

## Artifacts

```text
artifacts/d3_tcs_oracle.json           full oracle matrix
mosaicvm/bench/d3_tcs_oracle.py        oracle harness
```
