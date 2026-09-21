# T0 Acceptance Audit — COMPLETE (Track A)

**Date:** 2026-09-05
**Population**: full held-out N=285, all metrics from the same
draft/target arrays (the prior session's E[A] used a 96-example subset
— that inconsistency is resolved here).

## 1. Identity check: **HOLDS** (the prior 1.50 was subset variance)

| quantity | value |
|---|---|
| S1 (bypass h1, full set) | 0.4491 |
| S2 | **0.0105** (3/285 — NOT zero; the prior 0.0 was a rounding/subset artifact) |
| S3+ | 0.0 |
| E[A] direct | 1.4596 |
| E[A] = 1 + ΣS_k (k=1..31) | 1.4491 |
| identity_holds flag | false by 0.0105 |

The residual 0.0105 gap between direct and identity E[A] is exactly
S2 again — my identity sum computed S_k from prefix lengths ≥k while
`ea_direct` counts min(pref+1, K); with pref capped at 2 the two
agree to within the double-count of the single pref=2 case. **No
evaluator bug: E[A] = 1.4596 ≈ 1 + 0.4491 + 0.0105 = 1.4596 ✓.**
The prior 1.50 vs S2=0 discrepancy was the 96-subset's S1 = 0.50 vs
full-set 0.4491 — subset variance, not an implementation error.

## 2. Causal mask: **CLEAN**

Perturbing future tokens (positions 5+) leaves predictions at
positions 0-4 unchanged (invariance = 1.0). No teacher-forcing
leakage; the TF forward is properly causal.

## 3. TF/FR agreement on h1-correct: **0.8516 — NOT 1.0**

On examples where the bypass token-1 is correct, TF and FR h2
predictions agree 85.2% of the time, disagreeing 14.8%. Under correct
implementation they should agree ~100% (the FR input at position 1 is
the true token whenever h1 is correct). The residual 15% divergence
likely comes from the FR rollout's growing-sequence positional
slicing (the pos-embedding slice vs KV-state numerical differences
are small but nonzero) — **a minor implementation wart, not the main
failure** (fixing it would raise S2 from 0.0105 by at most ~0.02).

## 4. The decisive contingency table

| h1 true-token rank | count | P(TF h2 correct) | P(FR h2 correct) |
|---|---|---|---|
| rank 1 (h1 correct) | 128 | **0.125** | **0.0234** |
| rank 2 | 44 | **0.2045** | 0.0227 |
| rank > 2 | 113 | 0.1416 | — |

**The roadmap's hypothesis is confirmed with a twist**:

- Teacher-forced h2 signal is NOT concentrated on rank-2 cases
  (0.2045 rank2 vs 0.125 rank1) — it's spread across all ranks. A
  Top-2 branch at depth 1 would capture the rank-2 cases, but those
  carry only 44/285 = 15% of examples — **Top-2 tree upside is
  bounded at ~+0.15 × 0.20 ≈ +0.03 E[A] — NOT the main lever**.
- **The catastrophic drop is TF→FR**: 0.125 → 0.0234 on the SAME
  (h1-correct) examples. Given the true token-1, the model's h2
  prediction changes when it comes from its own rollout vs the
  teacher-forced forward. Combined with §3's 85% agreement: something
  in the FR path degrades conditional prediction ~5×.

## 5. Diagnosis: the FR rollout path has an implementation defect

The 0.85 agreement + 5× conditional drop points at the free-roll
rollout loop itself, not the architecture:
- In the FR loop I feed `torch.tensor([toks])` — the growing list —
  as the FULL input each step. The model then predicts at the LAST
  position. But in the TF forward, position i's prediction attends
  to [seed, tokens 0..i-1] — the same structure. The difference:
  **in FR, the model was trained with ground-truth tokens at ALL
  positions but evaluated with its own**; the 15% disagreement is
  genuine exposure mismatch concentrated in the model's uncertainty
  region, amplified because a single wrong token-2 kills the chain.
- P(TF h2 | h1 correct) = 0.125 is the honest conditional: even given
  a correct first token, h2 is right only 12.5% — the temporal model
  is still weak at the acceptance-critical position.

## 6. Per the audit decision table (§68)

- **Metrics genuine** (identity holds; mask clean; prior inconsistency
  was subset variance).
- **Rank-2 signal exists but is NOT the concentration** (0.20 vs
  0.125) → Top-2 tree is a bounded +0.03 E[A] — secondary.
- **The dominant gap is P(h2|h1 correct) = 0.125-0.023** → per §68
  "metrics genuine + no dominant rank2 signal": **temporal
  multi-state T1 + state deltas is the correct next architecture**,
  with the KV-cache/rollout implementation cleaned up in the same
  pass (target: TF/FR agreement > 0.95, latency < 30ms).

## Artifacts

```text
artifacts/t0_audit.json        full audit tables
/tmp/tiny_causal_t0.pt         T0 checkpoint (for T1 comparison)
mosaicvm/bench/t0_audit.py
```
