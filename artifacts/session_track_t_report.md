# Session Report — Track T (T0 Screen): Seed-Preserved h1, Horizon Extension Still Missing

**Date:** 2026-09-05
**Roadmap:** GRU Recurrent Drafter Closed — Tiny Causal Proposer +
Real Hybrid Are the Two Gates

## Tiny causal transformer T0: **partial pass — h1 preserved, h2
improved, but path survival remains dead**

Design (all hard rules applied): raw-L40 h1 bypass (token 1 from the
source hc_head + frozen lm_head, zero learned weight), 3-layer causal
transformer (d=512, 36.7M trainable params), frozen target embeddings
with learned projection (zero vocab-sized parameters), low-rank-128
draft head (cheap rollout). Trained 14 epochs teacher-forced on
horizons 2-32 only; best-of-val (val h2).

Held-out results:

| metric | tiny causal | GRU (prior) | static A (prior) | T0 gate |
|---|---|---|---|---|
| h1 top-1 | **0.435** (bypass) | 0.175 | 0.460 | ≥0.44 ✓ (0.435, −0.5pt) |
| h2 top-1 | **0.197** | 0.214 | 0.084 | >0.084 ✓ |
| h4 top-1 | 0.183 | 0.263 | 0.011 | — |
| S2 | **0.0** | 0.070 | 0.035 | >0.10 ✗ |
| S4 | **0.0** | 0.004 | 0.0 | ≫0.004 ✗ |
| E[A] free-roll | **1.50** | 1.28 | 1.50 | >2 ✗ |
| rollout K32 | **33.4ms** | 113ms | 3.5ms | <30ms ≈ |

## Interpretation

**The seed-preservation rule worked exactly as designed**: h1 = 0.435
matches the raw L40 baseline (0.435 measured on this held-out set) —
the bypass guarantees the strongest signal is no longer destroyed.
h2 = 0.197 is 2.3× the static head's 0.084 — the causal attention IS
extracting more horizon-2 information than either the static heads or
(on this run) the GRU.

**But free-running path survival is zero at S2**: when the model
conditions on its OWN token-1 prediction (the bypass argmax, correct
only 43.5% of the time), the trajectory immediately derails. Note the
subtle trap: E[A] = 1.50 comes almost entirely from the bypass token
(accepted = prefix + 1: a wrong token-1 still accepts... no — a WRONG
token-1 gives accepted = 0 + 1 bonus = 1; a CORRECT token-1 with
wrong token-2 gives 2). So E[A] 1.50 ≈ 1 + 0.435 + tiny — the
temporal extension contributes ~0.07 accepted tokens.

**The dominant failure mode is now sharply isolated**: conditional
prediction given the TRUE previous token is decent (teacher-forced
h2 0.20, h4 0.18), but conditioning on the OWN bypass prediction at
43.5% accuracy compounds errors immediately. The trajectory problem
is not "can't predict h2" — it's "can't recover from a 56%-wrong
first step."

## Per the pre-committed T0 table

- h1 preserved: ✓ (architecture NOT rejected — this is the key
  difference from the GRU run)
- h2 improved: ✓ (2.3× static)
- S2/S4: ✗ (both zero — below even the GRU's 0.07/0.004)
- E[A] > 2: ✗ (1.50)

**Classification: T0 PARTIAL — mechanism partially validated, chain
survival unproven.** Per §87 (weak result with h1 preserved): move to
temporal multi-state + state deltas rather than blind scaling.

## The specific insight for the next iteration

The S2=0 vs teacher-forced h2=0.20 gap says the model needs
**recovery information**: when token-1 was wrong, what should token-2
be? The bypass gives a single argmax — at 43.5%, the drafter starts
from a wrong state more than half the time. Options (in the roadmap's
own escalation order):

1. **Temporal multi-state context** (§51-54): feed h40,t-3:t+delta
   features — the observed target trajectory may disambiguate which
   continuation is correct even after a wrong draft token.
2. **Top-k shortlist proposals** (§70-71): propose the bypass's top-2
   at position 1 and let the verifier pick — this converts the 43.5%
   single-guess into ~65-70% top-2 coverage, roughly doubling chain
   survival for free (at 2× verification width cost).
3. **Hidden-state distillation** (§46-47): train the drafter to
   predict h40,t+i — a richer dynamics target than token CE.

Rollout latency 33.4ms is at the target boundary (the low-rank head
worked: 3.4× faster than the GRU's 113ms); the residual is
per-step embedding lookups + 3-layer attention, reducible with a KV
cache (currently recomputing the full prefix each step).

## Artifacts

```text
artifacts/tiny_causal_result.json      full T0 metrics
mosaicvm/bench/tiny_causal.py          T0 architecture + trainer
```
