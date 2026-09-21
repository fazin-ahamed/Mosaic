# Session Report — Track P: Recurrent Drafter R0-R2 Result

**Date:** 2026-09-05
**Roadmap:** VNNI v4 PASS-GOOD — Real Hybrid + Temporal Proposer Are
the Final Architectural Gates

## Recurrent target-feature drafter: **E[A] ≈ 1.27 — FAILS the R0 signal gate (≤2 = weak)**

Architecture: z0 = tanh(seed(h40)); GRU recurrence on token
embeddings; frozen lm_head batched over K. 118.6M params (embed table
dominates). Trained R0 (teacher-forced, 12 ep) → R1 (scheduled
sampling p=0.7, 6 ep) → R2 (p=0.4, 4 ep) on the 1076/267/285
Stage-1 windows; best-of-val selection.

Held-out results:

| metric | value |
|---|---|
| top1 @ h1 | 0.175 |
| top1 @ h2 | 0.214 |
| top1 @ h4-h32 | 0.23-0.26 |
| survival @ 2 | 0.070 |
| survival @ 4 | 0.004 |
| survival @ 8+ | 0.0 |
| **E[A] (teacher-forced argmax)** | **1.274** |
| **E[A] (free-roll subset)** | **1.281** |
| rollout latency K=32 | 113 ms |

## Interpretation per the pre-committed gate

E[A] ≤ 2 → **"SIMPLE RECURRENT DESIGN WEAK"**. The GRU-on-embeddings
recurrence did NOT fix the horizon collapse: horizon-2 top-1 is even
worse than the static heads (0.21 vs 0.08 — similar class), and path
survival beyond 4 tokens is zero. Free-roll ≈ teacher-forced E[A]
confirms the recurrence isn't even helping with exposure — the state
evolution does not track the target.

## Honest caveats before escalation

1. **Data scale**: 1076 windows is 5-10× smaller than the recurrent
   family likely needs (the embedding table alone is 66M params ×
   129280 vocab — the model may be memorizing the prompt diversity
   rather than learning target dynamics).
2. **Horizon-flat top1 (0.23-0.26 for h4-h32)** is suspicious — the
   GRU has learned a marginal distribution, not a conditional
   trajectory (perplexity plateaued around loss 0.9-1.2 = memorization
   of the common-token distribution).
3. **L40 seed quality**: the seed alone carries h1 top-1 = 0.46, but
   the recurrent model's h1 = 0.175 — the recurrence is DESTROYING
   seed information rather than extending it. The seed projection and
   the GRU are not co-trained productively.

## Per the roadmap §58: escalation order

- **Next: tiny causal transformer** seeded by L40 + proposed-token
  history (attention over the token history is a fundamentally
  different mechanism for using the trajectory).
- **Then: temporal multi-state** (h40,t-k:t as context — real observed
  target dynamics rather than a learned simulation).
- Data scale-up (1000+ prompts) is justified only after an
  architecture shows it can at least preserve the seed's h1 = 0.46
  while extending h2+ beyond 0.084.

## Rollout latency note

113ms for K=32 is ABOVE the ~30ms budget (the per-step head GEMM at
batch 1 is 3.4ms × 32 steps). The batched-head design (head after all
recurrence) only works teacher-forced; a production free-running
drafter needs either (a) a small vocabulary head (top-k restricted),
or (b) the tiny-causal design with KV-cached attention and one final
head pass. This is a design constraint for the next iteration, not a
fundamental blocker.

## Artifacts

```text
artifacts/recurrent_race_result.json     full metrics
mosaicvm/bench/recurrent_drafter.py      R0/R1/R2 trainer + eval
```
