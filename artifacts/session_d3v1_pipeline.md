# Session Report — DATA scale-up + D3-v1 token-conditioned student

**Date:** 2026-09-08
**Roadmap:** First Real D3 Student Failed — Residual Geometry, Data
Scale, and Production Hybrid Parity Are Now the Main Gates

## DATA: 105-prompt capture in progress

`artifacts/d3_data_105/` — all 84 train (`*_01.._*12`) + 21 val
(`*_13.._*15`) manifest prompts, 33 generated states each
(S1_SOURCE=1, S1_N_GEN=33, SCG_CACHE_CAP=96). Resume-safe;
seeded with the 14 existing tranche captures (validated + skipped).
Recorder running in background (~187s/prompt after CPU contention).

Test split (`*_16.._*18`) remains excluded from source capture.
Protected-corpus (G1/G2) exact-text overlap check enforced by the
recorder. This exceeds the roadmap's 70-prompt first learning-curve
gate (§77) and provides 84×32 = 2688 train transitions + 672 val.

## D3-v1: token-conditioned one-step Δh40 student — BUILT, three
## training-pathology bugs found and fixed

`mosaicvm/bench/d3_v1.py` (trainer/exporter, CPU) and
`mosaicvm/bench/d3_v1_eval.py` (closed-loop evaluator: student →
real L41/L42 tail → FP32 head → draft token fed back).

Architecture: DeltaNet residual MLP — input
[h40_t | Δh40_t | emb(token_{t+1}) | emb(token_t) | pos] (41024),
hidden 512, depth 3, full-state Δ output (16384), per-dim
input+target standardization, teacher-forced one-step training,
full-batch AdamW.

### Bugs found (each verified by controlled bisection)

1. **Minibatch gradient noise freezes the 41k-dim input layer.**
   Batch-64 training could not fit ANY target (real, random, or
   deterministic rank-64 linear; stuck at 0.9957). Full-batch Adam
   fits the deterministic control to 0.0007 in 300 steps. With
   DI=41024 and hidden=512, random init captures only ~1.2% of the
   relevant input subspace; minibatch noise prevents the rotation.
   SGD-momentum diverges; wider hidden does not help. Fix: full
   batch (default --batch 0).
2. **Input common-mode cone saturates GELU identically across
   samples.** Real inputs share a strong common mode (state-block
   pairwise cos ≈ 0.28; full-x ≈ 0.15) plus a 66× scale mismatch
   (state|Δ| ≈ 2.5 vs token-emb ≈ 0.04 — the driver token was
   invisible at init). REALX+random-Y could not be memorized
   (0.995) while RANDX+real-Y memorized (0.016). Fix: per-dim
   input standardization (x_mean/x_std, train-only, saved with the
   checkpoint; applied identically in all rollouts).
3. **Residual re-activation kills the skip path.** The original
   `h = GELU(m(h) + h)` re-activates the residual stream, so the
   common mode saturates every layer. True skip
   `h = h + GELU(m(h))` + standardization → real mapping memorizes
   to 0.073.

Additional latent bugs fixed en route: `S, K = c["S"], min(..., S)`
RHS-unbound crash; rollout token conditioning off-by-one (driver
token for position p is `tokens[p]`, not `tokens[p+1]`); export
diagnostics shape mismatch; delta-vs-absolute target semantics.

### Smoke result (7 train prompts — the known data-starved regime)

- Train loss → 0.016 (memorization works; pipeline healthy).
- Val per-step rel-L2: 1.065 (no generalization — expected at 7
  prompts; the roadmap's reason for DATA-first).
- `one_step` val (gold state + true token): rel-L2 0.87-1.01,
  cos 0.56-0.64 — consistently better than hold (rel-L2 1.05-1.31,
  cos 0.30-0.50). A weak but real token-conditioned signal.
- `tf_chain` (chained states, true tokens): explodes (rel-L2
  15-130) without exposure — per-step errors ≈ 1.0 compound
  off-manifold. Scheduled state exposure (§53) is implemented and
  reduces this to 2.2-4.3 (below).

### Closed-loop evaluator (ready, GPU-gated)

`d3_v1_eval.py` implements the roadmap §51 production loop with the
student's own tail-decoded drafts fed back (`cl_loop`) plus a
true-token variant (`tf_loop`), both gated per prompt by exact
h42 bit-equal + greedy-label replay. It requires CUDA: the CPU
reference tail differs from captured GPU h42 (known max_abs 4.0,
D2-E open item) and would fail the bit-exact gate.

### Scheduled state exposure (roadmap §53) — implemented and hardened

Phase-2 training rolls the model's own chained states on train
prompts and trains the recovery target (gold − chained state).
Naive exposure NaN'd (memorized model is brittle off-manifold;
chains explode even on train; recovery targets blew the loss).
Hardened with: drift re-anchoring (chains beyond rel-L2 0.75 reset
to gold — training sees only plausible drift), standardized-target
clamp ±4, gradient clipping. Result on the 7-prompt smoke:
tf_chain rel-L2 **15-130 → 2.2-4.3** with one_step skill unchanged
(0.85-1.01, cos 0.56-0.65). No NaN.

The smoke model remains data-starved (val per-step rel-L2 ~0.9).
NOTE on interpretation: synthetic D1-B showed that 0.3-rel-L2
synthetic perturbations remain useful after the real tail, but
that establishes NO acceptance threshold for real structured/
chained student residuals (roadmap §6: scalar rel-L2/cosine gates
are permanently retired as promotion criteria). Real-student
acceptance is OPEN until the exact GPU-tail evaluation runs; the
synthetic numbers are diagnostic only.
## Next

1. When the 105-prompt capture completes: train D3-v1 on 2688
   transitions (full-batch, hidden 512, ~800 steps + exposure),
   then run `d3_tail_eval.py` (exported modes) and `d3_v1_eval.py`
   (closed-loop) on GPU against the 21 val prompts.
2. If E[A] > 1 emerges: learning-curve per roadmap §122 table.
3. D2-E1 (CPU VNNI expert vs source GPU expert parity) remains
   queued — needs GPU, blocked behind the capture.

## Roadmap S14/S16/S61 hardening (post-advisory)

- `mosaicvm/bench/test_d3v1_alignment.py`: frozen regression
  proving IDENTICAL (driver, cur, pos) token triples across
  training / one_step export / scheduled exposure / chained
  rollout, and that cl_loop feeds ONLY root-draft + own drafts
  (no gold future token, no gold state) while tf_loop feeds true
  tokens. Runs on a stub tail — no GPU needed. PASSING.
- All feature construction now routes through a single
  `make_feat` (single source of truth for the pairing contract).
- The verification pass caught two real bugs before the automated
  chain hit them: a double-appended position block (41088-dim
  input vs 41024 model) and `weights_only=True` rejecting `Path`
  objects in the checkpoint args (would have crashed the
  closed-loop eval's loader). Both fixed and round-trip verified
  through the eval's own `load_student`.
- Checkpoints now carry script/manifest hashes, sanitized args,
  and seed (roadmap S61).
- Gradient accumulation (roadmap S25) is deliberately NOT
  implemented yet: the 2688-transition corpus fits full-batch
  (441MB). It becomes required at the 210+ prompt scale-up, with
  the S26 equivalence gate.

## Incidents

Two of my debug jobs OOM-killed the recorder (16GB leaked heredoc
process; 673M-param control Linear). Recorder restarted both times
(resume-safe, no capture loss beyond progress). Debug jobs now
memory-budgeted while the recorder runs.
