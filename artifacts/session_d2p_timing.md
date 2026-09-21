# Session Report — D2-P: Adapter-Inclusive CPU Expert-Path Timing

**Date:** 2026-09-10
**Roadmap:** B-Race Complete (§38 — D2-P is the next system gate)

## Methodology (production-correct, after two rejected drafts)

- **Real routed activations** hooked from live kit-model replays
  (3 val prompts, prefill + 8 decode steps), stratified by
  (layer, matrix): L41/L42 × w1/w3/w2, 113-115 experts/stratum,
  M values 1-14 from natural routing.
- **Full hybrid call timed end-to-end**: GPU-resident input →
  blocking D2H → source E4M3/E8M0 act_quant adapter → VNNI GEMM
  (prepared weights) → bf16 → H2D → `cuda.synchronize()`.
- **Contention from a separate process** (independent CUDA
  context, sustained 4096² bf16 matmul), not a same-stream
  thread.
- Regimes: warm (same expert), dram-cold (round-robin across the
  stratum's experts; all prepared/RAM-resident), contended.
  p50 over 50 reps (20 for first-use).
- Earlier same-run rows (CPU-kernel-only diagnostics, no
  transfers) are retained in the JSON under `regimes` with
  explicit semantics; e2e rows are the production numbers.

## End-to-end results (p50 ms per expert call, v5 corrected)

The first cold-row draft was INVALID (an extra CPU->GPU upload
inside the timed lambda); v5 preloads activations GPU-side
outside timing so the timed path is exactly D2H + adapter + GEMM
+ H2D + sync. Corrected numbers:

| regime | M=1 | M=8-14 |
|---|---|---|
| idle, warm | 0.52-0.99 | 2.26-4.15 |
| idle, dram-cold | 0.67-0.99 | 2.14-3.92 |
| contended, warm | 2.63-5.48 | 2.73-7.79 |
| contended, cold | 2.63-4.57 | 2.72-10.36 |

Warm and dram-cold idle latencies CONVERGE (prepared weights are
RAM-resident; streaming 13.4MB/ expert is largely hidden under
the adapter at M<=6). Contention — a separate-process sustained
GPU workload — is the dominant factor (3-5x idle).

### Startup-cost reconciliation (roadmap S42)

Three measurements disagree and are now reconciled:
- **0.56 s/expert** = the v3 batch (684 experts in 381s) —
  includes COLD disk reads of safetensors shards and first-touch.
- **0.149 s/expert** = warm-cache p50, direct (49.7 ms/matrix ×
  3 matrices; disk already paged).
- **0.0349 s/expert** (JSON field) = batch mean with reads
  amortized differently — NOT a per-expert wall figure.

Canonical: startup prepare ≈ **0.15 s/expert warm** (~76s for the
full 512-expert bank single-threaded; ~6s across 12 threads);
first-ever cold pass adds disk-read time. Steady-state D2-P
latency is unaffected.

### Prepared-bank RAM footprint (roadmap S43 — MEASURED, not
### inferred)

Per matrix (e.g. L41.e0.w1): tiled int8 codes 8.39 MB + float32
scales 1.05 MB + int32 weight-sums 1.05 MB = **10.49 MB prepared
vs 4.46 MB source-packed (2.35x expansion)**.

**Full 2-layer prepared bank: 16.11 GB vs 6.85 GB source-packed.**

This is a material RAM-budget change: on the ~24 GB host budget
(with OS + model + verifier state resident) 16.11 GB of prepared
tail bank is tight-to-infeasible AS-IS. Mitigation options before
D2-COMPLETE-TAIL: store scales as raw E8M0 uint8 (0.26 MB) with
on-the-fly expansion in the kernel; pack weight-sums as int16
where values fit; or prepare-on-demand with an LRU above a smaller
resident set. Decision deferred to the complete-tail measurement —
but the 6.86 GB tail-bank claim is RETIRED.

## First-order production implications

- A decode-position tail step routes 6 experts × 3 matrices ×
  2 layers = **36 expert calls**. At contended M1 (~3.5-5ms):
  ~125-180ms/step serial. At idle (~0.5-1ms): ~20-36ms/step.
  The idle warm/cold convergence means RAM residency works —
  expert identity rarely costs latency once prepared.
- **Batching across K positions is architecturally essential**:
  the same experts recur; the VNNI M-scaling above shows
  M=14 costs only ~2-4× an M=1 call while serving 14× the rows.
  Block/chunk tail geometry (roadmap S39) must be measured next
  with real K-batched routing.
- Contention roughly doubles-to-quadruples per-call latency —
  the H2 scheduler must interleave, not co-saturate.
- These are expert-path numbers ONLY; the complete tail wall
  (attention/DSA, core, shared, transfers, head) is the next D2
  measurement. No production throughput claim is made.

## Artifacts

```text
artifacts/d2p_expert_timing.json            e2e + CPU diag + transfers
artifacts/d2p_expert_timing_v3_cpu_diag.json (superseded intermediate)
mosaicvm/bench/d2p_expert_timing.py         harness
```

## C2 pre-flight verdict: STAGE/BOUND (the free rail is
## pathological raw)

Full-contract pre-flight (all 84 train prompts, independent
raw/clamp-30/clamp-10 trajectories from the exact post-1000-step
base; artifacts/d3_c2_preflight.json):

| rail | mean free loss | ratio vs TF | drift max |
|---|---|---|---|
| raw | 3140.9 | 5,940× | 288.3 |
| clamp-30σ | 2778.5 | 5,254× | 266.0 |
| clamp-10σ | 2635.6 | 4,984× | 265.9 |

All finite, but: features reach 564σ (86% of positions touch the
30σ clamp), predictions reach |12,426|, drift ~288. Input
clamping barely helps — the explosion is driven by the
PREDICTION magnitude feeding back, so the earlier C2 NaN was
inevitable, and B1's round-24 bounded drift (max 3.13) was
exclusively the 0.75→3.0 curriculum's doing (its round-1 raw max
was 11.68). The ±30σ input clamp alone is NOT a fix and the
skip-guard would have trained on garbage.

METHODOLOGY CAVEAT (post-hoc): the pre-flight's loss-ratio gate
compared the UNCLIPPED recovery loss against the training
objective's teacher-forced loss; the actual training loss clamps
recovery targets at ±4 standardized. The recorded STAGE/BOUND
verdict was the conservative direction (raw pathology was
overwhelming under EITHER loss), so the decision stands; the
harness now reports both losses and gates on the actual training
objective for future pre-flights.

DECISION (revised after the B1-equivalence catch): a plain
staged-free arm (0.75→3.0 schedule on the training model's own
chains) is ALGORITHMICALLY IDENTICAL to B1 — same schedule, same
re-anchor semantics, same dual loss, fresh 1000-step base — and
was stopped before wasting the run. B1 remains that control.

C2 is redesigned around a COLLECTOR rail: uninterrupted rollout
states from the FROZEN B1 checkpoint (drift-bounded 0.75→3.0 at
GENERATION time — trajectories drop beyond the bound, never
re-anchor), recovery target = gold − collector state, anchor rail
(gold one-step loss) in every update. The free-rail distribution
is therefore (i) genuinely uninterrupted, (ii) distinct from
anything the training model generates, and (iii) progressively
deeper via the schedule. Unit-verified: uninterrupted trajectories,
drop-not-anchor semantics, drift bounded by construction.

The unconditional ±30σ input clamp is REMOVED (it silently
changed reset/B0/C1 objectives too, and its "targets stay exact"
comment was false — targets are ±4-clamped by definition). Input
clamping survives only as an explicit free-arm option.
