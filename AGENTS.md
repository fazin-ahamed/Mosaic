# MosaicVM — current continuation contract (2026-09-06)

The user's latest roadmap is preserved verbatim in
`artifacts/roadmap_d1b32_handoff.md`. It supersedes the D1-B handoff in
`artifacts/roadmap_d1b_handoff.md` and the earlier B3-S roadmap,
archived in `artifacts/roadmap_pre_d1b.md`. Read the latest handoff together
with `artifacts/session_d1b_continuation.md` and `artifacts/d2_readiness.md`;
the latter documents add code-backed corrections, not new performance claims.

## Objective and exactness

RTX 3060 12 GB, deployment host budget approximately 24 GB plus NVMe.
Primary success: >=30 sustained final accepted/output tokens/s; ultimate
success at 1,048,576 exact context tokens. Report actual host capacity separately.

Preserve source FP8/FP4 weights/scales, activation quantization, exact DSA,
source routing/hash/selection-only bias, all six Top-6 experts, normalization,
shared expert, SwiGLU, ordered accumulation and source cast boundaries.
The source FP32 final head projection does not remove earlier BF16 casts.
Approximate state prediction is draft-only; full exact verification controls emission.
Never promote a shim/reference timing, synthetic workload, storage census,
or perfect-residency projection to measured production throughput.
Keep exactness and hardware evidence classes separate.

## Current direction

Future-h40 predictor -> actual source L41/L42 -> source head -> exact full
verifier. D1-R recorded bit-exact h42 on three coding prompts. Historical
D1-B is a promising synthetic perturbation probe, not proof of real-student
acceptance, route redundancy, cross-domain robustness, or production readiness.

D1-B v2 repairs complete nested KV/compressor reset, source head casts,
terminal-position scoring and K accounting. K means uncertain speculative
decisions excluding the exact root; prefix+1 has ceiling K+1. Score one
exact committed boundary per prompt. Sliding crops of perturbed history
must not be counted as independent production cycles. Report K and target
verification width N separately; N=K is not yet an integrated-cycle proof.

No heuristic engineering percentage certifies success. Retain historical
artifacts unchanged and record superseding interpretation separately.

## Next work, in parallel where independent

1. D1-B32: exact replay sanity, then matched K8/K16/K32, all seven held-out
   domains. Report per-prompt prefix survival, Top-k, final drift and uncertainty.
   Test-only captures cannot enter student training. Synthetic residuals must
   eventually be replaced by actual held-out D3 residuals.
2. D2: source tail storage inventory is measured; actual runtime footprint
   and production kernel latency remain open. VNNI-expanded weights cost
   more than packed FP4. Include head, KV/DSA, workspaces and coexistence.
   Source continuation currently runs serially; chunked continuation requires
   its own causal correctness proof.
3. DATA/D3: scale distinct prompts with prompt-level splits; predict future
   h40, initially K4/K8 chunks, then evaluate through real tail. Hidden-state
   loss or cosine alone cannot select the production student.
4. H2: real (layer, expert) diversity, full GU/SwiGLU/DOWN, actual CPU/H2D
   traffic and strategic residency. H1 remains a historical synthetic baseline.
5. Combine measured student/tail/verifier cost with held-out E[A], then P3/P4.
   At K32, perfect prefix+1 allows 1.10 seconds total at 30 tok/s; this is
   a ceiling, not a pass. Long-context qualification follows short-context success.

Closed old T1 trees/GRU/static proposer branches remain closed. Reopen
BranchPack only with real future-h40 student/tail logits and measured economics.
Do not infer tail MoE removal from route changes under synthetic perturbations.

## Working environment

This workspace has no Git metadata; do not claim commits or clean Git status.
Use `.venv-cu/bin/python` for local CUDA work. The full pinned checkpoint
is locally present. Bound the source harness expert cache on this 12 GB GPU
(e.g. SCG_CACHE_CAP=24 for correctness probes). Source torch shims are slow
reference implementations, not production D2/P4 benchmarks.

## Evidence update — 2026-09-08

Seven-domain D1-B32 completed: all exact replays pass. At 0.1 synthetic
relative L2, K32 E[A] ranges 15.14–25.57 across families; at 0.3 it ranges
8.71–13.00. Top16 remains 100%. This is seven test prompts, not real D3
residuals or a production geometry decision. See
`artifacts/session_d1b32_cross_domain.md`.

Historical VNNI v4-B lacks a source signed/fractional E4M3 adapter. The new
`vnni_source` three-plane adapter passes bounded integer and one-real-expert
CPU-reference checks; full-tail/native-source certification and production
latency remain open. Its footprint includes weight-sum compensation buffers.
See `artifacts/d2_cpu_adapter_v2.md`. Never apply old v4-B timing to it.

Source-class DATA recording is now available through `S1_SOURCE=1` in
`stage1_record.py`; select train/val categories explicitly with S1_CATS.
Two pilot captures (coding_01 train, coding_13 val) exist and pass schema/
alignment checks. They are not enough to establish a useful student. See
`artifacts/d3_data_readiness.md`; keep all D1-B test captures excluded.

## Continuation evidence — first real student, 2026-09-08

The source corpus now includes seven train and seven validation prompts,
33 generated-state rows each, in `artifacts/d3_multidomain_tranche1/`.
The train-only PCA/ridge baseline and predicted-state recursive K8 variant
are implemented; neither is token-conditioned or receives tail feedback.
See `artifacts/session_d3_tranche1.md` and the completed evaluator output
`artifacts/d3_tranche1_tail.json` when present. State loss is not acceptance.

CPU adapter matrix: 24/24 complete expert cases and 72/72 linear stages
match ordered CPU reference. The full captured coding CPU tail also matches
with VNNI substitution, including routes (54 L41 and 49 L42 distinct experts).
However CPU reference h42 differs from captured CUDA h42 by up to 4.0;
greedy labels agree in this one probe. Native-source bitwise certification
remains OPEN. See `artifacts/d2_cpu_tail_pilot.json`; do not use tolerance or
greedy agreement to close that gate. Production D2/H2 latency remains open.

Seven-domain real-student evaluation completed: all exact GPU replay gates pass,
but hold/open-loop/recursive K8 modes all yield E[A]=1 at K8/K16/K32
(zero speculative tokens accepted). This tiny PCA pilot fails; synthetic D1-B
robustness must not be substituted for actual student acceptance. Expand
distinct training prompts and revisit token-conditioned capacity before P3.

## Continuation evidence — SCALE700, 2026-09-16

Fresh SCALE-DEV2 paired ledger HIST1_210 → HIST1_700:
G=11, L=2, ΔH=+9, S1 40/70 → 49/70. Strong S30 PASS. Measured
ledger string `STRONG_PASS_SCALE700_AUTHORIZED` is the label
from `d3_scale_ledger.py`; SCALE700 already ran, so the
authorized next scale step is SCALE1000 (new fresh 70-prompt
holdout first). Capability E[A]≥4 still OPEN (authority-set
mean 2.343). See `artifacts/session_d3_scale700_verdict.md`.
Do not treat leftover ledger wording as a next-step.

## Continuation evidence — SCALE1000 setup, 2026-09-16

SCALE-DEV3 frozen (70 prompts, IDs 127-136, val-only).
Manifest sha256 41c5b929560e4605…18eacf. 301 train additions
frozen (IDs 137-179). Combined capture manifest 371 prompts.
Pipeline armed; phase A capturing. See
`artifacts/session_d3_scale1000_setup.md`. Do not edit
SCALE-DEV3 after baselines.

## Continuation evidence — SafeCapture resume, 2026-09-17

Host-RAM OOM (10:46) killed `omp` and the recorder (pid
3888476). Pre-OOM captures: 155 (70/70 val + 85/301 train),
all schema-valid (bitwise re-keys proven for 2 prompts).
8-prompt soak epoch passed cleanly; supervisor now owns
worker recycling. No SCALE1000 recipe change. See
`artifacts/session_d3_safecapture_resume.md`.
