# D3 first cross-domain baseline and D2-E continuation — 2026-09-08

This supersedes the pilot-only status, without changing historical artifacts or the frozen exactness contract. No production throughput or engineering completion percentage is established.

## DATA

`d3_multidomain_tranche1/` contains 14 source-class BF16 captures: seven training prompts (`*_01`) and seven validation prompts (`*_13`), one of each per domain. Each has 33 generated-state rows, providing an exact root and 32 future states. All captures passed the student loader's provenance, full-prefix, dtype, finiteness, split and terminal-label checks. The collector completed all 14 prompts. The selection's normalized exact-text overlap with protected G1/G2 corpora is zero; semantic near-duplicate coverage is not exhaustive. No D1-B test capture was used for training.

This remains a tiny learning-curve tranche, not the requested eventual 1000+ distinct-prompt corpus.

## D3 baseline scope

`mosaicvm/bench/d3_student.py` fits rank-32 PCA and per-displacement ridge maps using training captures only. The orthogonal root component is held fixed. At K32 there are only seven training pairs; at K1 there are 224 overlapping training pairs. These do not constitute 224 independent prompts.

The evaluated modes are constant-root hold, learned open-loop K32, and recursive K8 chunks using predicted states. Recursive chunks have no token or tail feedback. This is an engineering baseline, not the planned token-conditioned/chunk-feedback architecture comparison. Validation target states are used only by diagnostics and the replay evaluator.

Training and evaluation are reproducible with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-cu/bin/python -m mosaicvm.bench.d3_student --data artifacts/d3_multidomain_tranche1 --out artifacts/d3_student_tranche1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-cu/bin/python -m mosaicvm.bench.d3_tail_eval --predictions artifacts/d3_student_tranche1 --output artifacts/d3_tranche1_tail.json
```

The source-tail evaluator requires exact h42 raw-bit replay and all exact root/future greedy labels before scoring a prompt. It runs the actual source L41/L42 blocks and source HC head/norm cast boundaries with the FP32 projection. Its kernels are reference torch shims. Inputs follow the reference path; acceptance is valid through the first rejection, while later marginal Top-k is explicitly true-path conditioned. K excludes the exact root; target verification width N remains unmeasured.

Mean first-step relative L2 is 0.996 for hold and 0.883 for the learned models. At K32 it is 1.280 for hold, 1.203 open-loop and 1.132 recursive. Residual lag-1 cosine is approximately 0.695, 0.640 and 0.639 respectively. These are much larger errors than the earlier synthetic 0.1/0.3 probes. Reduced state error is not a student pass.

## D2-E evidence classes

`d2e_matrix_v3.json` records 24 complete real-checkpoint expert cases (L41/L42 experts 0/1, M=1/2/3/4/8/16) on synthetic inputs: 72 linear stages and 24 complete experts match the ordered CPU reference bitwise.

`source_tail.py` now supports replay on CPU or GPU with the same source blocks. `d2_tail_parity.py` compares the full CPU reference tail against its VNNI routed-expert substitution using captured states and actual source routing. This checks the adapter inside the whole CPU tail; it does not make CPU core arithmetic match CUDA.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-cu/bin/python -m mosaicvm.bench.d2_tail_parity --capture artifacts/d3_source_pilot/coding_13.pt --output artifacts/d2_cpu_tail_pilot.json
```

The initial coding smoke matched CPU h42 and routes exactly, but CPU reference h42 differed from the captured GPU result (maximum absolute difference 4.0). No tolerance-based promotion is permitted. Source CPU/native parity must remain open even if greedy labels agree. Neither this replay nor historical unsigned VNNI timing supplies production D2 latency.

## Checks and next decisions

Seven targeted regression checks pass across D1-B geometry/reset, DATA alignment/splits, H1 queue accounting, train-only student prediction and D3 root/terminal/first-rejection scoring. Full historical regression remains required before production integration.

A failing tiny PCA baseline calls for more distinct training prompts and a token-conditioned student, followed by the same held-out source-tail acceptance test. It does not invalidate future-h40 prediction as an architecture. For CPU exactness, isolate the first CPU/CUDA divergence in core/tail arithmetic before making any native certification or production-source timing claim. H2 full verifier traffic and integrated P3/P4 remain open.

## Completed seven-domain acceptance result

All seven validation exact replays passed h42 raw-bit and source-head greedy checks. All three modes rejected at the first uncertain decision on every prompt. At K8, K16 and K32, E[A] = 1.0 (the exact root only), with zero accepted speculative tokens and every survival probability zero. The bootstrap interval [1,1] describes these seven observations; it does not prove population certainty.

| Mode | K32 E[A] | K32 marginal Top1 | K32 marginal Top16 |
|---|---:|---:|---:|
| hold | 1.0 | 0.0000 | 0.0670 |
| open_loop | 1.0 | 0.0000 | 0.0759 |
| recursive_chunk | 1.0 | 0.0045 | 0.0714 |

The learned pilot fails the acceptance gate. Do not infer synthetic Top16 robustness for this actual residual distribution or reopen BranchPack on these results. No integrated verifier or production cycle was executed.

The reproducible CPU tail probe also completed: 20 positions, 54 distinct L41 experts and 49 distinct L42 experts. CPU adapter h42/routes matched bitwise; CPU versus GPU h42 did not; CPU greedy labels matched. Evidence is `d2_cpu_tail_pilot.json`. Source snapshots and hashes are archived in `d3_tranche1_code/`.
