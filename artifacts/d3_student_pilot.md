# D3 first full-state student — 2026-09-08

`mosaicvm/bench/d3_student.py` implements a CPU-only, train-only PCA/ridge
trajectory baseline with all four h40 streams (16,384 values per position).
This is an engineering baseline before a larger student architecture race.

PCA, centering, latent scaling and per-displacement ridge fits use training
prompts only. Training pairs may slide within each training continuation;
validation uses exactly the initial committed boundary once per prompt.
At boundary `S-1`, input is exact h40 at `S-1`, outputs are h40 positions
`S:S+K`, and labels are tokens `S+1:S+K+1`. The exact root is excluded from K.
The loader rejects test captures, split mismatches, provenance/schema failures
and missing terminal labels. It records file hashes and source revision.

The same fitted model produces open-loop K32 and recursively applied K8
chunks, alongside hold-state. Recursive chunks refresh from predicted h40
only, without tail/token feedback or oracle exact-state refresh. Therefore
this is not the future closed-loop chunked-tail architecture. The fitted
subspace changes while the unmodeled orthogonal root component stays fixed.

## Actual completed check

The existing coding-only source pilot trained successfully on coding_01 and
exported predictions for coding_13 at K8, rank 4 to
`artifacts/d3_student_coding_smoke`. Train time was about 0.048 seconds on
CPU with two Torch threads. This smoke capture has one training and one
validation prompt; it cannot select a production student.

Validation relative L2 at horizons 1/8 was approximately 0.937/1.349 for the
student and 0.995/1.341 for hold-state. These are state diagnostics only.
Source-tail acceptance has not been measured for this checkpoint; no D3 PASS
or throughput claim follows. Open-loop and one K8 chunk are identical at K8.

The CPU regression passes: full-state shape/dtype, future-state alignment,
terminal labels, split rejection, train-only fitting and recursively predicted
refresh are checked by `tests/test_d3_student.py`.

## Multi-domain run

Once all fourteen train/validation captures have arrived:

```bash
.venv-cu/bin/python -m mosaicvm.bench.d3_student \
  --data artifacts/d3_multidomain_tranche1 \
  --out artifacts/d3_student_tranche1 \
  --categories coding_01,math_01,agent_01,convo_01,retrieval_01,longform_01,json_01,coding_13,math_13,agent_13,convo_13,retrieval_13,longform_13,json_13
```

`student.pt` saves the train-only checkpoint; `diagnostics.json` records
training pair counts, per-horizon drift/cosine, residual bias/lag-1 cosine,
and CPU prediction latency at K8/K16/K32. Model selection remains pending
real source-tail prefix acceptance and runtime. CPU prediction timing excludes
source tail, verifier and scheduling and is not production throughput.

Each `<category>.predictions.pt` exports `h40_committed` including the exact
root, `predicted_h40` keyed by hold/open_loop/recursive_chunk, diagnostic
`h40_target`, `h42_target`, tokens, S, K, split and provenance. Only the first
two state fields are proposal inputs. True future states are evaluator data,
never prediction inputs. N is left null until the integrated verifier exists.

The source-tail evaluator must reset all nested KV/compressor state, replay
the exact committed prefix, then feed the actual predictions. If it uses
reference input tokens after a mismatch, later Top-k is true-path conditional;
prefix-stop acceptance remains the relevant valid statistic. No approximate
draft token may be emitted without full exact target verification.
