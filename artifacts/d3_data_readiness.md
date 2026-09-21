# DATA/D3 readiness audit — 2026-09-07

Evidence: CPU-only local file/schema inspection and recorder/loader tracing. No GPU allocation, training, checkpoint download, or performance measurement. Scope is the configured data locations and workspace tensor files, not an exhaustive search of every host mount.

## Available data

`artifacts/stage1_prompts.json` contains **126 distinct IDs and 126 distinct normalized prompt texts**. All IDs occur in exactly one split; train/val/test ID intersections are empty.

| Split | Distinct prompts | Per domain | Existing `/tmp/stage1_data` captures |
|---|---:|---:|---:|
| Train | 84 | 12 | 0 |
| Validation | 21 | 3 | 0 |
| Test | 21 | 3 | 0 |

Seven domains: coding, math, agent, convo, retrieval, longform, json. `/tmp/stage1_data` and `/tmp/tiny_causal_t1.pt` are absent. Historical reports state that 126 prompts were once recorded, with 1,076/267/285 full-K training/validation/test windows; those reports do not establish present tensor availability. The workspace `.pt` inventory contains no replacement Stage-1 dataset or T1 checkpoint.

At inspection, `artifacts/d1b32_test_captures/coding_16.pt` is the sole D1-B32 capture: **one test prompt, zero train/validation prompts**. Its actual schema is:

- `h40`, `h42`: `[40,4,4096]`, BF16; `S=8`.
- `tokens`: 41 IDs, including the label after the final captured state.
- `routes41`, `routes42`: `[40,6]`, int64.
- Category, prompt, explicit `split='test'`, pinned snapshot revision, `h42_bit_exact=True`.

This is held-out correctness/evaluation material and cannot enter training. More test captures do not remove the training-data gap.

The local source/G2 corpora are not substitute full-state datasets. Inspected source, calibration, and certified-32 coding files contain `hiddens` entries with `h_mean[4096]` FP16. `source_corpus_gen.py` explicitly averages the four streams (`t[0,-1].float().mean(0)`), so their full `[4,4096]` h40 cannot be recovered. They also contain repeated prompts across source/reference/implementation captures; these are not independent examples. RFIX proposal records belong to the historical speculative-state experiments, not a demonstrated full future-h40 training corpus.

## Recorder provenance and alignment

`stage1_record.py` records layer-index 40 output as `[positions,4,4096]` FP16, together with exits 36/38/42 and generated tokens. It substitutes **FastStreamLinear** for routed experts. This dequantizes weights and activations into BF16 and performs a different GEMM accumulation path from the source-class shim. Its docstring's assertion that the deviation is immaterial is not D3 evidence; classify any reused captures as approximate-teacher data until compared against the reference. FP16 serialization is another conversion and should not be used for new exact BF16 captures.

The existing recorder uses `BATCH=12`, despite its stale introductory batch-of-16 text. A prompt's recorded length is `max_prompt_length_in_batch + 40 - its_prompt_length`, so the corpus is not uniformly 40 generated positions. It saves neither split nor revision/hash/provenance nor prompt token IDs; split association currently depends on the external filename/manifest. It also lacks the corrected nested per-chain reset used by D1-B v2. Do not use this recorder unchanged to create exact D3 data.

Its row `states[40][p]` is the state **before emitting** `gen[p]`: absolute input position `S-1+p`, next token `gen[p]`. Thus future-state training from this schema would use current state row p and targets rows `p+1..p+K`, requiring `p+K < number_of_states`. Copying the old token-loader bound would admit a terminal off-by-one, because its K token labels start at p while K future states start at p+1.

The corrected D1-B capture stores all prefix and continuation positions: `h40[t]` consumed `tokens[t]` and predicts `tokens[t+1]`. For the committed boundary `t=S-1`, the exact root is `tokens[S]`; K uncertain future-state targets are `h40[S:S+K]`, whose next-token labels are `tokens[S+1:S+K+1]`. Preserve the terminal label. The interface is **16,384 values per position**, not 4,096.

## Leakage and evaluation gates

Split prompts before producing windows, fitting normalization/PCA, or selecting a model. The manifest has no normalized-text duplicates, but this is not a semantic-duplicate audit and does not guarantee independence from other historical corpora. Preserve prompt IDs, split, text hash, positions, source revision and capture implementation hash in every new file. Keep all descendants/crops of a prompt in its original split. Fit transforms on train only; tune on validation; retain test for final reporting.

`stage1_train.py` and `t1_multistate.py` predict token/head-space outputs, not full future h40 for the real tail. Their old branches remain closed. Their sliding windows are training samples, not independent production cycles. A new D3 student must be selected through the source L41/L42 + source head, using one exact committed boundary per held-out prompt and reporting K separately from verifier width N. Hidden-state loss/cosine alone cannot certify it.

## Minimal next action

1. Reuse the repaired source setup, full nested reset, trajectory capture and exact tail replay from D1-B v2 in a **separate train/validation capture entry point**. Do not relax D1-B's test-only guard or repurpose its existing captures. Use durable `artifacts/d3_*` storage, BF16 full h40, bounded expert cache (`SCG_CACHE_CAP=24`), and source-class reference provenance. Save all prefix states so tail history can be reconstructed; include terminal next-token ID.
2. First capture `coding_01` (train) and `coding_13` (validation) for eight future positions to check schema, alignment and replay. Then collect at least one train and one validation prompt per domain, followed by the available 84/21 split. These are engineering pilots; even all 126 current prompts are not a large diverse training corpus.
3. Build only the K4/K8 future-h40 training path after that data exists. Learn from train prompts, evaluate actual predicted residuals through the real source tail on validation, and preserve the test-only captures for a later held-out check. There is currently **no verified locally usable full-h40 training/validation capture set** at the inspected paths. Neither synthetic residuals nor historical token-proposer checkpoints fill that gap.

No production-student acceptance, tail latency, or Exact-30 result is claimed by this audit.

## Continuation completed — 2026-09-08

The availability statements above describe the initial audit. Seven D1-B32
test captures now exist. A new source-class recorder option also produced
`artifacts/d3_source_pilot/coding_01.pt` (train) and `coding_13.pt` (validation),
with nine generated labels each. Full BF16 h40/h42, prefix history, terminal
labels, split/category, revision, prompt hash and source script hash are saved.
Both pass saved state/token alignment, dtype, shape and finiteness checks in
`d3_source_pilot_validation.json`. This is two engineering captures, not a
useful student training set or a source-tail replay certification of these files.

The recorder retains its approximate default but labels its provenance.
Source mode uses all source experts and rejects held-out test IDs. Split
overlap/text duplication is checked; incompatible existing source captures
are rejected rather than silently resumed. KV/compressor state resets between
batches, and captures now have exactly N_GEN generating rows.

Reproduce the pilot (source snapshot archived as `d3_source_pilot.source.py`):

```bash
S1_SOURCE=1 S1_CATS=coding_01,coding_13 S1_N_GEN=9 S1_BATCH=1 \
S1_OUT=artifacts/d3_source_pilot SCG_CACHE_CAP=24 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-cu/bin/python -u mosaicvm/bench/stage1_record.py
```

Next capture separate train/validation prompts across all seven domains and
more boundaries. K8 needs at least nine generated-state rows (root + eight
future rows). Fit normalization and any predictor only on training prompts.
The DATA1000 and real D3 student gates remain open.
