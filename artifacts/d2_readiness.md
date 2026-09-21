# D2 readiness — local audit, 2026-09-06

D2 remains **unmeasured**. The complete checkpoint is locally available; missing weights are not a blocker. The immediate blocker is a production tail execution path and honest representation/cycle accounting.

## Reproduced storage measurement

Run from the repository:

```bash
.venv-cu/bin/python -m mosaicvm.bench.d2_tail_inventory
```

This ran successfully, reading all 48 shard headers through their actual local targets and validating tensor offset bounds plus all 3,072 routed-tail tensors. It neither materializes weights nor launches remote jobs. Output: `artifacts/d2_tail_inventory.json`. This checks storage availability, not checkpoint content hashes or numerical correctness.

| Actual source payload | Bytes | Decimal GB |
|---|---:|---:|
| L41/L42 routed FP4 weights + E8M0 scales | 6,845,104,128 | 6.845 |
| Shared experts | 50,334,720 | 0.050 |
| Attention/DSA/core/router/norm | 263,008,688 | 0.263 |
| Source tail total | 7,158,447,536 | 7.158 |
| Entire checkpoint | 166,878,536,440 | 166.879 |

These are measured storage lengths, **not resident runtime footprints**. Additional representation-derived quantities:

- VNNI v4-B routed bank, expanded int8 weights plus FP32 scales: **14,495,514,624 bytes**. This follows the actual `vnni_gemm_tiled` interface; the approximately 6.85 GB source bank cannot be used as its resident byte count.
- Production FP32 head: **2,118,123,520 bytes** (129,280 × 4,096 × 4).
- Keeping both canonical packed tail and fully expanded VNNI bank plus FP32 head costs **23,458,742,272 bytes**, before core/shared weights, KV/DSA, scratch, student, verifier, allocator, or OS. Avoid assuming this comfortably fits a 24 GB budget.
- At audit time the physical host reported 33,442,811,904 bytes RAM, approximately 17.37 GB available, and approximately 7.75 GB swap in use. This differs from the deployment budget; both must be reported. RTX 3060: 12,288 MiB total, 1,393 MiB used. These are transient snapshots.

Runtime census must include actual storage identity to avoid double-counting aliases: attention compressor KV aliases the attention cache, indexer compressor KV aliases the indexer cache. Include compressor `kv_state` **and** `score_state`, indexer state, RoPE tables, temporary FP32 dequantization, head copies, and cache allocations. Runtime compressor weights may expand BF16 storage to FP32.

## What existing code can and cannot measure

`d1r_tail_replay.py` and `d1b_tail_robustness.py` run checkpoint blocks through `source_corpus_gen.build_kernel_module` and `StreamLinear`. These provide a source-class reference path. Their FP4 shim expands codes to FP32 and uses einsum; they do not run production imma4 or VNNI. Their timings cannot be labeled production D2 latency.

`StreamLinear._materialize` reads safetensors on a cache miss and copies to GPU. Its default 3,000-linear cache may grow beyond a 12 GB card during a full-model run. It does not implement a dedicated RAM-resident tail invariant. Tail-only preload can fit packed storage in isolation, but full verifier/core coexistence is unproven. A full-model reference harness is therefore not a safe production memory census.

The checkpoint `Attention.forward` nonzero-position branch writes one ring slot using `kv.squeeze(1)` (`inference/model.py:535`). It supports serial continuation, not arbitrary K4/K8/K32 continuation tensors. Compressor continuation is likewise one-position. Block/chunk speedups require source-equivalent causal implementation and comparison against serial results; timing a start_pos=0 prefill is not a substitute.

VNNI v4-B exists and has isolated matrix tests (`vnni_v4_race.py`), but this harness uses synthetic inputs and permits a numerical tolerance; it is not itself proof of a full source tail. Integrating quantization, GU, source SwiGLU, DOWN, route weighting, shared expert, and ordered reduction remains necessary for end-to-end CPU tail measurements.

H1 (`track_h_hybrid.py`) is not a usable D2 coexistence oracle. The historical CPU worker repeatedly executed one GU matrix, while its accounting credited GU **plus DOWN**, inflating its displayed full-expert TFLOPS-equivalent by 1.5×. GPU executes a full surrogate block with a resident synthetic bank and no timed production H2D. Its 1.113 s wall remains an observed harness wall; neither 0.246 TFLOPS nor the arbitrary pass lines should be promoted to production throughput. The historical shared request/result queue also permitted the sender to consume its own work message.

The current harness now uses separate request/result queues, validates completed task counts, accounts only executed GU FLOPs, and labels new results `H1_SYNTHETIC_CONTENTION_GU_ONLY` with `production_correct: false`. Misleading GPU-share accounting and arbitrary production pass lines were removed. New runs write `artifacts/track_h_hybrid_v2.json`; the invalid GPU-max/wall-median overlap metric was removed. Historical JSON measurements were preserved; this is a harness repair, not a new timing result. The CPU-only regression (`.venv-cu/bin/python -m pytest -q tests/test_track_h_hybrid.py`) passes and exercises the actual worker loop using small fake-kernel tasks. No GPU contention run was launched during the concurrent D1-B smoke.

## Concrete next execution

1. Keep the measured source inventory above; add a tail-only runtime census using actual L41/L42 buffers and selected expert representation. Count head separately and shared with verifier when physically shared. Preload all packed tail experts in host RAM only after combined budget fits; verify absence of disk/page-fault traffic during timed trials.
2. Reuse valid committed h40 captures and source tail state snapshots. Establish serial continuation at K1/K4/K8/K16/K32; run the repaired isolation/poison checks before timings. Separate initialization, committed-boundary replay, proposal tail, FP32 head, and restoration.
3. Establish production source-equivalent GPU and CPU expert dispatch against reference outputs. Race CPU VNNI, GPU misses/H2D, strategic resident subset, and hybrid on genuine tail routes and per-expert row counts. Record routes, complete GU/SwiGLU/DOWN work, bytes, memory peaks, and CPU/GPU synchronization wall.
4. Implement chunk continuation only after step 2; require exact/reference comparison including compressor boundaries (4 and 128) and ring wrap (128). Until then report K as serial horizon with chunk size 1.
5. Run with corrected H2 using actual distinct experts, H2D, and contended CPU traffic. Measure full cycle rather than adding mutually overlapping medians. Report thermal state and final E[A]/cycle wall; K32's 33/30 = 1.10 s is only a perfect-acceptance ceiling.

Read-only checks confirm `.venv-cu` has torch 2.13.0+cu130, CUDA access, and `float4_e2m1fn_x2`. `modal_app.py` contains remote infrastructure, but no live remote allocation/readiness was verified and no paid job was started. No remote result is needed to establish the local storage inventory.

Evidence labels: inventory = **LOCAL_SOURCE_STORAGE_INVENTORY_NOT_RUNTIME_TIMING**; D1 reference correctness remains its original reference evidence; old H1 = local synthetic contention wall; D2 production tail economics = **OPEN**. No tail latency, Exact-30, or completion percentage is certified by this audit.

## Adapter update — 2026-09-08

A separate source-activation adapter now passes finite-code integer-oracle
checks and complete one-real-expert CPU-reference checks. See
`d2_cpu_adapter_v2.md`. Its derived all-tail prepared weight footprint is
16.106 GB including int32 block sums; historical v4-B latency is inapplicable.
Full-tail native-source parity, source scale edge support, production latency
and coexistence are still open.
