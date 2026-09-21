# Graph Report - MosaicVM  (2026-09-11)

## Corpus Check
- 227 files · ~205,292 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2172 nodes · 5176 edges · 134 communities (119 shown, 8 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 179 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- pack_mxfp4
- _synthetic_capture
- build_tree
- _xsimt_triton_kernel
- modal_app.py
- layers.py
- tiny_causal.py
- true_p2_mixed.py
- mxint.py
- test_dspark_tree.py
- dsa.py
- true_p2_sweep.py
- test_qpa_mma.py
- skinny_fp4.py
- test_exact_dsa_decode_parity.py
- window_stats
- qpa_mma.py
- streaming_model.py
- p2_exact_dsa.py
- xsimt.py
- test_track_h_hybrid.py
- test_dspark_wire.py
- w4a16_grouped.py
- V4Config
- ExactAttention
- pack_w4
- ptx_jit.py
- validate_exact_model.py
- snapshot
- dspark_wire.py
- floor_b_moe.py
- stage1_train.py
- test_grouped_imma_parity.py
- d3_bridge_eval.py
- source_tail.py
- config.py
- b4o_analysis.py
- DSparkAttention
- d1b_tail_robustness.py
- d3_h42_eval.py
- DSparkStage
- validate_dsa_real_weights.py
- MosaicVM Continuation Contract (AGENTS.md)
- d2e2b_per_case_ablation.py
- precompute_freqs_cis
- ShardIndex
- cast_exact_attention
- DenseMoELayer
- C192WorkingSet
- nanospark_b3s_rfix.py
- recurrent_drafter.py
- compute_floor.py
- test_dsa_sparse_fused.py
- pack_mxfp4_nk
- mxfp4_decode.py
- d3_v1.py
- expert_entropy.py
- stage05_heads.py
- d3_v1_eval.py
- HashRouter
- BudgetExceeded
- StreamLinear
- dsa_layers.py
- d2e2_hybrid_tail.py
- run
- d2_cpu_expert.py
- fused_moe_mxint
- warmup_steady
- skinny_race.py
- vnni_kernel.c
- load_captures
- stage1_record.py
- vnni_bench.py
- fetch_core.py
- residency_oracle.py
- route_overlap_3060.py
- vnni_v4_race.py
- cpu_expert_bench.py
- dspark_measure.py
- moe_ops.py
- read_roofline.py
- _sparse_attn_fused
- SourceTail
- belady_bound.py
- branchpack_oracle.py
- core_inventory.py
- transport.py
- layout_assignments
- d3_r2_teacher_audit.py
- _FakeSlice
- mlp
- hc_split_sinkhorn
- fp8_linear
- p3_deadline_sim.py
- core_classify.py
- d2_tail_inventory.py
- dspark_recursion_diag.py
- nanospark_evidence.py
- phase25_report.py
- routing_coverage.py
- tau_h_analysis.py
- StubRunner
- _fused_fp8_linear
- dsa_bench.py
- ff_ref_hiddens.py
- ff_runtime_probe.py
- ff_runtime_probe2.py
- ff_runtime_probe3.py
- g2_certify_eval.py
- phase29r_report.py
- phase02_report.py
- phase210_report.py
- phase210x_report.py
- phase26_report.py
- phase27_report.py
- phase28_report.py
- phase29_report.py
- route_capture_revalidate.py
- exact_target
- hotbank_study.py
- cutlass_grouped.py
- marlin_moe.py
- _v5_partial
- rebuild_core_hub.py
- mosaicvm
- _skinny_grouped_kernel
- .prefill

## God Nodes (most connected - your core abstractions)
1. `fused_moe_mxint()` - 83 edges
2. `V4Config` - 78 edges
3. `pack_mxfp4_nk()` - 53 edges
4. `ExactAttention` - 49 edges
5. `warmup_steady()` - 47 edges
6. `_allocate_packed4_experts()` - 44 edges
7. `snapshot()` - 43 edges
8. `_allocate_mxint_experts()` - 39 edges
9. `DSparkStage` - 36 edges
10. `_golden_windows()` - 35 edges

## Surprising Connections (you probably didn't know these)
- `resident_core_check()` --calls--> `_check()`  [INFERRED]
  modal_app.py → tests/test_grouped_imma_parity.py
- `test_backend_mix_queues()` --calls--> `_backend_mix()`  [EXTRACTED]
  tests/test_wide_verify_stats.py → mosaicvm/bench/wide_verify.py
- `test_bf16_storage_bitwise_identical()` --uses--> `V4Config`  [INFERRED]
  tests/test_dsa_bf16_storage.py → mosaicvm/model/config.py
- `test_g3_long_history_selection_output()` --uses--> `V4Config`  [INFERRED]
  tests/test_dsa_sparse_fused.py → mosaicvm/model/config.py
- `test_g4_g5_chunk_decode_lifecycle_fused()` --uses--> `V4Config`  [INFERRED]
  tests/test_dsa_sparse_fused.py → mosaicvm/model/config.py

## Import Cycles
- None detected.

## Communities (134 total, 8 thin omitted)

### Community 0 - "pack_mxfp4"
Cohesion: 0.15
Nodes (18): _f32_to_e2m1(), _floor_e8m0_scale(), load_expert_mxfp4(), load_expert_mxfp4_prefix(), pack_mxfp4(), _pack_uint4(), Tensor, Source-preserving MXFP4 pack (E2M1 data + E8M0 scale), not INT4. The Exact-30… (+10 more)

### Community 1 - "_synthetic_capture"
Cohesion: 0.07
Nodes (32): expert_frequency_analysis(), extract_expert_set(), full_frontier_analysis(), _gini(), neighboring_overlap(), §39 canonical NEXT — real ≥256-position route-capture pipeline. Runs on the…, Sanity gates per §9. Returns {ok, errors}., Extract the set of expert IDs from a single routing record. (+24 more)

### Community 2 - "build_tree"
Cohesion: 0.08
Nodes (32): _build_bestfirst(), _build_levelwise(), build_tree(), d3_real(), _dict_bias(), evaluate(), markov_candidates(), node_count() (+24 more)

### Community 3 - "_xsimt_triton_kernel"
Cohesion: 0.67
Nodes (3): constexpr, jit, _xsimt_triton_kernel()

### Community 4 - "modal_app.py"
Cohesion: 0.06
Nodes (49): function, local_entrypoint, capture_routes(), capture_routes_local(), compute_floor(), download_model(), dsa_proof(), golden_reference() (+41 more)

### Community 5 - "layers.py"
Cohesion: 0.11
Nodes (13): ExactAttentionAdapter, Bridge ExactAttention to the eager Attention interface ([b, t, d]).…, Reset decode position AND all session state (ring, compressed history,…, Attention, ExpertMLP, hc_split_sinkhorn(), HyperConnections, V4 layer implementations (Phase 1 correctness path). Attention uses the DSA… (+5 more)

### Community 6 - "tiny_causal.py"
Cohesion: 0.07
Nodes (37): fixed_rollout(), kv_rollout(), main(), Track A1.5: E[A] invariant fix + KV rollout parity certification. 1. E[A] must…, Corrected naive rollout (semantic reference)., Clean single-pass KV-cached rollout. Processes the seed at position 0, then…, main(), Track A1: TF/FR h2 path-equivalence harness (roadmap §16-21). On h1-correct… (+29 more)

### Community 7 - "true_p2_mixed.py"
Cohesion: 0.15
Nodes (22): compute_layer_fused(), compute_layer with all 8 core GEMMs fused-decoded from FP8., build_shared(), main(), NEXT 1+2: matrix-selective single-call hybrid — parity + K56 golden timing. gu…, One deterministic source -> (packed4 bank, hot8 gu codes, hot8 down codes)., _allocate_hybrid_experts(), One pass over random NK filling BOTH the Hot8 and packed4 banks. (+14 more)

### Community 8 - "mxint.py"
Cohesion: 0.07
Nodes (63): decode_e4m3(), float8_e4m3fn or uint8-viewed e4m3 -> fp32., _as_u8(), _cdiv(), decode_activation_int(), _dp4a_grouped_src(), _dp4a_src(), _get_fn() (+55 more)

### Community 9 - "test_dspark_tree.py"
Cohesion: 0.10
Nodes (26): accepted_prefix(), analyze(), d0_sampled(), d1_greedy(), d2_coverage(), d3_score_mass_surrogate(), _depth_stats(), DSpark analysis from full-stream draft records — POST-AUDIT semantics. Operates… (+18 more)

### Community 10 - "dsa.py"
Cohesion: 0.09
Nodes (32): act_quant_sim(), apply_rotary_emb(), _bits_to_f32(), compress_topk_idxs_static(), dequant_scale(), dequant_weight(), fp4_act_quant_sim(), hadamard() (+24 more)

### Community 11 - "true_p2_sweep.py"
Cohesion: 0.12
Nodes (39): build_block(), main(), Track P3: REAL C192 streaming implementation benchmark. Implements the roadmap…, alloc_workspace(), decode_layer(), load_core_resident(), Phase 2.11C2: source-FP8 core execution + T_core(K) + VRAM census. Production…, All 43 layers FP8-resident + BF16 small tensors + LM head. (+31 more)

### Community 12 - "test_qpa_mma.py"
Cohesion: 0.13
Nodes (21): concat_gate_up(), pair_gate_up(), _perm(), Tensor, Source-preserving gate/up output-row pairing. Permute output rows: g0..15,…, n must be even; n/2 output rows per matrix, n total after concat., Permute output rows: g0..15, u0..15, g16..31, u16..31, ... packed_* [N, K/2],…, Bit-exact inverse of pair_gate_up. packed [2N,K/2] -> (g, sg, u, su). (+13 more)

### Community 13 - "skinny_fp4.py"
Cohesion: 0.18
Nodes (18): fused_moe_skinny(), _grouped_skinny_gemm(), last_launch_count(), _lut_tensor(), pack_gate_up_mxfp4(), Tensor, Skinny SIMT MXFP4 GEMM: M=1-8, no tl.dot, no BLOCK_M=16 padding. Ampere…, Single-expert skinny MXFP4 GEMM. No tl.dot, no BLOCK_M=16 padding. (+10 more)

### Community 14 - "test_exact_dsa_decode_parity.py"
Cohesion: 0.11
Nodes (38): _indexer_case(), parametrize, skipif, §81 NEXT B — long-history top-512 selection validation (AGENTS §27, §52). The…, Production bf16 dtype flow on both sides: set overlap per row., Prefill-time indexer selection on both implementations at context S., test_top512_selection_bitwise_fp32(), test_top512_selection_overlap_bf16() (+30 more)

### Community 15 - "window_stats"
Cohesion: 0.36
Nodes (6): Routing truth for one window: expert_ids [positions][top_k]., window_stats(), test_backend_mix_queues(), test_window_stats_basic(), test_window_stats_flat_reuse(), test_window_stats_single_token()

### Community 16 - "qpa_mma.py"
Cohesion: 0.11
Nodes (27): Exact-30 production MoE entry point. Gated on ``MOSAICVM_EXACT_TARGET=1``.…, e8m0_to_fp32(), float8_e8m0fnu or uint8 -> fp32 power-of-two., e4m3_lut_f32(), _as_u8(), _cdiv(), fused_moe_qpa(), _gather() (+19 more)

### Community 17 - "streaming_model.py"
Cohesion: 0.08
Nodes (30): _try_load_checkpoint_experts(), Capture or replay golden MoE routes. Never fabricate from weights+token-ids., run(), inspect_checkpoint_expert_layout(), If MOSAICVM_MODEL_ROOT is set, inspect one expert w1.weight + .scale. Does not…, HashMoELayer, MoE layer with hash routing (the num_hash_layers=3 target layers)., Tensor (+22 more)

### Community 18 - "p2_exact_dsa.py"
Cohesion: 0.15
Nodes (30): compute_layer(), Identical op sequence to core_timing.py per-layer components., main(), §86 NEXT 0 — memory-accounting audit (AGENTS §54–56). FINAL FORM. Resolves the…, smi_used_free(), stage(), build_dsa_modules(), certify() (+22 more)

### Community 19 - "xsimt.py"
Cohesion: 0.11
Nodes (30): _load_roofline(), _packed_bytes_nk(), Phase 2.7 kernel race: native XSIMT/QPA vs Triton prototypes vs grouped INT4.…, run(), _try_tinygemm(), prepack_qpa(), Tensor, qpa_mma_gemm() (+22 more)

### Community 20 - "test_track_h_hybrid.py"
Cohesion: 0.50
Nodes (3): Event, CPU-only regression for H1 task transport and executed-work accounting., test_h1_gu_accounting_and_separate_request_result_queues()

### Community 21 - "test_dspark_wire.py"
Cohesion: 0.10
Nodes (17): dequant_fp4(), dequant_fp8(), MoEDQ, FP8 [out, in] + scale [out, in/128] (or [out/128, in/128]) -> bf16., Packed FP4 int8 [out, in/2] + scale [out, in/32] -> bf16 [out, in]., Kit-exact MoE for one MTP stage (learned router)., SwiGLUExpert, _attn_tensors() (+9 more)

### Community 22 - "w4a16_grouped.py"
Cohesion: 0.10
Nodes (33): _capture(), _pool(), CUDA Graphs for the MoE-only K-set. Eager stays if capture fails., run(), _backend_sequential(), x [M, H], expert_ids [M] -> y [M, H]. Unique-expert loop, one w4a16_gemm per…, w4a16_grouped_moe_sequential(), emit_tiles() (+25 more)

### Community 23 - "V4Config"
Cohesion: 0.15
Nodes (18): calibrate(), Calibrate mean AED k on router logits (no full-model forward)., V4Config, Tensor, scoring_func=sqrtsoftplus per checkpoint config., Linear + sqrtsoftplus scores + noaux_tc top-k, always bf16 (doc §22)., Router, _router_scores() (+10 more)

### Community 24 - "ExactAttention"
Cohesion: 0.09
Nodes (26): _clone_weights(), _full_recompute(), main(), Tensor, §74 NEXT 1 — full-model exact-DSA decode soak (AGENTS §51–52). State-lifetime /…, Fresh-module full-prefix recompute of positions [0, upto)., Single-processing lifecycle: every position is consumed by the soak module…, soak_layer() (+18 more)

### Community 25 - "pack_w4"
Cohesion: 0.09
Nodes (34): Linear, measure_grouped_routed(), _pack_pool(), Oracle speculative-verifier ceiling. Perfect residency, no trained drafter., One w4a16_gemm (or bf16 GEMM) per unique expert, not K independent GEMVs., run(), dequant_w4(), pack_gate_up() (+26 more)

### Community 26 - "ptx_jit.py"
Cohesion: 0.14
Nodes (29): _occupancy(), Driver occupancy + resource introspection for KSplit variants., run(), _as_grid(), compile_cuda(), compile_mma_zeros(), _drv(), _ensure_ctx() (+21 more)

### Community 27 - "validate_exact_model.py"
Cohesion: 0.06
Nodes (48): dequant_fp4(), fit_linear_weight(), maybe_unpack_fp4(), Tensor, FP8/FP4 block-scaled dequant (checkpoint quantization_config)., Packed FP4 E2M1 [rows, cols/2] uint8 + E8M0 scale [rows, cols/32] -> bf16…, Coerce a packed/transposed expert weight into [rows, cols] bf16., Two fp4 values per uint8 along the last dimension. (+40 more)

### Community 28 - "snapshot"
Cohesion: 0.14
Nodes (20): _bench(), Native five-token DSpark kill gate — RTX 3060 K1–K6 verifier cycle (roadmap…, run(), ab_interleave(), classify(), Thermal measurement contract utilities (AGENTS.md §5). Every GPU benchmark must…, Interleaved A/B (ABAB...) under the §5.2 contract. Returns medians and delta =…, NVML snapshot: temperature, clocks, power, limits. Falls back to {'nvml':… (+12 more)

### Community 29 - "dspark_wire.py"
Cohesion: 0.14
Nodes (19): log(), main(), DSpark recursive-speculation experiment (roadmap §41-42, decisive). QUESTION…, load_mtp_tensors(), log(), main(), DSpark Stage-2: wire the drafter from the OBSERVED Stage-1 interface. Runs on…, Extract all mtp.* tensors from the HF shards; organize per stage. (+11 more)

### Community 30 - "floor_b_moe.py"
Cohesion: 0.11
Nodes (29): _allocate_experts(), _allocate_mxfp4_experts(), _allocate_nk_experts(), _charge_pool(), _count_launches(), pack_pool(), Tensor, Floor B: one V4 MoE layer, distinct W4 experts resident. (+21 more)

### Community 31 - "stage1_train.py"
Cohesion: 0.13
Nodes (16): ArchA, ArchB, ArchC, full_eval(), get_src_hc(), load_head(), load_split(), main() (+8 more)

### Community 32 - "test_grouped_imma_parity.py"
Cohesion: 0.18
Nodes (16): exact_moe(), Exact-30 MoE: native grouped XSIMT+QPA. Requires exact_target()., _banks(), _check(), _exact_target(), fixture, parametrize, Permanent regression coverage for GROUPED_IMMA_TILE_BUG (AGENTS §10). The 2.9R… (+8 more)

### Community 33 - "d3_bridge_eval.py"
Cohesion: 0.18
Nodes (17): BridgeNet, build_examples(), load_a2(), load_bridge(), main(), R2-5 evaluator: closed-loop Decision Bridge proposer. Production loop (roadmap…, load_embed(), load_head() (+9 more)

### Community 34 - "source_tail.py"
Cohesion: 0.18
Nodes (12): CensusLinear, D2-COMPLETE-TAIL (roadmap S44-S46): complete L41/L42 draft-tail wall time at K…, Wraps an expert linear; counts calls/rows; optionally runs on CPU via VNNI., main(), D2-E1: CPU VNNI expert vs source GPU expert parity on REAL routed activations…, Pinned source L41/L42 replay; correctness harness, not production timing., library(), linear() (+4 more)

### Community 35 - "config.py"
Cohesion: 0.12
Nodes (14): _bench_component(), _load_layer(), Phase 2.11C/D: exact core execution + T_core(K,C) + VRAM census. Loads FP8/BF16…, Load one layer's core tensors, dequantizing FP8→BF16., run(), Authoritative V4 config loader. The checkpoint's config.json on the volume…, V4Block, no_grad (+6 more)

### Community 36 - "b4o_analysis.py"
Cohesion: 0.15
Nodes (19): coverage_ladder(), ea_from_ladder(), internal_count(), load_ranks(), main(), merge(), node_count(), per_ladder_depth() (+11 more)

### Community 37 - "DSparkAttention"
Cohesion: 0.15
Nodes (7): DSparkAttention, Lin, FP8 linear dequantized to bf16 at load, on `device`. If `scale` is None the…, Kit-exact DSparkAttention in bf16. Ring KV cache (WINDOW) stores the TARGET's…, RMSNorm, TestDSparkAttention, TestLin

### Community 38 - "d1b_tail_robustness.py"
Cohesion: 0.17
Nodes (16): main(), prefix_metrics(), prompt_mean_ci(), Track D1-B: L41/L42 tail robustness to approximate h40 trajectories. THE…, K uncertain draft decisions, excluding exact root; prefix+1 emission. Input…, Percentile bootstrap over independent prompts, never sliding windows., Reset all nested source attention AND indexer compressor buffers., reset_decode_state() (+8 more)

### Community 39 - "d3_h42_eval.py"
Cohesion: 0.18
Nodes (14): load_student(), main(), D-race evaluator for direct-h42 drafters (roadmap S34-S35, S39-S41): the…, student_delta(), load_student(), main(), no_grad, R2-1 (roadmap S31): downstream teacher dataset generation. For every TRAIN… (+6 more)

### Community 40 - "DSparkStage"
Cohesion: 0.22
Nodes (6): DSparkStage, One MTP stage: HC attn + HC MoE (kit Block + DSparkBlock)., Kit DSparkBlock.forward_embed. main_hidden [1, S, 3*dim]., Kit DSparkBlock.forward_head (last stage). Kit hc_head: hc_head_fn [hc_mult,…, _stage_tensors(), TestDSparkStage

### Community 41 - "validate_dsa_real_weights.py"
Cohesion: 0.13
Nodes (6): lc(), dq_fp8(), get32(), load_real(), §24 IMMEDIATE NEXT — real-weight DSA numeric validation (authoritative bytes).…, Dequantize layers.L.attn.<proj>.{weight,scale} -> bf16 [out, in].

### Community 42 - "MosaicVM Continuation Contract (AGENTS.md)"
Cohesion: 0.18
Nodes (11): 1,048,576-Token Exact Context Target, MosaicVM Continuation Contract (AGENTS.md), DeepSeek-V4-Flash-0731 (Target Model), E[A] Expected Acceptance Metric, Exact-30 Throughput Target (>=30 tok/s), Exact Full Verifier (Emission Gate), Source Exactness Contract, Future-h40 + Source L41/L42 Tail Architecture (+3 more)

### Community 43 - "d2e2b_per_case_ablation.py"
Cohesion: 0.24
Nodes (6): main(), D2-E2b: per-case delta ablation (roadmap S45/S46, closure of the 4 E1 ULP-…, Pass-through GPU reference except at the target call, where the VNNI output is…, SubstitutingLinear, VNNICompute, xhash()

### Community 44 - "precompute_freqs_cis"
Cohesion: 0.18
Nodes (12): apply_rotary(), precompute_freqs_cis(), Kit-exact YaRN rope (inference_model.py)., Kit-exact rotary on the last dim (complex pairs)., log(), main(), NanoSpark B1+B2: functional sensitivity + margin analysis (§36-37). B1 — Logit-…, log() (+4 more)

### Community 45 - "ShardIndex"
Cohesion: 0.24
Nodes (5): ExpertLRU, Path, Maps parameter name -> (shard file, dtype, shape)., Small LRU over routed-expert tensors charged to the simulated VRAM budget., ShardIndex

### Community 46 - "cast_exact_attention"
Cohesion: 0.14
Nodes (9): cast_exact_attention(), Enforce the source dtype scheme after any module-level .to() cast: bf16 GEMM…, §60-1: side-by-side validation of mosaicvm ExactAttention vs the checkpoint's…, parametrize, skipif, _quantize_checkpoint_representable(), BF16-storage DSA residency mode — bitwise equivalence (AGENTS §54–57). The DSA…, test_bf16_storage_bitwise_identical() (+1 more)

### Community 47 - "DenseMoELayer"
Cohesion: 0.14
Nodes (10): DenseMoELayer, MoELayer, Shared expert + gather/scatter around top-6 routed experts., All-experts-resident variant used for tests and golden parity runs., ExpertSurrogate, Adaptive Expert Collapse: rank-64 surrogate for omitted expert mass., Rank-64: Linear(H, 64) - GELU - Linear(64, H). Forward(x, omitted_mass) ->…, test_moe_aec_zero_omitted_equals_exact() (+2 more)

### Community 48 - "C192WorkingSet"
Cohesion: 0.16
Nodes (9): C192WorkingSet, PinnedExpertPool, Tensor, Slot-mapped VRAM expert cache over mutable bank tensors., Evict the LRU-oldest NON-PINNED resident expert. Invariant: self.lru keys ==…, Pin the demand set; async-load misses (best-effort when the cache is saturated…, Block until all needed residents' transfers completed., Expert ids -> slot ids. One D2H + dict map + one H2D per call. Defensive: any… (+1 more)

### Community 49 - "nanospark_b3s_rfix.py"
Cohesion: 0.17
Nodes (13): Apply rotary to the last ROPE_DIM of x in place-ish (clone)., Inverse rotary on the last ROPE_DIM (4D [B, S, H, D] input)., x [1, block, dim] draft tokens; main_x [1, S, dim] target proj., _rope_inv_last(), _rope_last(), log(), main(), B3-S-RFIX: sequential rollout with committed-KV ring writes. Fixes the Δ>1 ring… (+5 more)

### Community 50 - "recurrent_drafter.py"
Cohesion: 0.17
Nodes (12): evaluate(), head_logits(), load_head(), load_split(), main(), no_grad, Track P: recurrent target-feature drafter (R0/R1/R2). Design: z_0 =…, H [N, K, 4096] -> logits [N, K, V] via frozen head, chunked over flattened… (+4 more)

### Community 51 - "compute_floor.py"
Cohesion: 0.10
Nodes (39): _bench(), evaluate(), _is_3060(), measure_d2d_bw(), measure_dense_attention(), measure_hyper_connection(), measure_routed_experts(), measure_router() (+31 more)

### Community 52 - "test_dsa_sparse_fused.py"
Cohesion: 0.19
Nodes (14): sparse_attn_kernel exact math: gathered-KV online-softmax with the sink as a…, sparse_attn_ref(), parametrize, skipif, Fused Triton sparse-attention kernel — permanent correctness gates (AGENTS §44…, Record the fused-vs-eager speedup at the K112 verification shape (informational…, G1/G2: fused vs eager reference at verification shapes (window + compressed…, G3: full module at a selective context (C/4 > 512) — fused path is active… (+6 more)

### Community 53 - "pack_mxfp4_nk"
Cohesion: 0.13
Nodes (26): act_quant_legacy_phase27(), act_quant_ref(), decode_e2m1_nk(), pack_mxfp4_nk(), Tensor, Source-semantic FP4 layout for Phase 2.7 QPA/XSIMT. Official V4 linears are not…, DeepSeek V4-0731 act_quant: per 128-K block, E8M0 power-of-two scale via IEEE…, Authoritative source-linear: act_quant then per-K32 A @ B.T * sA * sW. x_bf16… (+18 more)

### Community 54 - "mxfp4_decode.py"
Cohesion: 0.22
Nodes (17): decode_e4m3_device(), decode_weight_nk(), e2m1_lut_f32(), exhaustive_e2m1_table(), _fn(), Tensor, Register-exact E2M1 / E4M3 decode for native XSIMT/QPA. Weight: 16 E2M1 codes x…, packed [N,K/2] + e8m0 [N,K/32] -> fp32 [N,K] (E2M1 * scale). (+9 more)

### Community 55 - "d3_v1.py"
Cohesion: 0.13
Nodes (26): build_collector_exposure(), build_exposure(), build_transitions(), main(), make_feat(), no_grad, quick_val(), D3-v1: token-conditioned one-step Delta-h40 student. Roadmap D3-v1 (supersedes… (+18 more)

### Community 56 - "expert_entropy.py"
Cohesion: 0.27
Nodes (12): _duplicate_tile_rate(), _get_range(), _nibble_entropy(), Phase 2.12W-B: exact expert entropy census. Fetches a representative sample of…, E2M1 nibble histogram + Shannon entropy., Fraction of tiles that are exact duplicates of an earlier tile., Entropy of consecutive-byte XOR (captures local redundancy)., run() (+4 more)

### Community 57 - "stage05_heads.py"
Cohesion: 0.23
Nodes (11): load(), main(), G2 calibration: kit-source (Blackwell) vs kit-source (RTX 3060). Compares the…, build_model(), eval_head(), load_split(), Track S Stage-0.5 (roadmap §47-53): trained deep-exit heads. Phase 1 (record):…, X: [N, 4, 4096] fp32 (GPU). Returns top-k accuracy dict. (+3 more)

### Community 58 - "d3_v1_eval.py"
Cohesion: 0.27
Nodes (9): load_student(), main(), D3-v1 closed-loop evaluator: student + real source tail + FP32 head. Implements…, Closed-loop or teacher-forced rollout for one prompt. Returns (predicted_h40…, run_loop(), student_step(), pos_encode(), Tensor (+1 more)

### Community 59 - "HashRouter"
Cohesion: 0.13
Nodes (24): build_capture_harness(), Attach forward hooks to every Router/HashRouter module. Returns (hooks,…, Exact contract part 1: config values match the V4-0731 source semantics.…, validate_exact_v4_config(), grouped_expert_forward(), HashRouter, Hash-routed layers: no bias, real gate-projection scores, literal normalize.…, Dispatch each token to its selected experts (V4-0731 source semantics). FP32… (+16 more)

### Community 60 - "BudgetExceeded"
Cohesion: 0.22
Nodes (5): BudgetExceeded, MemoryBudget, RuntimeError, Bug 6 (mechanism): load_streaming_model treats ram_overflow entries as…, test_ram_overflow_is_not_silent()

### Community 61 - "StreamLinear"
Cohesion: 0.18
Nodes (8): capture_real_calls(), main(), D2-P: adapter-inclusive CPU VNNI expert-path timing (roadmap S38). v2 —…, Load the kit model, run val prompts, capture routed expert calls (prefix,…, Routed-expert linear: packed FP4 + E8M0 scale read from the snapshot shards on…, StreamLinear, FastStreamLinear, Recording-speed expert path: cache the unpacked, scale-pre- applied bf16 weight…

### Community 62 - "dsa_layers.py"
Cohesion: 0.18
Nodes (9): get_freqs_table(), Compressor, Indexer, Exact V4 DSA attention — Compressor, Indexer, ExactAttention. Ported op-for-op…, Top-k compressed-KV selection (source Indexer, prefill path)., Gated pooled KV compression (source Compressor, prefill path). wkv/wgate output…, RMSNormExact, precompute_freqs_cis() (+1 more)

### Community 63 - "d2e2_hybrid_tail.py"
Cohesion: 0.32
Nodes (4): HybridLinear, main(), D2-E2: hybrid-tail parity (roadmap S43-S46). Production topology under test:…, Routed-expert linear executed on CPU via the VNNI adapter; everything else in…

### Community 64 - "run"
Cohesion: 0.24
Nodes (13): _row(), run(), _trace(), _coactivation(), _entropy(), _family_n_unique(), make_trace(), Tensor (+5 more)

### Community 65 - "d2_cpu_expert.py"
Cohesion: 0.18
Nodes (13): main(), Track D1-R: exact L41/L42 tail replay from true h40 states. THE decisive…, main(), Source L41/L42 expert matrix on CPU; reference evidence only, never D2 timing.…, run_expert(), signed_activation_check(), main(), Track S Stage-0 (roadmap §65-67): early-exit layer scan. Zero-training oracle:… (+5 more)

### Community 66 - "fused_moe_mxint"
Cohesion: 0.15
Nodes (25): fused_moe_mxint(), MXINTHybridBank, prepack_expert_bank_mxint(), prepack_expert_bank_mxint_packed4(), MXINT fused MoE: DP4A (M<=crossover) + IMMA (M>crossover) + Hexa (K1).…, Build an MXINTExpertBank from NK packed tensors (concat or split gate/up)., Build an MXINTPacked4Bank from NK packed tensors (no qint_encode_w)., _make_bank() (+17 more)

### Community 67 - "warmup_steady"
Cohesion: 0.08
Nodes (60): main(), Track D: split fused_moe dispatch into GPU work vs launch/sync. Per…, _allocate_mxint_experts(), 256 distinct experts as an MXINTExpertBank, generated in chunks of 32., Named compute floors and the 30 tok/s equation used by every Phase 2.5 JSON., main(), Track D: GU parameter race at the Exact-30 widths (N16-32). Sweeps…, hot8_partial_from_full() (+52 more)

### Community 68 - "skinny_race.py"
Cohesion: 0.29
Nodes (9): _packed_bytes_mxfp4(), _packed_bytes_w4(), Skinny MXFP4 kernel race: skinny_mxfp4 vs grouped_blockm16 (INT4) baseline.…, run(), _try_tinygemm(), dequant_mxfp4(), packed [K, N/2] uint8 + e8m0 [K, N/32] -> [K, N] bf16 via E2M1_LUT * scale.…, A zero-scale block dequantizes to the E2M1 codebook values directly. (+1 more)

### Community 69 - "vnni_kernel.c"
Cohesion: 0.27
Nodes (5): __m256i, hsum8(), hsum8i(), vnni_gemm_packed(), vnni_gemm_scaled2()

### Community 70 - "load_captures"
Cohesion: 0.31
Nodes (11): main(), C2/C3 pre-flight (full contract): no-update free rollout from the ACTUAL…, fit(), load_captures(), main(), predict(), CPU-only future-h40 PCA/ridge pilot; exact source tail remains the evaluator.…, Train-only PCA and per-displacement ridge, returning full-state decoder. The… (+3 more)

### Community 71 - "stage1_record.py"
Cohesion: 0.57
Nodes (5): capture_span(), main(), prompt_splits(), Full four-stream state recorder with explicit teacher provenance. Default:…, test_prompt_split_and_future_state_boundary()

### Community 72 - "vnni_bench.py"
Cohesion: 0.43
Nodes (6): e4m3_decode(), main(), make_inputs(), Track V: exact VNNI packed-FP4 expert kernel — parity + benchmark. Parity:…, fp32 reference with the SAME scale structure: y[m,n] = sum_wb sW[n,wb] *…, reference()

### Community 73 - "fetch_core.py"
Cohesion: 0.39
Nodes (8): _get_range(), Phase 2.11A: acquire exact core tensors locally via HTTP range requests. The…, bytes [start, end] inclusive., # NOTE: safetensors requires contiguous tensor payloads (no gaps);, _revision(), run(), _shard_header(), _url()

### Community 74 - "residency_oracle.py"
Cohesion: 0.33
Nodes (8): _belady(), _block_sequence(), _bytes_for(), _load_corpus(), Phase 2.12A: golden-route residency oracle (byte-aware Belady). Simulates…, Byte-aware Belady VRAM cache + strict-hierarchy RAM warm tier. granularity…, Group positions into verification blocks of width K; each block step is (layer…, run()

### Community 75 - "route_overlap_3060.py"
Cohesion: 0.40
Nodes (5): capture_one_prompt(), Generate n_positions consecutive tokens greedily, capturing routes. Returns…, log(), main(), RTX 3060 overlap corpus vs Blackwell routes. Compares generated token IDs,…

### Community 76 - "vnni_v4_race.py"
Cohesion: 0.60
Nodes (4): main(), make_inputs(), Track V: v4 four-arm race — parity + M=1..8 benchmark. Arms: v3 (int8 hsum),…, reference()

### Community 77 - "cpu_expert_bench.py"
Cohesion: 0.29
Nodes (7): cpu_expert(), e8m0_to_scale(), main(), Track C: exact CPU expert kernel benchmark (Dual-Rail economics). Source-class…, e8m0 byte -> 2^(b-127) exactly., x_q: int8-viewed e4m3 codes [rows, 4096]; x_s: [rows, 32] pow2. p_gu: uint8…, ndarray

### Community 78 - "dspark_measure.py"
Cohesion: 0.43
Nodes (7): inspect(), install_deps(), load_model(), log(), main(), Phase 2.13P-D: actual DSpark/MTP drafter measurement (K_CHAIN, eta_draft). Runs…, Stage-1 inspection: enumerate every possible MTP/DSpark surface. Produces the…

### Community 79 - "moe_ops.py"
Cohesion: 0.25
Nodes (7): grouped_gemm(), _grouped_gemm_kernel(), constexpr, jit, Tensor, Triton grouped GEMM over selected experts — one launch per MoE layer. Phase 1…, y[m,:] = x[m,:] @ w[expert_ids[m]] x: [M, K]; w: [E, K, N]; expert_ids: [M]

### Community 80 - "read_roofline.py"
Cohesion: 0.33
Nodes (8): _kernel(), measure_shape(), _packed_bytes(), Tensor, Phase 2.7 read roofline: cold DRAM, expert reuse, L2 ceiling. Authoritative…, Read nbytes from buf, optionally rotating start by stride each iter., run(), _run_stream()

### Community 81 - "_sparse_attn_fused"
Cohesion: 0.29
Nodes (6): constexpr, jit, Tensor, Fused exact-DSA sparse attention — round-2 Triton kernel (AGENTS §42–46).…, q [s,h,d] bf16, kv [n,d] bf16, topk_idxs [s,t] int32 (-1 masked), sink [h] fp32…, _sparse_attn_fused()

### Community 82 - "SourceTail"
Cohesion: 0.16
Nodes (8): main(), main(), Full CPU tail adapter comparison on captured states; not native certification., Incremental source-tail: prefill once, then step per position. Blocks keep…, FP32 head on one h42 row [4, 4096] -> [V] logits., TailRunner, inference_mode, SourceTail

### Community 83 - "belady_bound.py"
Cohesion: 0.38
Nodes (6): belady(), load_streams(), main(), Track R: Belady-optimal residency upper bound on GENUINE routes. Access stream:…, {cat: [ (token, layer, expert) ... ] in execution order., Returns (hits, misses, per-layer hits/misses).

### Community 84 - "branchpack_oracle.py"
Cohesion: 0.43
Nodes (6): classify(), load_coverage(), oracle_dp(), Phase 2.13P-O: BranchPack accepted-depth oracle (AGENTS.md §28-30). Uses…, DP: best {b_d} allocation maximizing E[A] with sum(b_d) <= n_budget. c_by_b:…, run()

### Community 85 - "core_inventory.py"
Cohesion: 0.48
Nodes (6): bf16_bytes(), fp8_e4m3_block128_bytes(), inventory(), mxfp4_bytes(), Print per-tensor bf16 vs W4 bytes for the resident core. No snapshot needed., w4_bytes()

### Community 86 - "transport.py"
Cohesion: 0.43
Nodes (6): _bench_component(), _bench_h2d(), _bench_nvme(), Phase 2.12W-A: real transport measurement on the target machine. Measures…, Sequential read benchmark., run()

### Community 87 - "layout_assignments"
Cohesion: 0.38
Nodes (5): layout_assignments(), Tensor, GPU-side MoE assignment layout. Expert ids never leave the device., expert_ids [A] int64, A = k*top_k. Returns: sorted_ids [A] expert ids sorted…, test_layout_assignments_counts_and_segments()

### Community 88 - "d3_r2_teacher_audit.py"
Cohesion: 0.67
Nodes (3): main(), R2-1 teacher-dataset audit: canonical, hash-bound, fail-closed. Canonical…, sha_full()

### Community 90 - "mlp"
Cohesion: 0.40
Nodes (5): mlp(), Tensor, log(), main(), Real >=256-position route capture on the Blackwell sandbox (BATCHED). All 7…

### Community 91 - "hc_split_sinkhorn"
Cohesion: 0.47
Nodes (3): hc_split_sinkhorn(), Kit-exact hc_split_sinkhorn (inference_kernel.py lines 391-425)., TestSinkhorn

### Community 92 - "fp8_linear"
Cohesion: 0.30
Nodes (11): _decode_parity_check(), dequant_kernel(), _e8m0_to_f32(), fp8_linear(), Tensor, F.linear(x, decode(w, s)) — GEMM-constant race winner (§57-1). Width dispatch…, Parity of triton dequant vs reference, plus 0xFF-scale census. NaN-aware…, bench() (+3 more)

### Community 93 - "p3_deadline_sim.py"
Cohesion: 0.47
Nodes (5): load_blocks(), main(), Track P3-A: per-layer H2D deadline simulator v2 (cache-aware). Correct model: a…, Warm-start cache (top-C by frequency, from `warm` set) + serialized H2D.…, simulate()

### Community 94 - "core_classify.py"
Cohesion: 0.50
Nodes (4): classify(), Phase 2.11B (§13): classify core tensors and compute B_core-resident. Reads…, Map a tensor name to its residency class (roadmap §13)., run()

### Community 95 - "d2_tail_inventory.py"
Cohesion: 0.60
Nodes (4): inventory(), main(), Path, Read actual local shard headers; no model loading, downloads, or GPU work.

### Community 96 - "dspark_recursion_diag.py"
Cohesion: 0.40
Nodes (3): fs(), Diagnostic: why does dspark_recursion's first draft fail at depth 0? Loads the…, EXACT copy of both scripts' shared forward_spec.

### Community 97 - "nanospark_evidence.py"
Cohesion: 0.60
Nodes (4): log(), main(), NanoSpark evidence phase (roadmap §43-48, §66 items 2-5). Phase A — capture…, _save()

### Community 98 - "phase25_report.py"
Cohesion: 0.60
Nodes (4): _decision(), _load(), Phase 2.5 G1–G4 ladder. Exit 0 always., run()

### Community 99 - "routing_coverage.py"
Cohesion: 0.50
Nodes (4): _coverage(), Routing coverage: expert concentration statistics from synthetic traces. Reads…, Compute coverage fractions for a flat expert-id list., run()

### Community 100 - "tau_h_analysis.py"
Cohesion: 0.60
Nodes (4): hid_map(), main(), Track E: prefix-conditioned hidden envelope tau_h(s). Compares the 3060-kit…, rel()

### Community 102 - "_fused_fp8_linear"
Cohesion: 0.40
Nodes (6): _dequant_block(), _fused_fp8_linear(), constexpr, jit, out = x @ W^T with W decoded from fp8 codes in-register. Per 128x128 quant…, One program = one 128x128 quant block -> one scalar scale. W: uint8 view of…

### Community 103 - "dsa_bench.py"
Cohesion: 0.22
Nodes (7): eager_forward(), Tensor, §60-1: exact DSA attention vs eager stand-in — matched full pipelines. Both…, Eager attention block on the SAME module weights: identical q/kv projections,…, _rms(), load_exact_attention(), Load real weights from the core archive into ExactAttention. `get(name) ->…

### Community 104 - "ff_ref_hiddens.py"
Cohesion: 0.67
Nodes (3): log(), main(), §64 localization — per-layer reference hiddens for the two failing prompts…

### Community 105 - "ff_runtime_probe.py"
Cohesion: 0.67
Nodes (3): log(), main(), 0xFF closure probe: what does the WORKING transformers path actually hold/run?…

### Community 106 - "ff_runtime_probe2.py"
Cohesion: 0.67
Nodes (3): log(), main(), 0xFF closure probe 2: full runtime attention state extraction. For every layer…

### Community 107 - "ff_runtime_probe3.py"
Cohesion: 0.67
Nodes (3): log(), main(), 0xFF closure probe 3: loading introspection. from_pretrained with…

### Community 108 - "g2_certify_eval.py"
Cohesion: 0.67
Nodes (3): bucket_tau(), main(), G2 §38-unit certification evaluator — applies FROZEN spec v1. Reads the frozen…

### Community 109 - "phase29r_report.py"
Cohesion: 0.47
Nodes (5): _load(), _p1_parity_check(), Phase 2.9R go/no-go. One precedence-ordered verdict line., Tiny E=8 fixture: fused_moe_mxint (Hot8) vs fused_moe_qpa numeric parity., run()

### Community 110 - "phase02_report.py"
Cohesion: 0.67
Nodes (3): _load(), Phase 2 go/no-go ladder. Exit 0 always; table contains pass/fail per row., run()

### Community 111 - "phase210_report.py"
Cohesion: 0.67
Nodes (3): _load(), Phase 2.10 WideVerify go/no-go. One precedence-ordered verdict line., run()

### Community 112 - "phase210x_report.py"
Cohesion: 0.67
Nodes (3): _load(), Phase 2.10X width-ceiling go/no-go. One precedence-ordered verdict line., run()

### Community 113 - "phase26_report.py"
Cohesion: 0.67
Nodes (3): _load(), Phase 2.6 report: S1-S4 gates and the Exact-30 decision string. Reads…, run()

### Community 114 - "phase27_report.py"
Cohesion: 0.67
Nodes (3): _load(), Phase 2.7 report. Exit 0 always; last line is exactly one decision string., run()

### Community 115 - "phase28_report.py"
Cohesion: 0.67
Nodes (3): _load(), Phase 2.8 go/no-go. One precedence-ordered verdict line., run()

### Community 116 - "phase29_report.py"
Cohesion: 0.67
Nodes (3): _load(), Phase 2.9 go/no-go. One precedence-ordered verdict line., run()

### Community 117 - "route_capture_revalidate.py"
Cohesion: 0.67
Nodes (3): log(), main(), Re-validate + re-analyze the capture from raw records (no model needed). Fixes:…

### Community 118 - "exact_target"
Cohesion: 0.33
Nodes (5): exact_moe_mxint(), MXINT fused MoE. Gated on exact_target(); QPA remains the Exact default., Path, exact_target(), True unless ``MOSAICVM_EXACT_TARGET`` is explicitly disabled.

### Community 122 - "_v5_partial"
Cohesion: 0.67
Nodes (4): constexpr, jit, _v5_partial(), _v5_reduce()

### Community 132 - "_skinny_grouped_kernel"
Cohesion: 0.67
Nodes (4): constexpr, jit, _skinny_grouped_kernel(), _skinny_m_kernel()

## Knowledge Gaps
- **10 isolated node(s):** `mosaicvm`, `1,048,576-Token Exact Context Target`, `DeepSeek-V4-Flash-0731 (Target Model)`, `E[A] Expected Acceptance Metric`, `Exact-30 Throughput Target (>=30 tok/s)` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 752 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `V4Config` connect `V4Config` to `config.py`, `layers.py`, `dsa_bench.py`, `validate_exact_model.py`, `validate_dsa_real_weights.py`, `cast_exact_attention`, `DenseMoELayer`, `test_exact_dsa_decode_parity.py`, `streaming_model.py`, `p2_exact_dsa.py`, `test_dsa_sparse_fused.py`, `core_inventory.py`, `exact_target`, `floor_b_moe.py`, `ExactAttention`, `pack_w4`, `HashRouter`, `dsa_layers.py`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `fused_moe_mxint()` connect `fused_moe_mxint` to `run`, `test_grouped_imma_parity.py`, `warmup_steady`, `true_p2_mixed.py`, `mxint.py`, `true_p2_sweep.py`, `phase29r_report.py`, `qpa_mma.py`, `p2_exact_dsa.py`, `pack_mxfp4_nk`, `exact_target`, `layout_assignments`, `w4a16_grouped.py`, `validate_exact_model.py`, `snapshot`, `floor_b_moe.py`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `act_quant_ref()` connect `pack_mxfp4_nk` to `d2_cpu_expert.py`, `source_tail.py`, `fused_moe_mxint`, `mxint.py`, `true_p2_sweep.py`, `qpa_mma.py`, `xsimt.py`, `validate_exact_model.py`, `floor_b_moe.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `V4Config` (e.g. with `Compressor` and `ExactAttention`) actually correct?**
  _`V4Config` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `ExactAttention` (e.g. with `eager_forward()` and `_clone_weights()`) actually correct?**
  _`ExactAttention` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mosaicvm`, `1,048,576-Token Exact Context Target`, `DeepSeek-V4-Flash-0731 (Target Model)` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `_synthetic_capture` be split into smaller, more focused modules?**
  _Cohesion score 0.06715063520871144 - nodes in this community are weakly interconnected._