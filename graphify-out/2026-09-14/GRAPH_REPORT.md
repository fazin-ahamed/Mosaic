# Graph Report - MosaicVM  (2026-09-14)

## Corpus Check
- 263 files · ~241,156 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2363 nodes · 5873 edges · 142 communities (122 shown, 13 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 189 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- apply_rotary_emb
- _synthetic_capture
- build_tree
- xsimt.py
- modal_app.py
- V4Config
- tiny_causal.py
- wide_verify.py
- mxint.py
- test_dspark_tree.py
- dsa_layers.py
- true_p2_sweep.py
- validate_exact_layer.py
- floor_b_moe.py
- test_exact_dsa_decode_parity.py
- d3_bridge.py
- layers.py
- matrix_hybrid.py
- p2_exact_dsa.py
- test_constrained_runtime_regressions.py
- compute_floor.py
- test_dspark_wire.py
- cast_exact_attention
- d3_scale_watchdog.sh
- ExactAttention
- w4a16_grouped.py
- ptx_jit.py
- validate_exact_model.py
- warmup_steady
- dspark_wire.py
- test_mxint_epilogue.py
- stage1_train.py
- d2_cpu_expert.py
- pack_mxfp4
- d3_lrdl_eval.py
- TailRunner
- b4o_analysis.py
- DSparkAttention
- test_aec.py
- mxfp4_decode.py
- DSparkStage
- validate_dsa_real_weights.py
- MosaicVM Continuation Contract (AGENTS.md)
- d3_dd1_hist_eval.py
- precompute_freqs_cis
- StreamingLoader
- d2e2b_per_case_ablation.py
- pack_mxfp4_nk
- C192WorkingSet
- nanospark_b3s_rfix.py
- recurrent_drafter.py
- Router
- fused_moe_mxint
- test_grouped_imma_parity.py
- act_quant_ref
- d3_v1.py
- expert_entropy.py
- stage05_heads.py
- test_dsa_sparse_fused.py
- HashRouter
- d3_scale_monitor.sh
- StreamLinear
- transport.py
- d2e2_hybrid_tail.py
- dequant_fp8
- source_corpus_gen.py
- d3_scale_orchestrate.sh
- _golden_windows
- load_captures
- vnni_kernel.c
- d3_student.py
- hw_profile.py
- vnni_bench.py
- fetch_core.py
- residency_oracle.py
- d3_scale700_author.py
- vnni_v4_race.py
- cpu_expert_bench.py
- dspark_measure.py
- moe_ops.py
- validate_dsa_source.py
- dispatch_profile.py
- DeltaNet
- belady_bound.py
- branchpack_oracle.py
- core_inventory.py
- dsa_attention.py
- dsa_bench.py
- d3_r2_teacher_audit.py
- MemoryBudget
- mlp
- hc_split_sinkhorn
- fp8_core.py
- p3_deadline_sim.py
- core_classify.py
- d2_tail_inventory.py
- dspark_recursion_diag.py
- nanospark_evidence.py
- phase25_report.py
- routing_coverage.py
- tau_h_analysis.py
- qpa_mma.py
- dsa_decode_attention
- d3_scale_author.py
- ff_ref_hiddens.py
- ff_runtime_probe.py
- ff_runtime_probe2.py
- ff_runtime_probe3.py
- g2_certify_eval.py
- stage1_record.py
- phase02_report.py
- phase210_report.py
- phase210x_report.py
- phase26_report.py
- phase27_report.py
- phase28_report.py
- phase29_report.py
- route_capture_revalidate.py
- StubRunner
- hotbank_study.py
- cutlass_grouped.py
- marlin_moe.py
- d3_scale_ledger.py
- rebuild_core_hub.py
- mosaicvm
- .forward
- d3_scale700_engine.sh
- d3_scale700_orchestrate.sh
- pack_w4
- d3_scale_teacher.py
- prompts.py
- CensusLinear
- ExpertMLP
- .generate
- capture_routes_local

## God Nodes (most connected - your core abstractions)
1. `fused_moe_mxint()` - 83 edges
2. `V4Config` - 78 edges
3. `load_captures()` - 62 edges
4. `pack_mxfp4_nk()` - 53 edges
5. `load_embed()` - 51 edges
6. `ExactAttention` - 49 edges
7. `load_head()` - 48 edges
8. `warmup_steady()` - 47 edges
9. `_allocate_packed4_experts()` - 44 edges
10. `snapshot()` - 43 edges

## Surprising Connections (you probably didn't know these)
- `resident_core_check()` --calls--> `_check()`  [INFERRED]
  modal_app.py → tests/test_grouped_imma_parity.py
- `test_pack_mxfp4_nk_shapes()` --calls--> `pack_mxfp4_nk()`  [EXTRACTED]
  tests/test_fp4_source.py → mosaicvm/kernels/fp4_source.py
- `test_silu_device_bitwise()` --calls--> `mxint_probe_silu()`  [EXTRACTED]
  tests/test_mxint_epilogue.py → mosaicvm/kernels/mxint.py
- `test_bf16_storage_bitwise_identical()` --uses--> `V4Config`  [INFERRED]
  tests/test_dsa_bf16_storage.py → mosaicvm/model/config.py
- `test_g3_long_history_selection_output()` --uses--> `V4Config`  [INFERRED]
  tests/test_dsa_sparse_fused.py → mosaicvm/model/config.py

## Import Cycles
- None detected.

## Communities (142 total, 13 thin omitted)

### Community 0 - "apply_rotary_emb"
Cohesion: 0.14
Nodes (17): act_quant_sim(), apply_rotary_emb(), fp4_act_quant_sim(), hadamard(), Compressor, Tensor, Chunked prefill (start_pos>0, s>1). Extends the source's prefill/decode state…, x [s, dim] -> emitted compressed rows (or None). Dispatch: prefill… (+9 more)

### Community 1 - "_synthetic_capture"
Cohesion: 0.07
Nodes (32): expert_frequency_analysis(), extract_expert_set(), full_frontier_analysis(), _gini(), neighboring_overlap(), §39 canonical NEXT — real ≥256-position route-capture pipeline. Runs on the…, Sanity gates per §9. Returns {ok, errors}., Extract the set of expert IDs from a single routing record. (+24 more)

### Community 2 - "build_tree"
Cohesion: 0.08
Nodes (32): _build_bestfirst(), _build_levelwise(), build_tree(), d3_real(), _dict_bias(), evaluate(), markov_candidates(), node_count() (+24 more)

### Community 3 - "xsimt.py"
Cohesion: 0.07
Nodes (48): _occupancy(), Driver occupancy + resource introspection for KSplit variants., run(), _load_roofline(), _packed_bytes_nk(), Phase 2.7 kernel race: native XSIMT/QPA vs Triton prototypes vs grouped INT4.…, run(), _try_tinygemm() (+40 more)

### Community 4 - "modal_app.py"
Cohesion: 0.20
Nodes (18): function, capture_routes(), compute_floor(), download_model(), dsa_proof(), golden_reference(), hw_profile(), phase01_report() (+10 more)

### Community 5 - "V4Config"
Cohesion: 0.11
Nodes (15): Authoritative V4 config loader. The checkpoint's config.json on the volume…, V4Config, ExactAttentionAdapter, Bridge ExactAttention to the eager Attention interface ([b, t, d]).…, Reset decode position AND all session state (ring, compressed history,…, Attention, HyperConnections, Checkpoint-faithful HC: hc_mult residual copies with Sinkhorn mix. Weight names… (+7 more)

### Community 6 - "tiny_causal.py"
Cohesion: 0.07
Nodes (37): fixed_rollout(), kv_rollout(), main(), Track A1.5: E[A] invariant fix + KV rollout parity certification. 1. E[A] must…, Corrected naive rollout (semantic reference)., Clean single-pass KV-cached rollout. Processes the seed at position 0, then…, main(), Track A1: TF/FR h2 path-equivalence harness (roadmap §16-21). On h1-correct… (+29 more)

### Community 7 - "wide_verify.py"
Cohesion: 0.08
Nodes (55): _allocate_mxint_experts(), _allocate_nk_experts(), 256 distinct experts as a QPAExpertBank, generated in chunks of 32., 256 distinct experts as an MXINTExpertBank, generated in chunks of 32., main(), NEXT 3: Hot8 allocation sweep — T_MoE vs bank budget Pareto frontier. Masked…, # NOTE: gu/down compact banks share the same frequency-rank prefix, so, _certify_window() (+47 more)

### Community 8 - "mxint.py"
Cohesion: 0.09
Nodes (54): decode_e4m3(), float8_e4m3fn or uint8-viewed e4m3 -> fp32., _as_u8(), _cdiv(), decode_activation_int(), _dp4a_grouped_src(), _dp4a_src(), _get_fn() (+46 more)

### Community 9 - "test_dspark_tree.py"
Cohesion: 0.10
Nodes (26): accepted_prefix(), analyze(), d0_sampled(), d1_greedy(), d2_coverage(), d3_score_mass_surrogate(), _depth_stats(), DSpark analysis from full-stream draft records — POST-AUDIT semantics. Operates… (+18 more)

### Community 10 - "dsa_layers.py"
Cohesion: 0.15
Nodes (23): _bits_to_f32(), compress_topk_idxs_static(), dequant_scale(), dequant_weight(), get_freqs_table(), hc_post_ref(), hc_pre_ref(), hc_split_sinkhorn_ref() (+15 more)

### Community 11 - "true_p2_sweep.py"
Cohesion: 0.10
Nodes (46): build_block(), main(), Track P3: REAL C192 streaming implementation benchmark. Implements the roadmap…, alloc_workspace(), compute_layer(), decode_layer(), load_core_resident(), All 43 layers FP8-resident + BF16 small tensors + LM head. (+38 more)

### Community 12 - "validate_exact_layer.py"
Cohesion: 0.13
Nodes (16): build_args(), dq_fp8_arch(), dq_fp8_shared(), _fp4_gemm(), _get_expert_weights(), LazyExpertLinear, _linear_nofp8act(), load_real_attention() (+8 more)

### Community 13 - "floor_b_moe.py"
Cohesion: 0.13
Nodes (25): _allocate_experts(), _allocate_mxfp4_experts(), _backend_sequential(), _charge_pool(), _count_launches(), pack_pool(), Tensor, Floor B: one V4 MoE layer, distinct W4 experts resident. (+17 more)

### Community 14 - "test_exact_dsa_decode_parity.py"
Cohesion: 0.11
Nodes (38): _indexer_case(), parametrize, skipif, §81 NEXT B — long-history top-512 selection validation (AGENTS §27, §52). The…, Production bf16 dtype flow on both sides: set overlap per row., Prefill-time indexer selection on both implementations at context S., test_top512_selection_bitwise_fp32(), test_top512_selection_overlap_bf16() (+30 more)

### Community 15 - "d3_bridge.py"
Cohesion: 0.10
Nodes (29): BridgeNet, build_examples(), load_a2(), load_bridge(), main(), R2-5 evaluator: closed-loop Decision Bridge proposer. Production loop (roadmap…, main(), make_bridge_feat() (+21 more)

### Community 16 - "layers.py"
Cohesion: 0.15
Nodes (16): DenseMoELayer, grouped_expert_forward(), HashMoELayer, hc_split_sinkhorn(), MoELayer, V4 layer implementations (Phase 1 correctness path). Attention uses the DSA…, Shared expert + gather/scatter around top-6 routed experts., All-experts-resident variant used for tests and golden parity runs. (+8 more)

### Community 17 - "matrix_hybrid.py"
Cohesion: 0.17
Nodes (12): PinnedExpertPool, All 256 experts of one real layer, packed4 layout, pinned., build_shared(), main(), NEXT 1+2: matrix-selective single-call hybrid — parity + K56 golden timing. gu…, One deterministic source -> (packed4 bank, hot8 gu codes, hot8 down codes)., exact_moe(), exact_moe_mxint() (+4 more)

### Community 18 - "p2_exact_dsa.py"
Cohesion: 0.17
Nodes (26): main(), §86 NEXT 0 — memory-accounting audit (AGENTS §54–56). FINAL FORM. Resolves the…, smi_used_free(), stage(), build_dsa_modules(), certify(), CFG_RATIO(), compute_layer_eager() (+18 more)

### Community 19 - "test_constrained_runtime_regressions.py"
Cohesion: 0.14
Nodes (18): dequant_fp4(), Packed FP4 E2M1 [rows, cols/2] uint8 + E8M0 scale [rows, cols/32] -> bf16…, Exact contract part 2: actual expert tensor layout on disk. Verifies FP4 E2M1…, validate_exact_v4_tensors(), skipif, Fail-closed regression tests for the six constrained-runtime bugs (AGENTS…, Bug 4: the authoritative config.json ships short spellings (dynamic/ue8m0/fp4 +…, Bug 5: the tensor validator must find per-expert indexed names and accept… (+10 more)

### Community 20 - "compute_floor.py"
Cohesion: 0.14
Nodes (30): _bench(), evaluate(), _is_3060(), measure_d2d_bw(), measure_dense_attention(), measure_hyper_connection(), measure_routed_experts(), measure_router() (+22 more)

### Community 21 - "test_dspark_wire.py"
Cohesion: 0.10
Nodes (17): dequant_fp4(), dequant_fp8(), MoEDQ, FP8 [out, in] + scale [out, in/128] (or [out/128, in/128]) -> bf16., Packed FP4 int8 [out, in/2] + scale [out, in/32] -> bf16 [out, in]., Kit-exact MoE for one MTP stage (learned router)., SwiGLUExpert, _attn_tensors() (+9 more)

### Community 22 - "cast_exact_attention"
Cohesion: 0.15
Nodes (16): _clone_weights(), _full_recompute(), main(), Tensor, §74 NEXT 1 — full-model exact-DSA decode soak (AGENTS §51–52). State-lifetime /…, Fresh-module full-prefix recompute of positions [0, upto)., Single-processing lifecycle: every position is consumed by the soak module…, soak_layer() (+8 more)

### Community 23 - "d3_scale_watchdog.sh"
Cohesion: 0.60
Nodes (5): count_captures(), newest_age(), recorder_running(), relaunch(), d3_scale_watchdog.sh script

### Community 24 - "ExactAttention"
Cohesion: 0.08
Nodes (21): analytic(), compression_event_latency(), decode_latency(), fill_caches(), prefill_latency(), §70 NEXT 1 — exact-DSA memory/performance census (AGENTS §39–43). Answers, with…, Isolated compression-event cost: decode step where (sp+1)%ratio==0, at small…, Exact per-buffer persistent bytes, split semantic vs artifact. (+13 more)

### Community 25 - "w4a16_grouped.py"
Cohesion: 0.11
Nodes (30): _capture(), _pool(), CUDA Graphs for the MoE-only K-set. Eager stays if capture fails., run(), emit_tiles(), _emit_tiles_kernel(), fused_moe_w4a16(), last_launch_count() (+22 more)

### Community 26 - "ptx_jit.py"
Cohesion: 0.15
Nodes (24): _kernel(), measure_shape(), _packed_bytes(), Tensor, Phase 2.7 read roofline: cold DRAM, expert reuse, L2 ceiling. Authoritative…, Read nbytes from buf, optionally rotating start by stride each iter., run(), _run_stream() (+16 more)

### Community 27 - "validate_exact_model.py"
Cohesion: 0.17
Nodes (19): arch_get(), build_args(), build_ours_layer(), build_src_layer(), dq_fp8_arch(), dq_fp8_shared(), _fp4_gemm(), _get_expert_weights() (+11 more)

### Community 28 - "warmup_steady"
Cohesion: 0.10
Nodes (35): _bench_component(), _load_layer(), Phase 2.11C/D: exact core execution + T_core(K,C) + VRAM census. Loads FP8/BF16…, Load one layer's core tensors, dequantizing FP8→BF16., run(), _bench(), Native five-token DSpark kill gate — RTX 3060 K1–K6 verifier cycle (roadmap…, run() (+27 more)

### Community 29 - "dspark_wire.py"
Cohesion: 0.14
Nodes (19): log(), main(), DSpark recursive-speculation experiment (roadmap §41-42, decisive). QUESTION…, load_mtp_tensors(), log(), main(), DSpark Stage-2: wire the drafter from the OBSERVED Stage-1 interface. Runs on…, Extract all mtp.* tensors from the HF shards; organize per stage. (+11 more)

### Community 30 - "test_mxint_epilogue.py"
Cohesion: 0.14
Nodes (18): layout_assignments(), Tensor, GPU-side MoE assignment layout. Expert ids never leave the device., expert_ids [A] int64, A = k*top_k. Returns: sorted_ids [A] expert ids sorted…, mxint_swiglu_quant_down(), GU bf16 [A, 2*inter] + router weights [A] -> (a0,a1,a2,sA) down inputs. One…, test_layout_assignments_counts_and_segments(), _make_bank() (+10 more)

### Community 31 - "stage1_train.py"
Cohesion: 0.13
Nodes (16): ArchA, ArchB, ArchC, full_eval(), get_src_hc(), load_head(), load_split(), main() (+8 more)

### Community 32 - "d2_cpu_expert.py"
Cohesion: 0.29
Nodes (12): main(), Source L41/L42 expert matrix on CPU; reference evidence only, never D2 timing.…, run_expert(), signed_activation_check(), main(), D2-E1: CPU VNNI expert vs source GPU expert parity on REAL routed activations…, library(), linear() (+4 more)

### Community 33 - "pack_mxfp4"
Cohesion: 0.08
Nodes (43): dequant_mxfp4(), _f32_to_e2m1(), _floor_e8m0_scale(), load_expert_mxfp4(), pack_mxfp4(), _pack_uint4(), Tensor, Source-preserving MXFP4 pack (E2M1 data + E8M0 scale), not INT4. The Exact-30… (+35 more)

### Community 34 - "d3_lrdl_eval.py"
Cohesion: 0.16
Nodes (15): build_basis(), load_a2(), load_bridge(), load_lrdl(), main(), LRDL evaluator: standalone boundary telemetry (S31) + mandatory splice through…, LRDLNet, main() (+7 more)

### Community 35 - "TailRunner"
Cohesion: 0.33
Nodes (3): Incremental source-tail: prefill once, then step per position. Blocks keep…, FP32 head on one h42 row [4, 4096] -> [V] logits., TailRunner

### Community 36 - "b4o_analysis.py"
Cohesion: 0.15
Nodes (19): coverage_ladder(), ea_from_ladder(), internal_count(), load_ranks(), main(), merge(), node_count(), per_ladder_depth() (+11 more)

### Community 37 - "DSparkAttention"
Cohesion: 0.15
Nodes (7): DSparkAttention, Lin, FP8 linear dequantized to bf16 at load, on `device`. If `scale` is None the…, Kit-exact DSparkAttention in bf16. Ring KV cache (WINDOW) stores the TARGET's…, RMSNorm, TestDSparkAttention, TestLin

### Community 38 - "test_aec.py"
Cohesion: 0.27
Nodes (6): ExpertSurrogate, Tensor, Adaptive Expert Collapse: rank-64 surrogate for omitted expert mass., Rank-64: Linear(H, 64) - GELU - Linear(64, H). Forward(x, omitted_mass) ->…, test_moe_aec_zero_omitted_equals_exact(), test_surrogate_zero_mass_is_noop()

### Community 39 - "mxfp4_decode.py"
Cohesion: 0.17
Nodes (23): decode_e2m1_nk(), e8m0_to_fp32(), float8_e8m0fnu or uint8 -> fp32 power-of-two., packed [..., K/2] uint8 -> fp32 E2M1 values [..., K], no weight scale., decode_e4m3_device(), decode_weight_nk(), e2m1_lut_f32(), exhaustive_e2m1_table() (+15 more)

### Community 40 - "DSparkStage"
Cohesion: 0.22
Nodes (6): DSparkStage, One MTP stage: HC attn + HC MoE (kit Block + DSparkBlock)., Kit DSparkBlock.forward_embed. main_hidden [1, S, 3*dim]., Kit DSparkBlock.forward_head (last stage). Kit hc_head: hc_head_fn [hc_mult,…, _stage_tensors(), TestDSparkStage

### Community 41 - "validate_dsa_real_weights.py"
Cohesion: 0.13
Nodes (6): lc(), dq_fp8(), get32(), load_real(), §24 IMMEDIATE NEXT — real-weight DSA numeric validation (authoritative bytes).…, Dequantize layers.L.attn.<proj>.{weight,scale} -> bf16 [out, in].

### Community 42 - "MosaicVM Continuation Contract (AGENTS.md)"
Cohesion: 0.18
Nodes (11): 1,048,576-Token Exact Context Target, MosaicVM Continuation Contract (AGENTS.md), DeepSeek-V4-Flash-0731 (Target Model), E[A] Expected Acceptance Metric, Exact-30 Throughput Target (>=30 tok/s), Exact Full Verifier (Emission Gate), Source Exactness Contract, Future-h40 + Source L41/L42 Tail Architecture (+3 more)

### Community 43 - "d3_dd1_hist_eval.py"
Cohesion: 0.24
Nodes (12): load_a2(), load_bridge(), load_hist(), main(), DD-1 HIST evaluator: standalone boundary S1 + splice with ps., boot_ci(), main(), Canonical proposer robust-statistics harness (roadmap S17). One single… (+4 more)

### Community 44 - "precompute_freqs_cis"
Cohesion: 0.18
Nodes (12): apply_rotary(), precompute_freqs_cis(), Kit-exact YaRN rope (inference_model.py)., Kit-exact rotary on the last dim (complex pairs)., log(), main(), NanoSpark B1+B2: functional sensitivity + margin analysis (§36-37). B1 — Logit-…, log() (+4 more)

### Community 45 - "StreamingLoader"
Cohesion: 0.11
Nodes (15): inspect_checkpoint_expert_layout(), If MOSAICVM_MODEL_ROOT is set, inspect one expert w1.weight + .scale. Does not…, Path, ExpertLRU, load_core_weights(), Path, Tensor, Memory-budgeted streaming safetensors loader. Resident set = core weights… (+7 more)

### Community 46 - "d2e2b_per_case_ablation.py"
Cohesion: 0.24
Nodes (6): main(), D2-E2b: per-case delta ablation (roadmap S45/S46, closure of the 4 E1 ULP-…, Pass-through GPU reference except at the target call, where the VNNI output is…, SubstitutingLinear, VNNICompute, xhash()

### Community 47 - "pack_mxfp4_nk"
Cohesion: 0.10
Nodes (36): pack_mxfp4_nk(), Source-semantic FP4 layout for Phase 2.7 QPA/XSIMT. Official V4 linears are not…, Authoritative source-linear: act_quant then per-K32 A @ B.T * sA * sW. x_bf16…, w_nk [N, K] -> packed uint8 [N, K/2], scale float8_e8m0fnu [N, K/32]. FLOOR…, reference_fp4_linear_source_semantics(), concat_gate_up(), pair_gate_up(), _perm() (+28 more)

### Community 48 - "C192WorkingSet"
Cohesion: 0.23
Nodes (7): C192WorkingSet, Tensor, Slot-mapped VRAM expert cache over mutable bank tensors., Evict the LRU-oldest NON-PINNED resident expert. Invariant: self.lru keys ==…, Pin the demand set; async-load misses (best-effort when the cache is saturated…, Block until all needed residents' transfers completed., Expert ids -> slot ids. One D2H + dict map + one H2D per call. Defensive: any…

### Community 49 - "nanospark_b3s_rfix.py"
Cohesion: 0.17
Nodes (13): Apply rotary to the last ROPE_DIM of x in place-ish (clone)., Inverse rotary on the last ROPE_DIM (4D [B, S, H, D] input)., x [1, block, dim] draft tokens; main_x [1, S, dim] target proj., _rope_inv_last(), _rope_last(), log(), main(), B3-S-RFIX: sequential rollout with committed-KV ring writes. Fixes the Δ>1 ring… (+5 more)

### Community 50 - "recurrent_drafter.py"
Cohesion: 0.17
Nodes (12): evaluate(), head_logits(), load_head(), load_split(), main(), no_grad, Track P: recurrent target-feature drafter (R0/R1/R2). Design: z_0 =…, H [N, K, 4096] -> logits [N, K, V] via frozen head, chunked over flattened… (+4 more)

### Community 51 - "Router"
Cohesion: 0.10
Nodes (27): calibrate(), Calibrate mean AED k on router logits (no full-model forward)., Capture or replay golden MoE routes. Never fabricate from weights+token-ids., run(), build_capture_harness(), capture_one_prompt(), Generate n_positions consecutive tokens greedily, capturing routes. Returns…, Attach forward hooks to every Router/HashRouter module. Returns (hooks,… (+19 more)

### Community 52 - "fused_moe_mxint"
Cohesion: 0.12
Nodes (25): hot8_partial_from_full(), main(), NEXT 1: hybrid-bank sweep — production-fit K56 MoE timing. VRAM budget: core…, Compact partial hot8 bank holding only hot experts (tensors sliced)., ids [K,6] -> per-assignment rows: (tok_idx, ids[N,1], wts[N,1])., split_window(), capture_layer_tokens(), certify() (+17 more)

### Community 53 - "test_grouped_imma_parity.py"
Cohesion: 0.23
Nodes (13): _banks(), _check(), _exact_target(), fixture, parametrize, Permanent regression coverage for GROUPED_IMMA_TILE_BUG (AGENTS §10). The 2.9R…, Some sibling modules scrub this env var; set it per-test, not at import., Single expert, top-1, weight=1 — the original bug ladder (M>=4 IMMA). (+5 more)

### Community 54 - "act_quant_ref"
Cohesion: 0.26
Nodes (11): act_quant_legacy_phase27(), act_quant_ref(), Tensor, DeepSeek V4-0731 act_quant: per 128-K block, E8M0 power-of-two scale via IEEE…, Phase 2.7 act_quant: s = max(abs(x))/448 (non-pow2). Kept for comparison only., _official_act_quant(), Tensor, DeepSeek V4-0731: E8M0 pow2 scale via IEEE bits, ceil(log2(amax/448)). (+3 more)

### Community 55 - "d3_v1.py"
Cohesion: 0.10
Nodes (36): main(), C2/C3 pre-flight (full contract): no-update free rollout from the ACTUAL…, build_collector_exposure(), build_exposure(), build_transitions(), load_student(), main(), D3-v1 closed-loop evaluator: student + real source tail + FP32 head. Implements… (+28 more)

### Community 56 - "expert_entropy.py"
Cohesion: 0.27
Nodes (12): _duplicate_tile_rate(), _get_range(), _nibble_entropy(), Phase 2.12W-B: exact expert entropy census. Fetches a representative sample of…, E2M1 nibble histogram + Shannon entropy., Fraction of tiles that are exact duplicates of an earlier tile., Entropy of consecutive-byte XOR (captures local redundancy)., run() (+4 more)

### Community 57 - "stage05_heads.py"
Cohesion: 0.23
Nodes (11): load(), main(), G2 calibration: kit-source (Blackwell) vs kit-source (RTX 3060). Compares the…, build_model(), eval_head(), load_split(), Track S Stage-0.5 (roadmap §47-53): trained deep-exit heads. Phase 1 (record):…, X: [N, 4, 4096] fp32 (GPU). Returns top-k accuracy dict. (+3 more)

### Community 58 - "test_dsa_sparse_fused.py"
Cohesion: 0.12
Nodes (18): constexpr, jit, Tensor, Fused exact-DSA sparse attention — round-2 Triton kernel (AGENTS §42–46).…, q [s,h,d] bf16, kv [n,d] bf16, topk_idxs [s,t] int32 (-1 masked), sink [h] fp32…, _sparse_attn_fused(), parametrize, skipif (+10 more)

### Community 59 - "HashRouter"
Cohesion: 0.12
Nodes (21): Exact contract part 1: config values match the V4-0731 source semantics.…, validate_exact_v4_config(), HashRouter, Hash-routed layers: no bias, real gate-projection scores, literal normalize.…, ExactContractError, RuntimeError, Exact-target contract for the production MoE path. When…, Raised when the loaded model/config violates the V4-0731 exact contract. (+13 more)

### Community 61 - "StreamLinear"
Cohesion: 0.18
Nodes (8): capture_real_calls(), main(), D2-P: adapter-inclusive CPU VNNI expert-path timing (roadmap S38). v2 —…, Load the kit model, run val prompts, capture routed expert calls (prefix,…, Routed-expert linear: packed FP4 + E8M0 scale read from the snapshot shards on…, StreamLinear, FastStreamLinear, Recording-speed expert path: cache the unpacked, scale-pre- applied bf16 weight…

### Community 62 - "transport.py"
Cohesion: 0.43
Nodes (6): _bench_component(), _bench_h2d(), _bench_nvme(), Phase 2.12W-A: real transport measurement on the target machine. Measures…, Sequential read benchmark., run()

### Community 63 - "d2e2_hybrid_tail.py"
Cohesion: 0.32
Nodes (4): HybridLinear, main(), D2-E2: hybrid-tail parity (roadmap S43-S46). Production topology under test:…, Routed-expert linear executed on CPU via the VNNI adapter; everything else in…

### Community 64 - "dequant_fp8"
Cohesion: 0.18
Nodes (16): _try_load_checkpoint_experts(), FP8 materialized baseline: dequant_fp8 + bf16 GEMM on [4096,512], M=1-8. Not…, run(), load_expert_mxfp4_prefix(), Load ``prefix + 'w1'/'w3'/'w2'``-style weights and pack to MXFP4. Returns…, dequant_fp8(), fit_linear_weight(), maybe_unpack_fp4() (+8 more)

### Community 65 - "source_corpus_gen.py"
Cohesion: 0.24
Nodes (9): main(), Track D1-R: exact L41/L42 tail replay from true h40 states. THE decisive…, main(), Track S Stage-0 (roadmap §65-67): early-exit layer scan. Zero-training oracle:…, build_kernel_module(), log(), main(), memdbg() (+1 more)

### Community 66 - "d3_scale_orchestrate.sh"
Cohesion: 0.33
Nodes (5): MALLOC_ARENA_MAX, OMP_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF, SCG_CACHE_CAP, d3_scale_orchestrate.sh script

### Community 67 - "_golden_windows"
Cohesion: 0.18
Nodes (17): main(), Track D: split fused_moe dispatch into GPU work vs launch/sync. Per…, Named compute floors and the 30 tok/s equation used by every Phase 2.5 JSON., main(), Track D: GU parameter race at the Exact-30 widths (N16-32). Sweeps…, main(), Track D (roadmap §53-54): routed-expert intercept decomposition. Runs the SAME…, _golden_windows() (+9 more)

### Community 68 - "load_captures"
Cohesion: 0.10
Nodes (61): load_embed(), load_head(), Frozen production norm weight + FP32 head (CPU)., build_dd_examples(), build_hist_examples(), DDHistNet, main(), DD-1 HIST: exact-root history race (roadmap S9-S24). Parameter-matched three-… (+53 more)

### Community 69 - "vnni_kernel.c"
Cohesion: 0.27
Nodes (5): __m256i, hsum8(), hsum8i(), vnni_gemm_packed(), vnni_gemm_scaled2()

### Community 70 - "d3_student.py"
Cohesion: 0.47
Nodes (7): fit(), main(), predict(), CPU-only future-h40 PCA/ridge pilot; exact source tail remains the evaluator.…, Train-only PCA and per-displacement ridge, returning full-state decoder. The…, residual_metrics(), test_train_only_future_alignment_and_recursive_no_oracle()

### Community 71 - "hw_profile.py"
Cohesion: 0.31
Nodes (10): cpu_gemm(), host_device_bandwidth(), power_clocks(), ram_profile(), Hardware profiler: emits one JSON profile for the current GPU + host. Measures…, Sequential + 4KB random reads from a file on the mounted volume., run(), _time_gpu() (+2 more)

### Community 72 - "vnni_bench.py"
Cohesion: 0.43
Nodes (6): e4m3_decode(), main(), make_inputs(), Track V: exact VNNI packed-FP4 expert kernel — parity + benchmark. Parity:…, fp32 reference with the SAME scale structure: y[m,n] = sum_wb sW[n,wb] *…, reference()

### Community 73 - "fetch_core.py"
Cohesion: 0.39
Nodes (8): _get_range(), Phase 2.11A: acquire exact core tensors locally via HTTP range requests. The…, bytes [start, end] inclusive., # NOTE: safetensors requires contiguous tensor payloads (no gaps);, _revision(), run(), _shard_header(), _url()

### Community 74 - "residency_oracle.py"
Cohesion: 0.33
Nodes (8): _belady(), _block_sequence(), _bytes_for(), _load_corpus(), Phase 2.12A: golden-route residency oracle (byte-aware Belady). Simulates…, Byte-aware Belady VRAM cache + strict-hierarchy RAM warm tier. granularity…, Group positions into verification blocks of width K; each block step is (layer…, run()

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

### Community 81 - "dispatch_profile.py"
Cohesion: 0.15
Nodes (15): Event, amdahl(), _event_ms(), _LaunchCounter, _make_ids(), _pack_pool(), _profiled_sequential_moe(), _profiler_stats() (+7 more)

### Community 82 - "DeltaNet"
Cohesion: 0.06
Nodes (49): main(), prefix_metrics(), prompt_mean_ci(), Track D1-B: L41/L42 tail robustness to approximate h40 trajectories. THE…, K uncertain draft decisions, excluding exact root; prefix+1 emission. Input…, Percentile bootstrap over independent prompts, never sliding windows., Reset all nested source attention AND indexer compressor buffers., reset_decode_state() (+41 more)

### Community 83 - "belady_bound.py"
Cohesion: 0.38
Nodes (6): belady(), load_streams(), main(), Track R: Belady-optimal residency upper bound on GENUINE routes. Access stream:…, {cat: [ (token, layer, expert) ... ] in execution order., Returns (hits, misses, per-layer hits/misses).

### Community 84 - "branchpack_oracle.py"
Cohesion: 0.43
Nodes (6): classify(), load_coverage(), oracle_dp(), Phase 2.13P-O: BranchPack accepted-depth oracle (AGENTS.md §28-30). Uses…, DP: best {b_d} allocation maximizing E[A] with sum(b_d) <= n_budget. c_by_b:…, run()

### Community 85 - "core_inventory.py"
Cohesion: 0.48
Nodes (6): bf16_bytes(), fp8_e4m3_block128_bytes(), inventory(), mxfp4_bytes(), Print per-tensor bf16 vs W4 bytes for the resident core. No snapshot needed., w4_bytes()

### Community 86 - "dsa_attention.py"
Cohesion: 0.32
Nodes (7): _dsa_attend_kernel(), _dsa_select_kernel(), constexpr, jit, Decode-time top-k sparse attention (DSA) Triton kernel. Doc §7: never…, Stage 1: per-head running top-k. Grid: (heads // HEAD_GROUP, batch). Never…, Stage 2: gather top-k KV per head and compute attention output.

### Community 87 - "dsa_bench.py"
Cohesion: 0.22
Nodes (7): eager_forward(), Tensor, §60-1: exact DSA attention vs eager stand-in — matched full pipelines. Both…, Eager attention block on the SAME module weights: identical q/kv projections,…, _rms(), load_exact_attention(), Load real weights from the core archive into ExactAttention. `get(name) ->…

### Community 88 - "d3_r2_teacher_audit.py"
Cohesion: 0.67
Nodes (3): main(), R2-1 teacher-dataset audit: canonical, hash-bound, fail-closed. Canonical…, sha_full()

### Community 89 - "MemoryBudget"
Cohesion: 0.22
Nodes (3): MemoryBudget, Bug 6 (mechanism): load_streaming_model treats ram_overflow entries as…, test_ram_overflow_is_not_silent()

### Community 90 - "mlp"
Cohesion: 0.40
Nodes (5): mlp(), Tensor, log(), main(), Real >=256-position route capture on the Blackwell sandbox (BATCHED). All 7…

### Community 91 - "hc_split_sinkhorn"
Cohesion: 0.47
Nodes (3): hc_split_sinkhorn(), Kit-exact hc_split_sinkhorn (inference_kernel.py lines 391-425)., TestSinkhorn

### Community 92 - "fp8_core.py"
Cohesion: 0.14
Nodes (24): compute_layer_fused(), _decode_parity_check(), _dequant_block(), dequant_kernel(), _e8m0_to_f32(), fp8_linear(), _fused_fp8_linear(), constexpr (+16 more)

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

### Community 101 - "qpa_mma.py"
Cohesion: 0.09
Nodes (33): _load(), _p1_parity_check(), Phase 2.9R go/no-go. One precedence-ordered verdict line., Tiny E=8 fixture: fused_moe_mxint (Hot8) vs fused_moe_qpa numeric parity., run(), e4m3_lut_f32(), _as_u8(), _cdiv() (+25 more)

### Community 102 - "dsa_decode_attention"
Cohesion: 0.32
Nodes (6): dsa_decode_attention(), Public API. q: [H, index_dim] bf16 queries for one decoding token (batch=1).…, Plain PyTorch reference implementing the same top-k sparse attention., reference_topk_attention(), skipif, test_dsa_parity()

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

### Community 109 - "stage1_record.py"
Cohesion: 0.57
Nodes (5): capture_span(), main(), prompt_splits(), Full four-stream state recorder with explicit teacher provenance. Default:…, test_prompt_split_and_future_state_boundary()

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

### Community 133 - "d3_scale700_engine.sh"
Cohesion: 0.60
Nodes (5): count(), launch_phase(), newest_age(), recorder_running(), d3_scale700_engine.sh script

### Community 134 - "d3_scale700_orchestrate.sh"
Cohesion: 0.33
Nodes (5): MALLOC_ARENA_MAX, OMP_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF, SCG_CACHE_CAP, d3_scale700_orchestrate.sh script

### Community 135 - "pack_w4"
Cohesion: 0.10
Nodes (34): Linear, _packed_bytes_mxfp4(), _packed_bytes_w4(), Skinny MXFP4 kernel race: skinny_mxfp4 vs grouped_blockm16 (INT4) baseline.…, run(), _try_tinygemm(), dequant_w4(), pack_gate_up() (+26 more)

### Community 136 - "d3_scale_teacher.py"
Cohesion: 0.53
Nodes (5): main(), Path, SCALE210 teacher generator: gold-only SourceTail replay for the 126 new…, sha(), topk_pack()

### Community 137 - "prompts.py"
Cohesion: 0.33
Nodes (4): Golden prompts shared by quality sweeps. Do not fork this list., Path, Static top-k quality sweep vs k=6 golden. Remote H100 run lives in…, write_blocked_stub()

### Community 139 - "ExpertMLP"
Cohesion: 0.40
Nodes (3): ExpertMLP, gate/up/down SwiGLU at moe_intermediate, FP32 with optional swiglu_limit clamp., SharedExpertMLP

### Community 140 - ".generate"
Cohesion: 0.40
Nodes (3): no_grad, Greedy loop: prefill then last-token decode. No speculation., Prefill on input_ids, then accept every future block in one forward.…

### Community 141 - "capture_routes_local"
Cohesion: 0.50
Nodes (4): local_entrypoint, capture_routes_local(), main(), Dispatch capture_routes on Modal and persist the payload locally.

## Knowledge Gaps
- **21 isolated node(s):** `d3_scale700_orchestrate.sh script`, `SCG_CACHE_CAP`, `PYTORCH_CUDA_ALLOC_CONF`, `MALLOC_ARENA_MAX`, `OMP_NUM_THREADS` (+16 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 821 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `V4Config` connect `V4Config` to `apply_rotary_emb`, `pack_w4`, `dsa_layers.py`, `validate_exact_layer.py`, `test_exact_dsa_decode_parity.py`, `layers.py`, `p2_exact_dsa.py`, `test_constrained_runtime_regressions.py`, `cast_exact_attention`, `ExactAttention`, `validate_exact_model.py`, `warmup_steady`, `test_aec.py`, `validate_dsa_real_weights.py`, `StreamingLoader`, `Router`, `test_dsa_sparse_fused.py`, `HashRouter`, `validate_dsa_source.py`, `core_inventory.py`, `dsa_bench.py`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `fused_moe_mxint()` connect `fused_moe_mxint` to `_golden_windows`, `qpa_mma.py`, `wide_verify.py`, `mxint.py`, `true_p2_sweep.py`, `floor_b_moe.py`, `pack_mxfp4_nk`, `matrix_hybrid.py`, `p2_exact_dsa.py`, `test_grouped_imma_parity.py`, `act_quant_ref`, `w4a16_grouped.py`, `validate_exact_model.py`, `warmup_steady`, `test_mxint_epilogue.py`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `stats()` connect `d3_dd1_hist_eval.py` to `d3_lrdl_eval.py`, `true_p2_sweep.py`, `load_captures`, `d3_bridge.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `V4Config` (e.g. with `Compressor` and `ExactAttention`) actually correct?**
  _`V4Config` has 19 INFERRED edges - model-reasoned connections that need verification._
- **What connects `d3_scale700_orchestrate.sh script`, `SCG_CACHE_CAP`, `PYTORCH_CUDA_ALLOC_CONF` to the rest of the system?**
  _21 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `apply_rotary_emb` be split into smaller, more focused modules?**
  _Cohesion score 0.13538461538461538 - nodes in this community are weakly interconnected._
- **Should `_synthetic_capture` be split into smaller, more focused modules?**
  _Cohesion score 0.06715063520871144 - nodes in this community are weakly interconnected._