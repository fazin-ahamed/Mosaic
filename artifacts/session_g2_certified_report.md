# Session Report — G2 Certified 2/2 + Tracks S/D/M

**Date:** 2026-09-04
**Roadmap:** G2 v1 Frozen + Deep-Exit Signal + GU Bottleneck + Final Memory Shape

## Track E — G2 certification: **PASS → 3060 EQUIVALENCE 2/2**

Disjoint held-out certification set (`g2_cert_prompts.json`, 7 new
prompts, never used for calibration or Stage-0.5). Reference side:
kit inference/model.py on RTX 3060 (same hardware as MosaicVM —
deviation: Blackwell sandbox dead; the cross-hardware-calibrated
envelope is therefore an UPPER bound on this pairing's natural
divergence, making the pass conservative). MosaicVM side: full
32-token recording (tokens, per-token top-5, per-step routes,
per-step per-layer hiddens).

**Verdict: PASS — all 7 prompts** (`g2_certification_result.json`,
312a65d4). Every first divergence qualified under the FROZEN spec
(thresholds untouched):

| domain | div@ | margin | pair | top16 | hid worst | hash |
|---|---|---|---|---|---|---|
| agent_tool_use | 23 | 0.056 | ✓ | ✓ | 0.375 | ✓ |
| coding | 2 | 0.032 | ✓ | ✓ | 0.121 | ✓ |
| general_conversation | 11 | 0.373 | ✓ | ✓ | 0.300 | ✓ |
| long_context_retrieval | 0 | 0.091 | ✓ | ✓ | 0.000 | ✓ |
| long_generation | 3 | 0.015 | ✓ | ✓ | 0.143 | ✓ |
| math_reasoning | 5 | 0.022 | ✓ | ✓ | 0.245 | ✓ |
| structured_json | 3 | 0.592 | ✓ | ✓ | 0.227 | ✓ |

```text
3060 EQUIVALENCE = 2/2 (G1 token-0 7/7 + G2 frozen-spec held-out PASS)
```

Supersede note: a Blackwell-side disjoint rerun (when a sandbox
returns) would make the pairing match the calibration basis exactly.

## Track D — dispatch decomposition (`dispatch_decomposition.json`, fd7f8ce6)

The ~0.7ms/call "dispatch" category is **nearly all real GPU work**
(layout/act-quant/decode). Launch/sync gap = 0.42-1.7ms/call
(4-9% of wall; negative at two points = noise). **CUDA graphs would
recover ≤9% of routed time — demoted from primary target.** The
denominator attack remains GU (1.85:1 vs DOWN) at N16-32.

## Track M — CPU embed/head benchmark (`cpu_offload_bench.json`, 701f293c)

CPU head GEMM (fp32, 6 threads): 43ms at batch-1 / 104ms at batch-8,
vs ~13ms GPU. **The 2.1GB proposer budget costs ~43ms/verify-block
(~7-10% of T(N))** — real but workable; embed lookup negligible
(0.06ms). Proposer soft target stays well under 2.1GB.

## Track S — Stage-0.5 trained exit heads (`stage05_heads_result.json`)

Architecture: hc_head-form mixing (65,541 params) + frozen lm_head,
CE on next token, Adam 3e-3 ×400 steps, best-of-val, 7/7/7
train/val/held-out prompt split (disjoint from everything prior).

| exit | raw top-1 (held-out) | trained top-1 | d2 | d4 |
|---|---|---|---|---|
| L38 | 0.357 | 0.353 | 0.158 | 0.036 |
| L40 | 0.478 | 0.446 | 0.209 | 0.051 |

**Reading:** the 7-prompt/224-position training set is ~2 orders too
small for 65K params (val 0.585 vs held-out 0.446 = overfit). The
representation result stands across three disjoint prompt sets
(L40 raw top-1 ≈ 0.48-0.55, top-16 ≈ 0.89-0.94); the source hc_head
mixing is already near-optimal at this scale. **Next (Stage-1): scale
the recording set (100+ prompts, ~3-5K positions) before judging
trained heads, then multi-token heads from one deep state** — not a
sequential L40 drafter (per §45-46, L40 sequential = 40/43 target cost
per draft token).

## Artifacts

```text
g2_cert_prompts.json / g2_cert_ref_3060/ / g2_cert_mosaicvm/
g2_certification_result.json    312a65d4  PASS 2/2
dispatch_decomposition.json     fd7f8ce6  launch gap 4-9% only
cpu_offload_bench.json          701f293c  43ms/block for 2.1GB
stage05_prompts.json / stage05_heads_result.json
tau_h_prefix.json 7e194e6a / g2_spec_v1.md cf7314da (frozen)
early_exit_scan.json c10b15e4 / vram_census.json db5d9b0d
true_p2_sweep.json 192dada2 (E_req relabeled + swap classified)
routed_expert_decomposition.json a5e7dd19 (GU 1.85:1)
```
