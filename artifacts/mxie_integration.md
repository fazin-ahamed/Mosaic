# MXIE integration + token-0 gate — 6/7 exact (§51-§60)

Date: 2026-09-03 · Harness: scripts/validate_exact_model.py
Archive: a9089f25fa85bac9…

## Integration (§53 one-primitive-at-a-time)

1. **Routed experts → MXINT packed4** (fused_moe_mxint + prepack_
   expert_bank_mxint_packed4 from the selected experts' snapshot
   tensors): FP8-act × FP4-weight exact integer arithmetic, source-
   exact swiglu/route-weight/w2-requant epilogue, ascending-order
   reduce. Verified: coding golden 3/3 exact, ~2.5× faster (19-21s/token
   vs 46s dequant).
2. **Shared expert → source FP8 class**: dynamic E4M3 act quant per-128
   (act_quant_ref = the source act_quant, pow2 E8M0 scale) on x and on
   the weighted SwiGLU input to w2, block-dequantized FP8 weights. This
   fixed long_generation (21429 → 270 == corpus).

## §60 token-0 gate: 6/7

| domain | tok0 | corpus | |
|---|---:|---:|---|
| coding | 62 | 62 | ✓ |
| agent_tool_use | 6948 | 6948 | ✓ |
| general_conversation | 418 | 418 | ✓ |
| long_context_retrieval | 304 | 304 | ✓ |
| structured_json | 270 | 270 | ✓ |
| long_generation | 270 | 270 | ✓ (fixed by shared-expert class) |
| math_reasoning | 15832 | 12803 | ✗ — see below |

## math_reasoning: reference sensitivity, not a runtime defect

Three-way comparison (scripts/validate_exact_model.py srcchain):

```text
my runtime (3060, MXINT+source-shared)   15832
kit source chain (3060, local)           15832   <- identical to mine
corpus (Blackwell transformers)          12803   <- differs from BOTH
```

Kit-source top-5 logits: 15832 (21.99) · 45773 (21.12) · **12803
(20.67)** — the corpus token sits at rank 3. The Blackwell transformers
runtime's own bf16 kernel-order numerics chose differently from the kit
source on this prompt. My runtime faithfully implements the source
semantics; the corpus reference is implementation-sensitive at this
position.

## Consequence for the 2/2 equivalence gate

Exact-token equivalence against the Blackwell corpus cannot be 7/7 when
the reference itself diverges from source semantics on near-ties. The
equivalence contract needs either:
- a regenerated corpus from a source-class reference, or
- a top-k/logit-tolerance equivalence class for near-tie positions, or
- acceptance that 6/7 + source-faithful math is the achievable exact-
  token bound for this corpus.

## Runtime numeric-class status

```text
routed experts       MXINT exact integer (source class)   DONE
shared expert        FP8 act-quant class                  DONE
attention            ExactAttention (validated)            DONE
router/hash          exact (fp32 bias)                    DONE
HC/head/lm_head      source math                          DONE
```

The constrained runtime is now full source-class on every expert
primitive. Remaining: full-corpus overlap run + the equivalence-class
decision above.
