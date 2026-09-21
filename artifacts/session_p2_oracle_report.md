# Session Report — P2 Oracle Tree: Current Proposer Family Lacks
# Probability Mass for Trees; L42 Shows Information Exists

**Date:** 2026-09-06
**Roadmap:** Linear Speculation Closed — Branching + Hybrid Execution
Must Now Be Co-Designed

## The oracle tree verdict: **current T1 logits cannot support a
## useful tree — E[A] caps at ~2-3 regardless of budget**

### Root Top-k (bypass on L40) — GOOD

| k | coverage |
|---|---|
| 1 | 50.9% |
| 2 | 66.7% |
| 4 | **75.2%** |
| 8 | **82.0%** |
| 16 | 87.4% |
| 32 | 94.1% |

### Conditional Top-k at h2+ (model, given TRUE prefix) — THE
### BOTTLENECK

| horizon | Top-1 | Top-4 | Top-8 | Top-16 |
|---|---|---|---|---|
| h2 | 16.7% | 23.9% | 30.6% | **35.1%** |
| h3 | 14.4% | 24.8% | 32.0% | 36.9% |
| h4 | 14.4% | 24.3% | 31.5% | 37.4% |
| h8 | 15.8% | 29.3% | 34.2% | 39.2% |

**The model's conditional distribution is extremely flat**: Top-16 at
h2 covers only 35% of true tokens. Even a width-16 tree at every
position captures the true path at h2 for only 31% of examples.

### Beam survival (bypass root + model h2+)

| width | S1 | S2 | S3 | S4 | S8 | S16 |
|---|---|---|---|---|---|---|
| w=4 | 75.2% | 18.9% | 5.9% | 2.3% | 0% | 0% |
| w=8 | 82.0% | 25.7% | 10.4% | 4.1% | 1.8% | 0.9% |
| w=16 | 87.4% | 30.6% | 14.0% | 9.0% | 3.6% | 1.4% |
| w=32 | 94.1% | 39.6% | 18.9% | 11.7% | 4.5% | 2.7% |

### Oracle tree E[A] under node budgets

| budget | best E[A] | config |
|---|---|---|
| 32 | 1.82 | w=2, D=4 |
| 112 | 1.94 | w=4, D=3 |
| 512 | 2.02 | w=6, D=3 |
| 8192 | 2.18 | w=8, D=4 |
| ceiling | 3.28 | maxrank<32, D=32 |

**Even unlimited budget with width-32 everywhere: E[A] ceiling ≈ 3.3.**
Far from the ~24-28 needed.

### Root quality by exit layer

| exit | Top-1 | Top-2 | Top-4 | Top-8 |
|---|---|---|---|---|
| L36 | 34.2% | 41.4% | 45.5% | 53.2% |
| L38 | 42.8% | 49.1% | 54.5% | 62.6% |
| **L40** | **49.1%** | 65.8% | 75.2% | 82.0% |
| **L42** | **76.1%** | **90.5%** | **97.3%** | **98.2%** |

## The critical insight

**L42 (the final layer) achieves 76.1% Top-1 and 97.3% Top-4.** The
information about the next token IS present in the deeper
representation — the last 2 layers add 27 percentage points of
next-token accuracy over L40. The problem is access:

1. L42 costs the full model forward (not a cheap drafter)
2. The L40-based tiny causal model can't extract even L40's full
   potential at h2 (conditional Top-16 = 35%)
3. The model was trained on only ~1k windows with ~37M parameters —
   likely data/capacity starved

**The proposer problem is not information-theoretic — it is a
model-capacity/data problem.** The target's next-token information
exists at L42 (76% Top-1, 97% Top-4). A drafter that can access a
sharper conditional distribution (through better training, more data,
or distillation from deeper states) could make trees viable.

## Per the roadmap decision table (§79-81)

**Oracle tree weak everywhere** → the current tiny-causal model
family lacks sufficient probability mass. Promote:

1. **Hidden-state distillation** (§73-74): train the drafter to
   predict h42,t+i (or h40,t+i) — teach it what the deeper layers
   know
2. **Large data scaling** (§71): 1000+ prompts — the model is
   almost certainly data-starved (37M params on 1k windows)
3. **Stronger root/exit** (§34-35): L42 shows the ceiling; if a
   cheaper approximation to L42's decision exists, root coverage
   could jump from 49% to ~76%

## Tree viability threshold

For E[A] = 24 at depth 32, root coverage ≥ 72% is needed (§41).
L40 Top-4 = 75.2% already exceeds this. The root is NOT the problem
— the h2+ conditional model is. If h2 conditional Top-4 improved from
24% to even 50%, a w=4 tree would give S2 ≈ 37%, S3 ≈ 18%, pushing
E[A] toward 4-6. With Top-4 at 70%: E[A] toward 10-15.

## Artifacts

```text
artifacts/p2_oracle_tree.json       full oracle analysis
mosaicvm/bench/p2_oracle_tree.py    analysis script
```
