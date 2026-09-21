# MosaicVM Canonical Master Roadmap

## Post B3-S Sequential Integrity Audit

**Target:** DeepSeek-V4-Flash-0731
**Production hardware:** RTX 3060 12 GB + ~24 GB RAM + NVMe
**Maximum exact context:** 1,048,576 tokens
**Primary target:** ≥30 sustained final accepted/output tok/s
**Ultimate target:** Exact-30 at full 1M context

**Status:** CANONICAL — supersedes all previous MosaicVM roadmaps.

---

# 1. Executive state

```text
TESTS                                  269 / 269 PASS

TARGET ENGINE
QPA                                    FROZEN
MXIE                                   VALIDATED
Grouped IMMA                           EXACT
Hot8 K56 ceiling                       0.6491 s
Production expert backend              Mixed-A
Mixed-A bank                           4.53 GB

CORE
Core GEMM race                         COMPLETE
N32/N56/N80/112                        148 / 232 / 260 / 325 ms
Exact DSA                              OPEN
0xFF semantics                         OPEN
TRUE P2                                NOT CERTIFIED

P2_COREBLOCK
N56                                    26.3 oracle/s measured
N80                                    29.1 oracle/s measured
N112                                   31.0 oracle/s measured

POST-GEMM PROJECTION
N80                                    ~29.7/s
N112                                   ~32.3/s

ROUTING
Blackwell route corpus                 ~260 × 7 × 43
N112 bank-touch p50                    46.1%
Classification                         AUTHORITATIVE_BLACKWELL_ROUTING
3060 equivalence                       BLOCKED 0/2

DSPARK
Sampled chain E[A]                     1.400
Greedy chain E[A]                      1.933
Fresh Top-1                            71.4%
Fresh Top-16                           96.7%
Native 5-token Case A                  CLOSED
3060 optimistic verify                 258.4 ms > 200 ms

NANOSPARK
main_x[4096]                           PROVEN SUFFICIENT
Low-rank state learning                POSITIVE
Generic greedy-recursion gain          NONE

B3-R                                   VALID
n                                      112 / Δ
Hold Δ1 Top-1/4/8/16                   26.8/48.2/57.1/66.1%
R64  Δ1 Top-1/4/8/16                   24.1/45.5/61.6/70.5%

B3-S CURRENT RUN
Δ1 E[A]                                2.111
Δ1 status                              PROVISIONALLY VALID / SMALL SAMPLE
Δ2 E[A]=2.111                          INVALIDATED
Δ3 E[A]=2.167                          INVALIDATED
Δ5 E[A]=2.500                          INVALIDATED
"Δ5 best"                              RETRACTED

NEXT                                   B3-S-RFIX
```

---

# 2. Exact target contract

Production must preserve:

```text
source FP8 core
source FP4 expert codes
source expert scales
dynamic E4M3 activation quantization
E8M0 scales
real router
hash routing
sqrtsoftplus
selection-only correction bias
Top-6
all six routed experts
route normalization ×1.5
source SwiGLU
route-weight placement
shared expert
ordered accumulation
source cast boundaries
exact target verification
```

Forbidden:

```text
Top4 target approximation
expert dropping
target pruning
lossy target compression
source requantization
approximate target experts
unverified speculative emission
```

Approximate drafting remains allowed because exact V4 verification controls final output.

---

# 3. Exactness classes

Keep separate:

```text
SOURCE_EXACT
BITWISE_MOSAIC_REFERENCE
NUMERIC_PARITY
GOLDEN_GREEDY_PASS
NO_QUALITY_LOSS_VERIFIED
```

---

# 4. Evidence classes

Keep separate:

```text
BLACKWELL_SANDBOX_ONLY
DRAFTER_NUMERICS_DEVIATION_RECORDED
3060_TIMING_KILL_GATE
3060_EQUIVALENCE
PRODUCTION_3060

AUTHORITATIVE_BLACKWELL_ROUTING
PRODUCTION_3060_AUTHORITATIVE
```

---

# 5. Final throughput contract

$$
R_{\rm output}
=
\frac{E[A_{\rm real}]}
{
T_{\rm proposal}
+
T_{\rm target}
+
T_{\rm movement}
+
T_{\rm runtime}
-
T_{\rm overlap}
}.
$$

Only:

$$
\boxed{P4\ge30}
$$

final sustained accepted/output tok/s is primary success.

---

# 6. Context contract

Preserve:

$$
\boxed{1,048,576}
$$

maximum exact context.

Final reporting separates:

```text
Maximum Exact Context
Maximum Exact-30 Context
1M-context P4 throughput
```

---

# 7. QPA

```text
COMPLETE
FROZEN
REFERENCE
FALLBACK
```

---

# 8. MXIE

MXIE remains the exact software-defined low-bit execution engine.

$$
q_W=2W
$$

$$
q_A=512A
$$

$$
AW=\frac{q_Aq_W}{1024}.
$$

No target requantization occurs.

---

# 9. Expert compute ceiling

Real-golden all-Hot8 K56:

$$
\boxed{0.6491s}.
$$

Fastest exact expert endpoint, but globally memory-incompatible.

---

# 10. Production expert backend

Mixed-A:

```text
GU      packed4
DOWN    Hot8
```

Footprint:

$$
\boxed{4.53GB}.
$$

---

# 11. Mixed-A measured frontier

```text
N56      0.8379 s
N63      0.9023 s
N80      1.0046 s
N112     1.1959 s
```

---

# 12. Partial Hot8 masks

Exact infrastructure remains available.

Closed as primary short-context accelerator.

Retain for:

```text
context-dependent residency
dynamic Hot8 caching
long-context pressure
route-aware allocation
```

---

# 13. Transport

RAM→GPU STEADY:

$$
\boxed{12.90GB/s}.
$$

NVMe sequential:

$$
\boxed{\sim11GB/s}.
$$

Sparse/random behavior remains open.

---

# 14. Core GEMM

### M ≤ 80

```text
fused FP8 decode/GEMM
BM32
BN64
BK128
warps4
stages3
```

### M > 80

```text
dequant
→ shared ~64MB BF16 scratch
→ cuBLAS
```

---

# 15. Core curve

$$
\boxed{148/232/260/325\text{ ms}}
$$

at:

$$
N=32/56/80/112.
$$

---

# 16. v5 split-K

Closed.

Valid repaired results:

```text
M80      3.66 ms/layer
M112     4.75 ms/layer
```

Historical 2.91 ms result remains invalid.

---

# 17. Core numerical class

Current winning paths:

```text
NUMERIC_PARITY PASS
```

not globally:

```text
BITWISE_MOSAIC_REFERENCE
```

---

# 18. Exact DSA

Still target-side NEXT.

Implement source-equivalent V4 DSA and measure:

```text
correctness
N32
N56
N80
N112
workspace
VRAM
STEADY thermals
```

Generic SDPA remains rejected.

---

# 19. 0xFF semantics

Still:

```text
OPEN_SEMANTICS_CASE
```

Must close before final source/no-quality-loss certification.

---

# 20. Current P2 truth

Measured:

```text
N56     26.3 oracle/s
N80     29.1 oracle/s
N112    31.0 oracle/s
```

Post-GEMM projections:

```text
N80     ~29.7/s
N112    ~32.3/s
```

Do not relabel projections as measured.

---

# 21. TRUE P2

Requires:

```text
production source-FP8 core
exact DSA
resolved source semantics
production expert backend
43 layers
LM head
all mandatory target components
perfect expert residency
production-compatible representation
STEADY thermals
```

---

# 22. Routing

Latest real Blackwell routing remains:

```text
AUTHORITATIVE_BLACKWELL_ROUTING
```

with N112 median bank touch:

$$
\boxed{46.1\%}.
$$

---

# 23. 3060 routing status

Still:

$$
\boxed{0/2}.
$$

Need:

```text
32-token overlap corpus
tokens
Top-6 IDs
hash routes
route weights
```

before production-3060 authority.

---

# 24. D0 sampled DSpark

$$
E[A]=1.400.
$$

---

# 25. D1 greedy DSpark

$$
E[A]=1.933.
$$

Approximately +38% over sampled recursion.

---

# 26. D2 fresh-state coverage

$$
Top1=71.4\%
$$

$$
Top16=96.7\%.
$$

This remains fresh-state evidence only.

---

# 27. D3

Remain classified:

```text
D3_SCORE_MASS_PATH_SURROGATE
```

not accepted-depth evidence.

---

# 28. Case A

Native approximately five-token DSpark linear chain:

```text
258.4 ms optimistic 3060 verification
>
200 ms perfect gate
```

Therefore:

$$
\boxed{\text{CASE A CLOSED}}.
$$

---

# 29. Case-A reopening

Only reopen with substantive exact evidence changing the relevant system timing enough to pass the gate.

---

# 30. P-B seed

Proven sufficient:

```text
post-main_proj + main_norm
main_x[4096]
```

Top-3 overlap:

$$
1.0.
$$

---

# 31. Smaller interfaces

Still untested:

```text
pre-norm projected state
main_kv[512]
other compressed interfaces
```

4096 is only the lowest dimension proven so far.

---

# 32. NanoSpark dataset

$$
7\times210
$$

positions with 205 block-spaced pairs.

---

# 33. Predictor latency

Blackwell:

$$
<0.06ms.
$$

Not yet a 3060 timing result.

---

# 34. Damped velocity

Closed as promoted direction.

---

# 35. Low-rank state learning

Rank-64 improves:

$$
0.4115\rightarrow0.5655
$$

cosine in the original ladder and cuts relative state error by about:

$$
18.6\%.
$$

State transition is partially learnable.

---

# 36. Earlier greedy-recursion result

Despite state improvement:

```text
block-3 = 0
recursive greedy output unchanged
```

Thus generic state reconstruction alone is insufficient.

---

# 37. Important revised low-rank interpretation

Do not now call rank-64 completely:

```text
FUNCTIONALLY INERT
```

That is too broad.

Correct distinction:

```text
GREEDY/OLD RECURSION IMPROVEMENT      NONE
TOP-K DISTRIBUTIONAL IMPROVEMENT      YES
```

---

# 38. B3-R hold curve

Replicated isolated hold-state curve:

|  Δ | Top-1 | Top-4 | Top-8 | Top-16 |    KL |
| -: | ----: | ----: | ----: | -----: | ----: |
|  1 | 26.8% | 48.2% | 57.1% |  66.1% | 5.395 |
|  2 | 13.4% | 27.7% | 35.7% |  46.4% | 5.562 |
|  3 | 17.0% | 23.2% | 28.6% |  33.0% | 6.084 |
|  5 |  9.8% | 18.8% | 24.1% |  29.5% | 7.539 |

---

# 39. B3-R rank-64 curve

The same artifact also reports:

|  Δ | Top-1 | Top-4 | Top-8 |    Top-16 |        KL |
| -: | ----: | ----: | ----: | --------: | --------: |
|  1 | 24.1% | 45.5% | 61.6% | **70.5%** | **4.602** |
|  2 | 17.0% | 37.5% | 46.4% | **53.6%** | **4.492** |
|  3 | 21.4% | 34.8% | 40.2% | **50.0%** | **4.431** |
|  5 |  9.8% | 23.2% | 32.1% | **39.3%** | **6.035** |

---

# 40. Why §39 matters

Rank-64 generally improves:

```text
KL
Top-8
Top-16
```

and at Δ2/3 also improves Top-1/Top-4.

At Δ1 it slightly hurts Top-1/Top-4 but improves:

$$
Top16:66.1\%\rightarrow70.5\%.
$$

That is potentially valuable for branch/tree proposal.

---

# 41. Rank-64 tree hypothesis

The current tree program should therefore compare:

```text
hold-state branches
vs
rank-64 predicted-state branches
```

rather than assuming the hold curve is the best tree baseline.

---

# 42. B3-R status

```text
VALID
REPLICATED
n=112 per Δ
BLACKWELL_SANDBOX_ONLY
```

---

# 43. Original B3-S artifact

The current sequential artifact contains:

```text
18 chains
3 domains:
    coding
    math_reasoning
    agent_tool_use

predictor:
    rank-64 PCA+ridge
```

---

# 44. Reported B3-S numbers

Artifact records:

```text
Δ1 E[A] 2.111
Δ2 E[A] 2.111
Δ3 E[A] 2.167
Δ5 E[A] 2.500
```

These values exist in the artifact.

They do **not** all have valid sequential semantics.

---

# 45. B3-S integrity failure

For Δ>1, after advancing the sequence position by Δ, the next speculative draft attends intermediate ring/KV positions belonging to committed accepted tokens.

The current chain did not populate all of those committed target KVs.

Therefore subsequent drafts could read:

```text
zeros
or
stale/predicted values from another chain/state
```

instead of the correct accepted target path.

---

# 46. Invalidated B3-S results

Therefore:

```text
Δ2 E[A]=2.111     INVALIDATED
Δ3 E[A]=2.167     INVALIDATED
Δ5 E[A]=2.500     INVALIDATED
```

for architectural comparison.

---

# 47. Retracted conclusion

Do not freeze:

> "Δ=5 is best."

Do not freeze:

> "The microstep hypothesis is inverted."

Neither is supported by the current valid sequential evidence.

---

# 48. Δ1 special case

The identified missing-intermediate-KV defect does not apply in the same way at:

$$
\Delta=1.
$$

Therefore current:

$$
\boxed{E[A]_{\Delta1}=2.111}
$$

is the only result from this B3-S run not invalidated by that specific defect.

---

# 49. Δ1 evidence class

Still classify it conservatively:

```text
B3S_DELTA1_PROVISIONAL_VALID
BLACKWELL_SANDBOX_ONLY
SMALL_SAMPLE
```

because:

```text
n = 18
only 3 domains
```

---

# 50. Δ1 current distribution

Artifact reports:

```text
coding          mean depth 3.00
math_reasoning  mean depth 2.50
agent_tool_use  mean depth 0.83

aggregate E[A]  2.111
p50             2
p90             4
reject          16.7%
max             4
```

---

# 51. Do not compare Δ1 directly to D1 yet

Although:

$$
2.111 > 1.933,
$$

the samples/geometries are not guaranteed matched.

Do not claim a 9% sequential improvement until evaluated on the same chains/prompts.

---

# 52. Correct B3-S fix

## B3-S-RFIX

For every accepted token:

1. retain the accepted-path target hidden state from `target_forward`;
2. derive/project the correct state required by the speculative path;
3. populate each required stage KV at:

   $$
   pos\bmod128
   $$

   for every committed token;
4. ensure all intermediate positions traversed by Δ are densely committed;
5. snapshot the relevant rings at chain start;
6. restore them for every independent chain from the correct target crop.

---

# 53. Ring invariant

Before every `forward_spec` call, all historical positions that the drafter is allowed to attend must correspond to:

```text
the actual committed target path
```

unless a position is explicitly part of the currently speculative branch.

No zeros/stale data may enter by accident.

---

# 54. New regression tests

Add tests that deliberately use:

```text
Δ2
Δ3
Δ5
```

and assert that every historical ring slot consumed by the next draft has been written by either:

```text
committed target state
or
declared speculative state
```

Never unspecified memory.

---

# 55. Poison test

A particularly strong regression:

1. prefill unused ring slots with obvious poison values;
2. run the rollout;
3. verify that no legal draft changes when poison values change.

If output changes, a hidden unwritten-slot dependency remains.

---

# 56. Chain isolation test

Run identical chains:

```text
A alone
B alone
A after B
B after A
```

and require A/B results to be invariant to previous chain execution.

This catches leftover-ring contamination.

---

# 57. B3-S-RFIX experiment

Rerun:

$$
\boxed{\Delta\in\{1,2,3,5\}}
$$

only after ring invariants pass.

---

# 58. B3-S-RFIX sample size

The current 18-chain run is too small for architectural freeze.

Prefer:

```text
all 7 domains
>=16 chains/domain if feasible
```

or at least enough to establish stable confidence intervals.

---

# 59. B3-S-RFIX outputs

For each Δ report:

```text
n
E[A]
confidence interval for E[A]
mean depth
p50
p90
p95
max
full-reject rate
depth survival
proposal latency
predictor latency
```

---

# 60. Matched-chain design

Use the **same starting chains** for every Δ.

This enables paired comparisons:

$$
A_{\Delta1,i}
,\;
A_{\Delta2,i}
,\;
A_{\Delta3,i}
,\;
A_{\Delta5,i}.
$$

Paired differences are much stronger than comparing independent means.

---

# 61. Hold vs rank-64 sequential race

Because B3-R shows meaningful Top-k differences, B3-S-RFIX should ideally include:

```text
hold
rank-64
```

for at least Δ1 and the eventual winning spacing.

This determines whether rank-64's distributional gain actually helps sequential acceptance.

---

# 62. B3-S decision gate

Only after B3-S-RFIX may we classify:

```text
best sequential Δ
greedy microstep viability
native-block viability
```

No spacing winner is currently frozen.

---

# 63. Current tree geometry status

The best evidence available **before corrected Mode R** still comes from B3-R.

That evidence favors:

```text
short displacement
+
Top-k branching
```

especially Δ1.

---

# 64. Current leading tree hypothesis

Until B3-S-RFIX says otherwise:

$$
\boxed{
\text{fresh root}
\rightarrow
\Delta1\text{ predicted state}
\rightarrow
\text{small shallow tree}
\rightarrow
\text{exact refresh}
}
$$

remains the leading hypothesis.

Not production design; hypothesis.

---

# 65. Do not pivot back to Δ5 tree yet

The invalid current Δ5 sequential result cannot justify:

```text
fresh root
→ native five-token block
→ tree
```

as the new preferred design.

That branch becomes eligible again only if corrected B3-S supports it.

---

# 66. B4-O timing

Do **not** build the authoritative sequential tree oracle from the corrupted B3-S Δ2/3/5 traces.

First complete B3-S-RFIX.

---

# 67. Static B4-O work allowed in parallel

A limited static oracle may still be built from valid B3-R distributions and labelled:

```text
B4_STATIC_ONE_STEP_ORACLE
```

It may estimate candidate-set coverage versus width.

It must not be treated as sequential tree accepted depth.

---

# 68. Authoritative B4-O

After corrected B3-S, build:

```text
fresh root
+
valid predicted child state
+
candidate branching
```

using actual valid sequential traces.

---

# 69. B4-O widths

Race:

```text
Top-2
Top-4
Top-8
Top-16
adaptive probability mass
adaptive entropy/margin
fixed node budgets
```

---

# 70. B4-O conditional quantity

Measure:

$$
P(
\text{target child survives}
\mid
\text{parent path correct}
).
$$

This is more relevant than unconditional B3-R Top-k.

---

# 71. B4-O frontier

For every candidate budget:

```text
candidate nodes
verification width
target-path survival
E[A]_oracle
unique experts
logical expert bytes
```

Produce a Pareto frontier.

---

# 72. B4-P production race

Only execute oracle-Pareto candidates.

Measure:

```text
T_proposal
T_target
T_total
E[A]
verification width
bank touch
H2D
RAM
NVMe
VRAM
thermals
```

---

# 73. B4-P objective

$$
\boxed{
R_{\rm tree}
=
\frac{E[A_{\rm real}]}
{
T_{\rm proposal}
+
T_{\rm target}
+
T_{\rm movement}
}
}
$$

---

# 74. Tree headroom requirement

Prefer a perfect-residency/tree P2 equivalent of at least:

$$
\boxed{35/s}
$$

and ideally:

$$
\boxed{40/s}
$$

to leave P3 overhead margin.

---

# 75. Bank-touch constraint

N112 already touches:

$$
46.1\%
$$

of the bank at p50.

Therefore also optimize:

$$
\boxed{
\frac{E[A]}
{\text{unique expert bytes}}
}
$$

not accepted depth alone.

---

# 76. BranchPack

BranchPack optimizes jointly:

$$
\text{expected acceptance}
-
\lambda_c(\text{target compute})
-
\lambda_b(\text{expert bytes}).
$$

---

# 77. Adaptive branching

Future policy inputs can include:

```text
entropy
top1/top2 margin
cumulative Top-k mass
rank-64 confidence
expert overlap
resident-status
movement estimate
```

---

# 78. Function-aware predictor

Still open.

If better state prediction is necessary:

$$
L
=
\lambda_sL_{\rm state}
+
\lambda_{KL}L_{\rm KL}
+
\lambda_kL_{\rm top-k}
+
\lambda_rL_{\rm rollout}.
$$

---

# 79. History-conditioned predictor

If `[state, token]` remains insufficient, incrementally add:

```text
previous state
state delta
recent tokens
proposal confidence
small logit summary
cheap recurrent state
```

---

# 80. Tiny nonlinear model

Still allowed.

Initial budget:

$$
0.5-1.5M
$$

parameters.

Not immediate NEXT.

---

# 81. True Jacobian work

Still optional.

A valid study requires:

```text
JVP
VJP
logit gradients
randomized low-rank probes
```

not directional finite differences.

---

# 82. P-C 512 interface

Still:

```text
UNTESTED
```

and off the current critical path.

---

# 83. Exact DSA Track A

Unchanged:

$$
\boxed{
\text{Exact DSA}
\rightarrow
\text{0xFF closure}
\rightarrow
\text{P2 rerun}
}
$$

---

# 84. Post-DSA P2

Rerun:

```text
N56
N80
N112
```

minimum.

Require:

```text
full mandatory target path
LM head
production representation
STEADY thermals
correctness classification
```

---

# 85. 3060 equivalence

When full checkpoint becomes available, immediately complete:

```text
32-token overlap
output tokens
Top-6 expert IDs
hash routes
route weights
```

---

# 86. Production routing

Until equivalence passes:

```text
AUTHORITATIVE_BLACKWELL_ROUTING
```

remains the correct label.

---

# 87. Full VRAM census

Before P3 freeze include:

```text
source-FP8 core
Mixed-A banks
KV/context
drafter
tree buffers
workspace
CUDA runtime
graphs
scheduler
LM head
temporary activations
```

---

# 88. CPU Dual-Rail

Still open.

Execute a RAM-resident expert on CPU whenever:

$$
T_{\rm CPU}
<
T_{\rm H2D}+T_{\rm GPU}.
$$

---

# 89. Sparse NVMe

Still measure:

```text
random expert reads
batched reads
superpages
queue depth
NVMe→pinned
NVMe→CPU
H2D overlap
page cache
```

---

# 90. 1M context census

Measure:

```text
4K
16K
64K
128K
256K
512K
1,048,576
```

for:

```text
KV/context VRAM
RAM
attention/indexer cost
TTFT
decode
expert residency
drafter/tree state
NVMe pressure
```

---

# 91. Likely P3 architecture

Pending corrected B3-S:

```text
source-FP8 exact V4
+
exact DSA
+
width-dispatched core
+
Mixed-A experts
+
packed4 canonical cold pool
+
DSpark/NanoSpark proposal layer
+
winning corrected speculative geometry
+
BranchPack
+
exact parallel verification
+
RAM warm tier
+
CPU Dual-Rail
+
NVMe cold tier
+
async prefetch
```

Do not freeze Δ1 or Δ5 into P3 yet.

---

# 92. P3 equation

$$
T_{\rm block}
=
T_{\rm proposal}
+
T_{\rm target}
+
T_{\rm GPU}
+
T_{\rm CPU}
+
T_{\rm H2D}
+
T_{\rm NVMe}
+
T_{\rm scheduler}
-
T_{\rm overlap}.
$$

$$
R_{\rm output}
=
\frac{E[A_{\rm real}]}{T_{\rm block}}.
$$

---

# 93. P4 reporting

Report:

```text
final output tok/s
verification tok/s
E[A]
proposal geometry
tree width/depth
proposal latency
target latency
expert-bank touch
H2D
RAM traffic
NVMe traffic
CPU expert fraction
VRAM
context
temperature
clocks
power
```

---

# 94. P4 success

$$
\boxed{\ge30\text{ sustained final output tok/s}}.
$$

---

# 95. Exact-30-1M

Ultimate gate:

$$
\boxed{
p50_{\rm output}(1,048,576)\ge30
}
$$

with:

```text
SOURCE_EXACT
GOLDEN_GREEDY_PASS
NO_QUALITY_LOSS_VERIFIED
STEADY thermal qualification
```

---

# 96. Closed branches

Closed absent contradictory evidence:

```text
generic expert compression
simple expert deduplication
generic wide-K LRU
global Hot8 residency
two-call hybrid
partial masks as primary short-context accelerator
permanent BF16 core
width-flat core
BM128/grid_m cure
v5 split-K
generic SDPA
broad GEMM race

damped velocity as production predictor
generic state MSE as sufficient objective
native Case A → Exact-30
deep stale-state tree as primary design
```

---

# 97. Retracted / invalidated claims

Do not use:

```text
"Δ5 is best sequentially"
"microstep hypothesis is inverted"
"Δ5 E[A]=2.5 is valid B3-S evidence"
"Δ2/3/5 B3-S define sequential spacing ranking"
```

until B3-S-RFIX.

---

# 98. Open branches

```text
corrected Δ1/2/3/5 sequential race
shallow Δ1 tree
possible corrected Δ5/native-block tree
rank-64 Top-k proposal
adaptive BranchPack tree
function-aware predictor
history-conditioned predictor
P-C 512
```

---

# 99. Immediate execution order

## NEXT 1 — Fix B3-S harness

Implement committed target KV writes + ring snapshot/restore.

## NEXT 2 — Run invariant/poison tests

Do not benchmark until these pass.

## NEXT 3 — B3-S-RFIX

Rerun matched Δ1/2/3/5 sequential chains.

## NEXT 4 — B4-O

Build oracle using only valid traces.

## NEXT 5 — B4-P

Run production-shaped tree economics.

## PARALLEL — Track A

Exact DSA → 0xFF → P2.

## INTERRUPT WHEN AVAILABLE

3060 equivalence.

---

# 100. Updated bottleneck ranking

```text
1. B3-S sequential harness correctness
2. Exact DSA / 0xFF semantic closure
3. corrected sequential Δ economics
4. shallow-tree oracle
5. production tree economics
6. 3060 equivalence
7. full 12GB runtime fit
8. BranchPack movement
9. function/history-aware prediction if required
10. CPU Dual-Rail
11. sparse NVMe
12. 1M context
13. production scheduler
```

---

# 101. Engineering progress

```text
Source/reference semantics              ~97–98%
QPA                                     100%
MXIE                                    ~98%
Expert compute                          ~95%
Transport                              100%

Core GEMM                               ~95%
Production target core                  ~88–90%
Exact DSA                               ~70–75%
TRUE P2 certification                   ~80–85%

Blackwell routing                       ~95%
Production routing authority            ~65%

DSpark characterization                 ~96%
Case-A evaluation                       100% / CLOSED
NanoSpark state research                ~95%
B3 isolated spacing                     ~98%
B3 sequential geometry                  ~65%
Tree geometry                           ~60%
Production drafter                      ~55–60%

Production residency                    ~65%
P3 runtime                              ~62–65%
P4 actual output                        ~42–45%
1M qualification                        ~25%

OVERALL EXACT-30 ENGINEERING            ~89–90%
```

The invalid Δ2/3/5 run should not increase completion.

The valid Δ1 result adds useful evidence, but the harness repair adds required work.

---

# 102. Current valid drafter truth

```text
Fresh DSpark:
Top1                     71.4%
Top16                    96.7%

B3-R HOLD Δ1:
Top1                     26.8%
Top4                     48.2%
Top8                     57.1%
Top16                    66.1%

B3-R RANK64 Δ1:
Top1                     24.1%
Top4                     45.5%
Top8                     61.6%
Top16                    70.5%
KL                       4.602

B3-S sequential:
Δ1 E[A]                  2.111 provisional-valid
Δ2                       INVALID
Δ3                       INVALID
Δ5                       INVALID
```

---

# 103. Current central insight

The most interesting new valid result is not Δ5.

It is:

> Rank-64 does not substantially improve greedy one-step selection at Δ1, but it improves the broader candidate distribution—especially Top-8/Top-16 and KL.

That makes NanoSpark increasingly plausible as:

$$
\boxed{\text{a branch-distribution improver}}
$$

rather than:

$$
\boxed{\text{a greedy-state predictor}}.
$$

---

# 104. Architecture status

The leading hypothesis remains provisionally:

$$
\boxed{
\text{fresh exact state}
\rightarrow
\text{short predicted transition}
\rightarrow
\text{small cost-aware branch set}
\rightarrow
\text{exact verification}
\rightarrow
\text{refresh}
}
$$

But the correct transition spacing must be re-established by B3-S-RFIX.

---

# 105. Canonical NEXT

$$
\boxed{
\text{repair B3-S ring/KV semantics}
\rightarrow
\text{poison + chain-isolation tests}
\rightarrow
\text{B3-S-RFIX}
\rightarrow
\text{B4-O}
\rightarrow
\text{B4-P}
}
$$

in parallel with:

$$
\boxed{
\text{Exact DSA}
\rightarrow
\text{0xFF closure}
\rightarrow
\text{TRUE P2 rerun}
}
$$

and when unblocked:

$$
\boxed{\text{3060 equivalence}}
$$

then:

$$
\boxed{
\text{production geometry freeze}
\rightarrow
P3
\rightarrow
P4
\rightarrow
Exact\text{-}30
\rightarrow
Exact\text{-}30\text{-}1M
}
$$

This roadmap supersedes all previous MosaicVM roadmaps.
