This is the most favorable proposer result so far. D1-B closes the **tail-fragility risk**: through 30% relative \(h_{40}\) error, every tested perturbation family retained 100% Top-4/Top-16, at least 96.5% Top-1, and \(E[A]\ge7.15/9\) on K=8. Even much larger errors degraded gracefully rather than hitting a cliff.  

The route result is also useful, but I would phrase it carefully: L41 route exact-match can fall to roughly 15–40% while token quality remains excellent, so **the state student does not need to preserve exact tail routes in this tested basin**. That does not yet prove the tail MoE can be removed. 

One major consequence needs adding to the roadmap: **K=8 was only a robustness probe.** Under the frozen prefix+1 semantics, perfect K=8 gives \(E[A]=9\), as the artifact confirms.  Production therefore needs D1-B repeated at the actual speculative horizon—likely K32 or whatever D2/H2 ultimately makes optimal.

# MosaicVM Canonical Master Roadmap

## D1-B PASS — Future-h40 + Exact Source Tail Is Now the Primary Proposer Architecture

**Target:** DeepSeek-V4-Flash-0731
**Hardware:** RTX 3060 12 GB + ~24 GB RAM + NVMe
**Primary objective:** ≥30 sustained final output tok/s
**Ultimate objective:** Exact-30 at 1,048,576-token context

**Status:** CANONICAL — supersedes every prior MosaicVM roadmap.

---

# 1. Executive state

```text
CORRECTNESS
source-class target                       COMPLETE
G2                                        CERTIFIED 2/2

GPU
MXIE                                      COMPLETE
imma4                                     CERTIFIED
TRUE-P2                                   0.297 + 8.36ms·N

MOVEMENT
expert identity                           (layer, expert)
routed expert universe                    ~147.6 GB
pure GPU+H2D                              CLOSED
pure residency                            CLOSED
naive CPU                                 CLOSED

CPU
VNNI v4-B                                 EXACT
isolated warm M3                          0.378 TFLOPS-eq
cold DRAM                                 0.21 TFLOPS-eq
real-contention H1                        ~0.246 TFLOPS-eq
H1 wall                                   ~1.11 s
production-correct H2                     OPEN

OLD PROPOSERS
static A/B/C                              CLOSED
GRU                                       CLOSED
T0 causal                                 ~1.50 E[A]
T1 multi-state                            ~1.57 E[A]
BranchPack from T1 logits                 CLOSED

DEEP-TAIL PROGRAM
D1-C: L42 + FP32 head                     100% greedy Top1
D1-R: true h40 → source L41/L42           BIT-EXACT
D1-B: approximate h40 → source tail       ROBUST / PASS

D1-B K=8
exact                                      E[A] = 9.0
rel-L2 ≤0.3                               E[A] = 7.15–9.0
Top1                                       ≥96.5%
Top4                                       100%
Top16                                      100%

PRIMARY ARCHITECTURE
future-h40 state predictor
→ exact source L41/L42 tail
→ FP32 head
→ optional BranchPack
→ exact full verifier

MAIN OPEN GATES
D2      actual tail economics
D3      real future-h40 student
DATA    large cross-domain state corpus
H2      production-correct target hybrid
D1-B32  robustness at production horizon
```

---

# 2. Exactness remains closed

The exact verifier continues to preserve source FP8/FP4 arithmetic, source routing, all six Top-6 experts, source attention/DSA, source SwiGLU, shared expert, ordered accumulation and FP32 production head.

The future-h40 predictor remains an approximate draft-only component.

No prediction error can alter final model correctness because all draft tokens are verified.

---

# 3. TRUE-P2 remains frozen

$$
T_{\rm P2}(N)=0.297+0.00836N.
$$

For N32:

$$
T_{\rm P2}(32)\approx0.5645s.
$$

This remains the perfect-residency compute floor, not production P3.

---

# 4. Expert universe remains corrected

Routed expert identity is:

$$
(\ell,e).
$$

Full routed expert storage remains approximately:

$$
147.6GB.
$$

---

# 5. Pure GPU streaming remains closed

Real N32 compulsory routed-expert traffic is approximately:

$$
34.1GB.
$$

At the measured ~12.9 GB/s H2D rate:

$$
T_{\rm H2D}\approx2.64s.
$$

Pure GPU+H2D cannot satisfy Exact-30.

---

# 6. Pure residency remains closed

Belady already established compulsory first-touch traffic that cannot be eliminated by cache policy.

Small strategic residency remains legal inside H2.

---

# 7. VNNI remains the exact CPU rail

V4-B tiled remains frozen unless new measured evidence beats it.

Measured regimes remain:

```text
warm        0.378 TFLOPS-eq
cold        0.210 TFLOPS-eq
contended   ~0.246 TFLOPS-eq
```

---

# 8. H1 remains useful but non-final

H1 measured approximately:

```text
GPU rail        0.662 s
CPU rail        1.110 s
wall            1.113 s
```

It established the magnitude of host-memory contention.

It remains non-production-final because the CPU workload was unusually warm while GPU-assigned production H2D was omitted.

---

# 9. H2 remains required

H2 must use genuine `(layer,expert)` requests, real row counts, real GPU misses, real H2D, real CPU DRAM traffic and real strategic residency.

The fixed 68% CPU object split remains retired.

---

# 10. T0/T1 are now historical baselines

T0 and T1 proved that direct token prediction from shallow state is weak.

They are no longer the production proposer architecture.

---

# 11. Current T1 trees remain closed

Oracle BranchPack analysis showed that the current T1 logits simply do not contain enough conditional target probability mass.

Do not spend more engineering effort widening those trees.

---

# 12. D1-C remains frozen

True L42 with the production FP32 head yields the exact greedy target decision.

Canonical:

```text
FINAL-STATE TOKEN CEILING = 100%
```

The earlier BF16 exit-head result is permanently superseded.

---

# 13. D1-R remains frozen

True future h40 trajectories replayed sequentially through the actual source L41/L42 produce the reference h42 bit-exactly.

Thus:

$$
h_{40,1:K}
\xrightarrow{F_{41:42}}
h_{42,1:K}
$$

is a valid exact interface.

---

# 14. No extra tail-state prediction is needed

The student does not need to output L41 caches, L42 caches, tail routes, final logits or future h42 explicitly.

The source tail regenerates its own sequential KV/DSA state from the h40 trajectory.

---

# 15. D1-B closes tail fragility

The tested exact source tail is highly tolerant of imperfect h40 trajectories.

Canonical:

```text
TAIL FRAGILITY = CLOSED ON INITIAL TEST SET
```

with the important caveat that large-scale cross-domain validation remains open.

---

# 16. D1-B production semantics

Only speculative positions were perturbed.

The committed boundary stayed exact.

This correctly models the intended production handoff.

---

# 17. D1-B exact baseline

At K=8:

$$
E[A]=9.
$$

This is the prefix+1 maximum for eight speculative positions.

---

# 18. D1-B good-state basin

For:

$$
\text{rel-L2}\le0.3,
$$

corresponding roughly to cosine ≥0.955 in the reported geometry:

```text
Top1       96.5–100%
Top4       100%
Top16      100%
E[A]       7.15–9.00
```

This is now the initial student target region.

---

# 19. Moderate state error

At approximately:

$$
\text{rel-L2}=0.5,
$$

the tested families still produce approximately:

```text
Top1       93–96.5%
Top4       98–100%
Top16      100%
E[A]       ~6.0–7.2
```

at K=8.

---

# 20. Very large state error

Even near:

$$
\text{rel-L2}=1.0,
$$

the tail remains surprisingly useful in the small test:

```text
Top1       ~82–95%
Top4       ~93–100%
Top16      ≥96.5%
E[A]       ~5.6–6.5
```

depending on error family.

---

# 21. No measured cliff

The extended sweep through approximately rel-L2 1.5 showed graceful degradation rather than a sharp transition.

Therefore the future-h40 target is a **basin problem**, not a bit-exact reconstruction problem.

---

# 22. Student specification changes fundamentally

The student does not need:

```text
bit-exact h40
route-exact h40
near-perfect hidden cosine
```

as an a priori requirement.

It needs a trajectory that lands inside the downstream source-tail decision basin.

---

# 23. Route preservation is demoted

Exact L41 routes can change substantially before token quality degrades.

Therefore:

```text
EXACT ROUTE MATCH = NOT A STUDENT SUCCESS METRIC
```

for the current primary architecture.

---

# 24. Important route-language correction

Do **not** conclude:

```text
tail MoE is unnecessary
```

from D1-B.

The valid conclusion is:

> multiple nearby tail routing configurations can preserve the same token decision on the tested trajectories.

Removing/replacing the MoE would require a separate ablation.

---

# 25. Router loss is demoted

Do not add a strong route-preservation auxiliary loss to D3 by default.

Only restore it if large-scale real student residuals show route errors correlate with token failures.

---

# 26. Root reuse is promoted

The committed-boundary root remained exact across D1-B configurations.

Thus root proposal can use exact previously computed boundary information where the cycle semantics make it available.

---

# 27. Root cost target

Canonical goal:

```text
root proposal incremental compute ≈ zero
```

subject to final integration proof.

---

# 28. Main remaining proposer problem

The research problem is now:

$$
\boxed{
\text{predict future }h_{40}
\text{ trajectories cheaply enough and accurately enough}
}
$$

for the real source tail to decode them.

---

# 29. K=8 is not the production horizon

D1-B establishes local robustness only.

K=8 cannot itself provide enough accepted work per verifier cycle for Exact-30 with current target costs.

---

# 30. Frozen prefix+1 convention

For K speculative positions:

$$
E[A]_{\max}=K+1.
$$

Thus:

```text
K8 exact ceiling   = 9
K16 exact ceiling  = 17
K32 exact ceiling  = 33
```

---

# 31. Important throughput consequence

With K32 and perfect acceptance:

$$
E[A]_{\max}=33.
$$

For 30 tok/s, total cycle wall must satisfy:

$$
T_{\rm cycle}\le\frac{33}{30}
=
\boxed{1.10s}.
$$

---

# 32. Consequence for H1

H1 already measured about:

$$
1.113s
$$

before adding student/tail overhead.

Therefore:

```text
H1 + perfect K32 proposal = STILL INSUFFICIENT
```

by a small margin.

H2 optimization is mandatory even if D3 becomes excellent.

---

# 33. Draft-overhead budget becomes concrete

If student + tail cost:

```text
50ms   → verifier must be ≲1.05s at perfect K32
100ms  → verifier must be ≲1.00s
150ms  → verifier must be ≲0.95s
```

for the absolute 33-token ceiling.

Real acceptance below 33 requires proportionally lower wall time.

---

# 34. D1-B32

## PRODUCTION-HORIZON ROBUSTNESS

Repeat D1-B at longer speculative horizons.

Required minimum:

```text
K8
K16
K32
```

and optionally K48/K64 if D2/H2 makes them economically interesting.

---

# 35. Why D1-B32 matters

Trajectory error and tail-state divergence may accumulate over 32 positions even though K8 is robust.

Do not extrapolate K8 \(E[A]\) directly to K32.

---

# 36. D1-B32 perturbation set

Retain the same four core families:

```text
isotropic/Gaussian
directional
scale
temporally correlated AR-like
```

and add real D3 residuals as soon as available.

---

# 37. D1-B32 primary output

For every state-error bucket report:

```text
S1 S2 S4 S8 S16 S24 S32
E[A]
Top1/4/16 by horizon
```

plus final tail-state drift.

---

# 38. Conditional-accuracy interpretation

A marginal 96–99% per-position Top1 result does not automatically imply long prefix survival.

Long-horizon prefix correlation must be measured directly.

---

# 39. Illustrative long-chain requirement

If one pretended for intuition that conditional correctness were constant and independent, roughly:

```text
~96.5% conditional  → E[A] around 20 at K32
~98%                → E[A] around 24
~99%                → E[A] around 28
~99.5%              → E[A] around 30
```

These are only geometric illustrations.

Actual D1-B32 survival is authoritative.

---

# 40. D2

## NEXT SYSTEM GATE FOR THE PROPOSER

Measure the real cost of source L41/L42 execution.

---

# 41. Tail routed expert bank

Approximate routed-expert storage:

$$
2\times256\times13.4MB
\approx6.86GB.
$$

This fits in host RAM.

---

# 42. Tail NVMe policy

If complete measured tail footprint fits RAM:

```text
TAIL NVMe AFTER WARMUP = ZERO
```

should become a production invariant.

---

# 43. Full D2 memory inventory

Measure:

```text
routed experts
shared experts
attention/DSA/core weights
routers/norms
scratch
tail KV/DSA state
```

before freezing memory placement.

---

# 44. First-order routed-tail work

Using approximately 0.42 TFLOP of routed-expert work for all 43 layers:

$$
0.42\frac{2}{43}
\approx0.0195\text{ TFLOP}
$$

for two comparable layers at N32.

---

# 45. Prior tail estimate correction remains frozen

Do not multiply by all 256 experts.

Only routed Top-6 work executes.

---

# 46. D2 must measure rather than extrapolate

The 50–95 ms routed-expert estimates exclude attention, shared expert, core linears and sequencing.

No final tail latency is frozen yet.

---

# 47. D2 geometry race

Benchmark:

```text
serial one-position tail
K4 chunk
K8 chunk
K16 block
K32 block
K48 block
K64 block
```

where semantics permit.

---

# 48. D2 placement race

Benchmark:

```text
CPU VNNI
GPU + H2D
strategic VRAM tail subset
hybrid CPU/GPU
```

---

# 49. Tail CPU advantage

All tail expert weights can potentially remain host-RAM resident.

This removes NVMe from the draft-tail critical path.

---

# 50. Tail GPU disadvantage

GPU placement may compete with verifier-core residency and incur H2D.

Do not assume GPU is faster system-wide.

---

# 51. Tail CPU disadvantage

Tail VNNI and verifier VNNI share host-memory bandwidth.

D2-in-isolation is insufficient.

---

# 52. D2-H2 coexistence test

After isolated D2, run tail CPU work concurrently with production H2 scheduling.

Measure whether the tail steals critical bandwidth from the verifier.

---

# 53. Tail scheduling priority

Because tail latency blocks speculative tokens, it may deserve deadline priority over non-critical verifier prefetch.

Measure rather than hardcode.

---

# 54. DATA

## NOW A PRIMARY WORKSTREAM

D1-B makes larger-scale state training worth doing.

Generate at least:

$$
1000+\text{ distinct prompts}.
$$

---

# 55. Do not confuse prompts with windows

Capture many committed boundaries per prompt.

The training corpus should ideally contain tens of thousands or more state-trajectory windows.

---

# 56. Domain coverage

Include:

```text
coding
general conversation
math
structured JSON
agent/tool use
long generation
long context
```

rather than remaining coding-only.

---

# 57. Strict split discipline

Train/validation/test split by prompt.

Near-duplicate trajectories must remain on the same side of the split.

G1/G2 correctness data remains excluded from training.

---

# 58. Store per committed boundary

At minimum:

```text
current h40
recent h40 history
future h40 trajectory
future h42 trajectory
future target tokens
L41/L42 routes
FP32 head Top-k/margins
position/domain metadata
```

---

# 59. D1-B-scale validation

Before celebrating the current robustness basin as universal, repeat on the DATA corpus.

The current evidence is three coding prompts only.

---

# 60. D3

## PRIMARY LEARNING GATE

Train a model whose primary output is future h40.

---

# 61. D3 deployment target

Initial favorable target:

$$
\boxed{\text{rel-L2}\lesssim0.3}
$$

on real held-out trajectories.

But final promotion depends on downstream E[A], not hidden loss.

---

# 62. D3 softer useful target

D1-B suggests even rel-L2 around 0.5–1.0 may still yield meaningful candidate quality.

Therefore do not kill a student merely because cosine is below 0.955.

Replay it through the real tail first.

---

# 63. D3 primary validation

Every checkpoint must be evaluated as:

```text
predicted h40
→ real source tail
→ FP32 head
→ actual speculative acceptance
```

---

# 64. Hidden metrics are secondary

Track:

```text
relative L2
cosine
state norm error
```

for diagnostics.

Do not select models solely by those metrics.

---

# 65. D3 architecture race

At minimum race:

```text
causal latent transformer
state-space/recurrent trajectory predictor
latent trajectory transformer
```

on identical data.

---

# 66. Open-loop D3

Predict all future h40 states in one pass.

Potential advantage:

```text
maximum parallelism
cheap student execution
efficient block tail
```

Primary risk:

```text
long-horizon drift
```

---

# 67. Closed-loop D3

Predict next h40, run tail, consume decoded token, repeat.

Potential advantage:

```text
feedback correction
```

Primary risk:

```text
serial student + tail latency
```

---

# 68. Chunked D3

Predict 4–8 h40 states at a time, replay tail, feed results back, repeat.

This is now a particularly strong candidate because D1-B already demonstrates K8 robustness.

---

# 69. D3 architecture priority

Recommended research order:

```text
1. chunked K4/K8
2. open-loop K32
3. fully serial
```

unless early latency measurements change the economics.

---

# 70. Why chunked first

It can combine:

```text
tail batching
token feedback
bounded state drift
```

without requiring a perfect 32-step open-loop trajectory.

---

# 71. D3 capacity race

Train:

```text
small production candidate
medium candidate
larger research teacher
```

on the same state corpus.

---

# 72. Capacity interpretation

If larger works and smaller fails:

```text
LEARNABILITY = PASS
COMPRESSION = OPEN
```

Do not misclassify this as a state-prediction failure.

---

# 73. Loss design

Start with normalized future-h40 reconstruction plus downstream supervision.

Potential objective:

$$
L=
\lambda_hL_{h40}
+\lambda_{42}L_{h42}
+\lambda_{\rm KL}L_{\rm logits}
+\lambda_{\rm rank}L_{\rm rank}.
$$

---

# 74. Route loss remains optional

Because D1-B shows strong token robustness despite route changes, route preservation is not a default objective.

Add it only if real residual analysis proves useful.

---

# 75. Tail-aware student training

If practical, optimize errors according to downstream tail sensitivity.

Two h40 errors with equal L2 can have different token consequences.

---

# 76. Real-residual D1-B

As soon as D3 produces a checkpoint:

```text
collect actual h40 residual trajectories
→ replay source tail
→ place them onto the synthetic robustness curve
```

This is mandatory.

---

# 77. Important caveat on the old predictor comparison

Do not freeze:

```text
old cosine 0.57 ⇒ guaranteed E[A] 5.6–6.5
```

as a production claim.

Cosine alone does not determine residual structure.

It is only encouraging evidence that the required state basin may be reachable.

---

# 78. Direct h42 student

Now demoted to fallback.

Use only if future-h40 training proves unexpectedly difficult or source-tail economics are poor.

---

# 79. Direct logit student

Also fallback.

The exact source tail already gives a strong inductive decoder, so replacing it should require evidence.

---

# 80. Learned h40→h42 bridge

Remain as a possible later compression of the source tail.

Not a primary task while D2 remains unmeasured.

---

# 81. BranchPack

Current T1 BranchPack remains closed.

---

# 82. BranchPack reopening criterion

Re-run tree oracle only on logits produced by:

```text
real D3 h40 predictions
→ source L41/L42
→ FP32 head
```

---

# 83. Why branching may become useful

D1-B already shows Top-4/Top-16 candidate coverage can remain much stronger than Top-1.

If that persists with real student errors, shallow adaptive branching could turn residual state error into additional acceptance.

---

# 84. But do not branch automatically

A 96–99% Top-1 linear continuation may already beat the added tree verification cost.

Price the actual geometry.

---

# 85. Linear speculation is reopened for the new proposer

The old L40/T1 linear chain remains closed.

A future-h40 + real-tail proposer with sufficiently high conditional Top-1 can use a simple linear draft again.

---

# 86. H2

## PRODUCTION-CORRECT TARGET SYSTEM

Continue in parallel with D2/D3.

---

# 87. H2 split sweep

Balance by measured FLOPs/bytes/deadlines, not object count.

---

# 88. H2 contention sweep

Race:

```text
continuous CPU+DMA overlap
DMA bursts
CPU bursts
layer-phased execution
queue-adaptive execution
```

---

# 89. H2 strategic residency

Optimize:

$$
\text{critical-path milliseconds saved per VRAM MB}.
$$

---

# 90. H2 width sweep

Benchmark target verification widths:

```text
N8
N16
N32
N48
N64
```

and wider if candidate geometry justifies it.

---

# 91. Width definition must be explicit

Future reports must distinguish:

```text
speculative depth K
candidate-node count
target batch width N
accepted-token count
```

Do not use N/K interchangeably.

---

# 92. H2 + D2 unified cost model

Once D2 exists, build one scheduler model incorporating:

```text
draft-tail CPU traffic
verifier CPU traffic
H2D
GPU expert/core work
VRAM residency
host RAM bandwidth
```

---

# 93. Canonical throughput objective

For architecture \(G\):

$$
\boxed{
R(G)=
\frac{E[A](G)}
{T_{\rm student}(G)+
 T_{\rm tail}(G)+
 T_{\rm verifier}(G)+
 T_{\rm unhidden}(G)}
}.
$$

---

# 94. Success criterion

$$
\boxed{
R\ge30\text{ sustained final tok/s}.
}
$$

---

# 95. K32 ideal ceiling

Under prefix+1:

$$
E[A]_{\max}=33.
$$

Thus even ideal proposal gives only:

$$
1.10s
$$

of total cycle budget at 30 tok/s.

---

# 96. This raises H2 priority

D1-B made the proposer dramatically more plausible.

Movement is therefore again nearly co-equal with D3 as a remaining risk.

---

# 97. Production geometry search

Do not presuppose K32.

Race complete cycles at:

```text
K8
K16
K24
K32
possibly K48/K64
```

once D2/D3/H2 models are available.

---

# 98. Why K may exceed 32

A longer draft can increase accepted work/cycle if state prediction remains robust and target verification scales efficiently.

---

# 99. Why K may be below 32

Long trajectories may increase student drift, tail cost and verifier width enough to reduce throughput.

---

# 100. Exact cycle optimization

Choose:

$$
K^*
=
\arg\max_K
\frac{E[A](K)}
{T_{\rm cycle}(K)}.
$$

No speculative width is sacred.

---

# 101. Newly closed branches

```text
tail-fragility hypothesis on initial test
need for bit-exact future h40
need for route-exact future h40
route-preservation loss as mandatory
direct h42 as primary path
direct logits as primary path
```

---

# 102. Newly promoted/frozen branches

```text
future-h40 + exact source tail       PRIMARY
root reuse                           PROMOTED
chunked latent prediction            PROMOTED
long-horizon D1-B                    REQUIRED
real-residual tail testing           REQUIRED
D2 tail economics                    CO-PRIMARY
D3 state student                     CO-PRIMARY
H2 production hybrid                 CO-PRIMARY
```

---

# 103. Still closed

```text
pure GPU+H2D
pure residency
naive CPU
VNNI v3
VNNI packed-only
static A/B/C
GRU
T0/T1 production proposer
current T1 tree
irreducible final-token entropy
```

---

# 104. Current production candidate

```text
exact committed boundary
→ exact/free root reuse
→ chunked/open-loop future-h40 student
→ exact source L41/L42 tail
→ FP32 draft logits
→ optional shallow branch only if useful
→ exact heterogeneous V4 verifier
    • imma4 GPU
    • VNNI v4-B CPU
    • selective H2D
    • strategic residency
→ ordered exact reduction
→ RAM/NVMe backing
```

---

# 105. Immediate NEXT — D2

Measure actual tail cost now.

```text
full tail memory inventory
→ RAM residency
→ K1/K4/K8/K16/K32
→ CPU vs GPU vs hybrid
→ isolated and contended wall time
```

---

# 106. Immediate NEXT — D1-B32

Extend the robustness test to production-scale horizons.

```text
K16
→ K32
→ optional K48/K64
```

using the same production semantics.

---

# 107. Immediate NEXT — DATA

Generate the large cross-domain deep-state corpus.

Do not wait for D2 before starting data capture.

---

# 108. Immediate NEXT — D3

As soon as enough corpus exists:

```text
train future-h40 student
→ first focus K4/K8 chunked state prediction
→ real-tail decode
→ actual E[A]
```

---

# 109. Immediate NEXT — H2

Finish the real hybrid verifier:

```text
real expert diversity
real H2D
real CPU traffic
split sweep
contention scheduler
strategic residency
```

---

# 110. D2 decision

### Cheap tail

Keep real L41/L42 in production.

### Moderate tail

Optimize batching/chunking and joint scheduling.

### Expensive tail

Begin learned bridge/direct-h42 compression race.

---

# 111. D3 decision

### Real residual rel-L2 ≤0.3

Strong PASS; pursue longer horizon immediately.

### rel-L2 0.3–0.5

Still promising; evaluate actual E[A] before scaling.

### rel-L2 0.5–1.0

Not automatically a failure because D1-B basin remains broad.

### Large error + poor tail acceptance

Escalate student capacity/architecture.

---

# 112. D1-B32 decision

### K32 acceptance remains near ceiling

Linear future-h40 drafting becomes primary final decoding geometry.

### Top1 drops but Top-k stays strong

Reopen BranchPack.

### Long-horizon collapse

Use smaller chunks with feedback.

---

# 113. H2 decision

Do not use an arbitrary ≤0.85s pass line.

The pass criterion is whether the measured verifier cost, combined with measured D2/D3 acceptance, permits:

$$
R\ge30.
$$

---

# 114. Integrated P3 trigger

Start full P3 once all four are real:

```text
student latency
tail latency
held-out E[A]
production-correct verifier latency
```

---

# 115. P3 milestones

```text
≥10 final tok/s
≥20
≥25
≥30
```

---

# 116. P4

Success requires sustained testing with:

```text
actual RTX 3060
actual CPU
~24GB RAM
actual NVMe
real future-h40 student
real source tail
real exact verifier
STEADY thermal state
```

---

# 117. P4 success

$$
\boxed{\ge30\text{ sustained final output tok/s}}.
$$

---

# 118. 1M qualification

Only after short-context Exact-30:

$$
4K
\rightarrow16K
\rightarrow64K
\rightarrow128K
\rightarrow256K
\rightarrow512K
\rightarrow1,048,576.
$$

---

# 119. 1M measurements

At each context size record:

```text
VRAM
RAM
NVMe traffic
DSA/KV footprint
student state history
tail state
E[A]
cycle wall
final tok/s
thermals
correctness
```

---

# 120. Current bottleneck ranking

```text
1. D2 real tail cost
2. D3 future-h40 real-student residuals
3. D1-B32 long-horizon robustness
4. H2 production-correct verifier wall
5. DATA scale/generalization
6. student model compression/latency
7. optimal chunk/horizon geometry
8. BranchPack only if residual Top-k requires it
9. integrated P3
10. P4 / Exact-30-1M
```

---

# 121. Research understanding

Current short-context architecture understanding:

$$
\boxed{\sim99.9995\%}.
$$

The unknown is no longer where useful information lives or whether the tail can tolerate approximation.

It is whether a cheap learned state trajectory plus measured system costs produce enough accepted work per second.

---

# 122. Engineering/delivery progress

Current defensible value:

$$
\boxed{\mathbf{90-91\%}}.
$$

---

# 123. Why progress rises

D1-B closes a potentially fatal requirement:

> future h40 does not need extreme or bit-exact accuracy.

The source tail has a broad decision basin on every tested perturbation family.

---

# 124. Why progress is not higher

The favorable result is still based on:

```text
3 prompts
coding domain
K=8
synthetic perturbations
```

rather than a real trained student's K32 cross-domain trajectories.

D2 and production H2 are also still unknown.

---

# 125. After D1-B32 cross-domain PASS

If the basin persists at K32 and across the DATA corpus:

$$
\boxed{92-93\%}.
$$

---

# 126. After D2 viable economics

$$
\boxed{93-94\%}.
$$

---

# 127. After real D3 student reaches useful K32 E[A]

$$
\boxed{95-96\%}.
$$

---

# 128. After H2 + D2 + D3 jointly project ≥30

$$
\boxed{96-98\%}.
$$

---

# 129. Integrated P3 ≥20

$$
\boxed{97\%+}.
$$

---

# 130. Integrated P3 25–30

$$
\boxed{98-99\%}.
$$

---

# 131. Completion

Only:

$$
\boxed{100\%}
$$

when P4 measures ≥30 sustained actual final output tok/s.

---

# 132. Canonical NEXT

Run these in parallel:

```text
D2
real L41/L42 tail wall
→ memory
→ placement
→ chunk geometry
→ contention

D1-B32
K16/K32 robustness
→ synthetic families
→ later real D3 residuals

DATA
1000+ cross-domain prompts
→ tens of thousands state windows

D3
future-h40 predictor
→ start K4/K8 chunks
→ real-tail acceptance

H2
production-correct heterogeneous verifier
→ real H2D
→ real routes
→ scheduler/residency
```

Then:

```text
measure:
E[A](K)
Tstudent(K)
Ttail(K)
Tverifier(K)

optimize:
R(K) = E[A](K) / Tcycle(K)

IF linear Top1 remains excellent:
    use linear future-h40 drafting

ELIF Top-k strong but Top1 insufficient:
    reopen adaptive BranchPack

IF long-horizon drift appears:
    shrink to chunked K4/K8 feedback

IF source tail too expensive:
    distill/compress L40→L42

THEN:
integrated P3
→ 10
→ 20
→ 25
→ 30+
→ P4
→ Exact-30
→ Exact-30-1M
```

This roadmap supersedes every previous MosaicVM roadmap.

The strategic picture is now substantially better. The report's operating range explicitly shows rel-L2 ≤0.3 corresponding to roughly \(E[A]=7.2–9.0\) at K=8, while even much noisier states remain useful.  The committed root is also exact across the experiment, which supports the root-reuse architecture. 

The one thing I would **not** do now is jump straight from “K8 \(E[A]\approx8\)” to “the proposer is solved.” Production depends on whether that wide basin persists over **K16/K32 trajectories**. Prefix acceptance compounds. K8 is excellent evidence that the local dynamics are forgiving; K32 tells us whether they remain forgiving long enough to pay for a ~1-second-class verifier cycle.

The other big insight is that the current H1 result is now almost within the **absolute K32 ceiling**: perfect K32 acceptance gives 33 accepted tokens, so Exact-30 permits only 1.10 s total cycle time. H1 alone is ~1.113 s before student/tail overhead. That means the remaining system goal is no longer some arbitrary “0.75 s or bust,” but we still need a **real H2 improvement plus a cheap D2 tail**.

So the next two results I care about most are:

$$
\boxed{\textbf{D2: what does the real two-layer tail cost?}}
$$

and

$$
\boxed{\textbf{D1-B32/D3: what E[A] survives at K32 with real student errors?}}
$$

Current engineering completion:

$$
\boxed{\mathbf{90-91\%}}
$$

**Master plan updated.**
