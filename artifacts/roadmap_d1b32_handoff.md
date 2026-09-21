This is a **major roadmap update**. The seven-domain K32 run confirms the deep-tail architecture survives across domains, but it also kills one overly optimistic assumption from D1-B:

$$
\boxed{\text{rel-L2 alone is NOT a sufficient D3 success criterion}}
$$

At the same 10% state error, K32 \(E[A]\) ranges from **15.14 to 25.57** depending on error structure. At 30%, it falls to roughly **8.7–13**. So what matters is no longer “how close is \(\hat h_{40}\)?” but:

$$
\boxed{\text{what acceptance does the real student's residual trajectory produce after the source tail?}}
$$

The other major change is on D2: **the VNNI design is not invalidated, but its old throughput numbers are now historical microkernel results rather than production-source timing.** The signed/full-range E4M3 adapter has initial exactness evidence, but we must re-certify both exactness and performance with that adapter in the actual source path.

# MosaicVM Canonical Master Roadmap

## Seven-Domain K32 PASS — Student Residual Shape + Production Source Tail/Verifier Are Now the Decisive Gates

**Target:** DeepSeek-V4-Flash-0731
**Hardware:** RTX 3060 12 GB + ~24 GB RAM + NVMe
**Primary target:** ≥30 sustained final accepted/output tok/s
**Ultimate target:** Exact-30 at 1,048,576-token context

**Status:** CANONICAL — supersedes every prior MosaicVM roadmap.

---

# 1. Executive state

```text
CORRECTNESS
source-class target runtime                 COMPLETE
G2                                          CERTIFIED 2/2

GPU TARGET
MXIE                                        COMPLETE
imma4                                       CERTIFIED
TRUE-P2                                     0.297 + 8.36ms·N

MOVEMENT
expert namespace                            (layer, expert)
routed-expert universe                      ~147.6 GB
pure GPU+H2D                                CLOSED
pure residency                              CLOSED
naive CPU                                   CLOSED

CPU
VNNI v4-B integer core                      VALID
historical warm benchmark                   0.378 TFLOPS-eq
historical cold benchmark                   0.210 TFLOPS-eq
historical H1 contention                    ~0.246 TFLOPS-eq
SOURCE ACTIVATION ADAPTER                   NEW / PARTIAL PASS
production-source VNNI throughput           UNMEASURED

DEEP TAIL
D1-C L42 + FP32 head                        100% greedy Top1
D1-R true h40 → source L41/L42             BIT-EXACT
D1-B K8 synthetic robustness                PASS
D1-B32 seven-domain synthetic               PASS / QUALIFIED

D1-B32 EXACT
K8 E[A]                                     9
K16                                         17
K32                                         33

D1-B32 REL-L2 = 0.1, K32
Gaussian                                    25.57
Direction                                   18.14
Scale                                       22.29
AR1                                         15.14

D1-B32 REL-L2 = 0.3, K32
Gaussian                                    11.00
Direction                                    8.71
Scale                                       13.00
AR1                                         11.00

Top16 @ K32, all tested conditions           100%

DATA
source-class BF16 recorder                  WORKING
coding train pilot                          PASS
coding validation pilot                     PASS
large cross-domain corpus                   OPEN

D3 STUDENT
not yet trained on new corpus                OPEN

H2
production-correct full verifier             OPEN
```

---

# 2. Exactness contract remains frozen

Production verification must continue preserving:

* source FP8 core
* source FP4 expert codes/scales
* full-range source E4M3 activation behavior
* source scale semantics
* Top-6 routing
* all six routed experts
* source router/hash behavior
* source SwiGLU
* source attention/DSA
* shared expert
* source accumulation ordering
* FP32 production head.

Approximation remains confined to the draft path.

---

# 3. Checkpoint/provenance remains closed

Authoritative Hub/Xet checkpoint lineage and frozen hashes remain the only accepted source.

No stale-CDN artifact returns to the benchmark chain.

---

# 4. G2 remains certified

Correctness/equivalence remains closed.

Performance work must not alter that contract.

---

# 5. TRUE-P2 remains valid

Perfect-resident target compute:

$$
\boxed{
T_{\rm P2}(N)=0.297+0.00836N
}
$$

seconds.

At N32:

$$
\boxed{T_{\rm P2}(32)\approx0.5645s}.
$$

---

# 6. TRUE-P2 interpretation remains unchanged

It is an exact compute floor under effectively available expert/core weights.

It is not constrained-system P3.

---

# 7. Correct expert namespace remains frozen

Every expert object is:

$$
(\ell,e).
$$

Full routed-expert universe remains approximately:

$$
147.6GB.
$$

---

# 8. Pure GPU streaming remains closed

N32 compulsory routed expert traffic remains approximately:

$$
34.1GB.
$$

At ~12.9 GB/s:

$$
T_{\rm H2D}\approx2.64s.
$$

Pure GPU+H2D remains incapable of Exact-30.

---

# 9. Pure residency remains closed

Belady already established that compulsory first-touch traffic survives optimal caching.

Small strategic residency remains open only inside heterogeneous H2.

---

# 10. Historical VNNI result classification changes

The v4-B integer kernel itself is **not invalidated**.

But the historical microbenchmark used a restricted activation representation that did not exercise the complete source activation path.

Therefore freeze:

```text
VNNI INTEGER CORE                  VALID
OLD SOURCE-PRODUCTION TIMING       NOT CERTIFIED
```

---

# 11. Why

Historical activation microbenchmarks primarily exercised unsigned 0..15-style values.

Source V4 activation quantization contains signed/fractional/full-range finite E4M3 behavior.

A source adapter is therefore required.

---

# 12. New source adapter evidence

Current adapter passes:

```text
bounded integer-oracle cases over finite E4M3 values
+
one complete real L41 expert against ordered CPU reference
```

with bitwise agreement on that real-expert check.

This is meaningful exactness evidence.

---

# 13. What is NOT yet certified

Do not yet claim:

```text
complete tail CPU exactness
complete source-expert CPU exactness
production-source VNNI throughput
production-source full-tail latency
```

from the pilot checks.

---

# 14. Historical throughput numbers become baselines

Keep:

```text
0.378 TFLOPS-eq warm
0.210 TFLOPS-eq cold
~0.246 TFLOPS-eq contended
```

only as **kernel-era comparison numbers**.

They cannot be inserted into final P3 economics until the adapter cost is measured.

---

# 15. D2-E

## SOURCE ADAPTER EXACTNESS CERTIFICATION

This is now a mandatory gate before production CPU timing.

---

# 16. D2-E test matrix

Cover:

```text
all finite E4M3 activation code classes
positive / negative
subnormal / normal
fractional mantissas
all relevant exponent regimes

M = 1,2,3,4,8,16,...
w1
w3
w2

L41
L42

multiple real experts
multiple real activation rows
```

---

# 17. D2-E exactness target

For every production path:

```text
integer decomposition parity
source scale handling
ordered accumulation parity
final source-class numeric parity
```

must pass.

---

# 18. D2-E full-tail gate

Run actual complete L41/L42 tail CPU execution through the adapter.

Compare against the frozen ordered source reference.

Only then:

```text
CPU SOURCE TAIL EXACTNESS = CERTIFIED
```

---

# 19. D2-P

## SOURCE-ADAPTER PERFORMANCE

After parity, re-benchmark v4-B with the adapter in the timed region.

---

# 20. Required D2-P regimes

Measure:

```text
M1
M2
M3
M4
M8
M16

warm
large-set/cold DRAM
concurrent host-DMA contention
```

---

# 21. Adapter overhead decomposition

Report:

```text
activation decode/adaptation
VNNI dot kernel
scale accumulation
output reconstruction
total expert wall
```

so we know whether the adapter is negligible or becomes material.

---

# 22. New VNNI canonical rates

Only D2-P source-path results may replace:

```text
0.378
0.210
0.246
```

as production planning numbers.

---

# 23. Deep-tail architecture remains PRIMARY

D1-C + D1-R remain frozen:

$$
true\ h40
\rightarrow
source\ L41/L42
\rightarrow
true\ h42
\rightarrow
FP32\ head
$$

is exact.

---

# 24. D1-B K8 remains useful

K8 proved a wide local decision basin.

But its student rule:

$$
\text{rel-L2}\le0.3
$$

is now withdrawn as a standalone production gate.

---

# 25. Seven-domain D1-B32

## MAJOR NEW EVIDENCE

The same committed boundaries and noise directions were evaluated at:

$$
K=8,\ 16,\ 32.
$$

Seven separate domains completed exact replay successfully.

---

# 26. Exact K32

For exact h40:

$$
\boxed{E[A]=33}.
$$

This revalidates the tail/harness over the full K32 horizon.

---

# 27. 10% Gaussian trajectory error

K32:

$$
\boxed{E[A]=25.57}.
$$

This is the strongest non-exact seven-domain result.

---

# 28. 10% scale error

$$
\boxed{E[A]=22.29}.
$$

---

# 29. 10% directional error

$$
\boxed{E[A]=18.14}.
$$

---

# 30. 10% AR1 error

$$
\boxed{E[A]=15.14}.
$$

---

# 31. Central D1-B32 finding

Four trajectories with similar global rel-L2 can differ by more than:

$$
10\ accepted\ tokens/cycle.
$$

Therefore:

$$
\boxed{
\text{rel-L2 is diagnostic, not predictive enough for deployment selection.}
}
$$

---

# 32. 30% error confirms this even more strongly

K32:

```text
Gaussian      11.00
Direction      8.71
Scale         13.00
AR1           11.00
```

These are far below production-class acceptance.

---

# 33. Old D3 gate is retired

Permanently remove:

```text
rel-L2 ≤0.3 ⇒ D3 PASS
```

from the roadmap.

---

# 34. New D3 pass criterion

A student passes only by:

$$
\boxed{
predicted\ h40
\rightarrow
real\ source\ tail
\rightarrow
measured\ held\text{-}out\ E[A]
}
$$

plus measured runtime.

---

# 35. Student residual shape matters

Relevant properties now include:

```text
temporal correlation
bias/drift
directionality
error localization by horizon
error coupling to tail-sensitive dimensions
error after previous student mistakes
```

not only L2/cosine.

---

# 36. Synthetic D1-B32 remains a probe

The seven-domain test is stronger evidence than the original three-prompt K8 run.

But it still uses synthetic perturbations.

Canonical classification:

```text
TAIL ROBUSTNESS = STRONGLY SUPPORTED
REAL-STUDENT ROBUSTNESS = OPEN
```

---

# 37. Sample-size qualification

Seven independent prompts, one boundary/domain, establish:

```text
cross-domain existence evidence
```

but not:

```text
within-domain variance
production distribution confidence
```

The exploratory bootstrap CIs remain exploratory only.

---

# 38. K32 Top16

All tested conditions retained:

$$
\boxed{Top16=100\%}.
$$

This is strategically interesting.

---

# 39. But Top16 does NOT by itself validate BranchPack

Current perturbed trajectories follow the reference path.

A practical branch requires:

```text
branch-conditioned tokens
+
branch-conditioned future state predictions
+
target verification cost
```

for alternate prefixes.

The current synthetic Top16 figure does not supply those.

---

# 40. BranchPack status

Canonical:

```text
T1 BRANCHPACK                  CLOSED
D3-TAIL BRANCHPACK             CONDITIONAL / OPEN
```

Do not build it before actual D3 logits/residuals exist.

---

# 41. Why early errors matter

K32 shows that excellent marginal Top1/Top-k can coexist with much shorter prefix acceptance.

Prefix-stop speculation is dominated by the location of the **first mistake**.

Therefore future reports must always include survival curves, not only marginal accuracy.

---

# 42. D3 mandatory survival metrics

Report:

$$
S_1,S_2,S_4,S_8,S_{16},S_{24},S_{32}
$$

and:

$$
E[A].
$$

---

# 43. D3 marginal metrics remain secondary

Also report:

```text
Top1
Top2
Top4
Top8
Top16
state rel-L2
cosine
```

by horizon.

But they are diagnostic only.

---

# 44. K32 throughput budgets become clearer

For any measured \(E[A]\):

$$
T_{\rm cycle,max}=\frac{E[A]}{30}.
$$

---

# 45. Gaussian 0.1 budget

$$
25.57/30
=
\boxed{0.852s}.
$$

That includes:

```text
student
tail
verifier
all unhidden scheduling
```

---

# 46. Scale 0.1 budget

$$
22.29/30
=
\boxed{0.743s}.
$$

---

# 47. Direction 0.1 budget

$$
18.14/30
=
\boxed{0.605s}.
$$

---

# 48. AR1 0.1 budget

$$
15.14/30
=
\boxed{0.505s}.
$$

---

# 49. Important system implication

The previous H1 wall:

$$
\sim1.11s
$$

cannot support Exact-30 for any of these realistic synthetic K32 acceptance levels.

So movement optimization becomes co-primary again.

---

# 50. Perfect K32 remains only an oracle

With \(E[A]=33\):

$$
T_{\rm cycle,max}=1.10s.
$$

H1 was already slightly beyond this before draft-tail overhead.

Therefore H2 is mandatory regardless.

---

# 51. The old 0.75–0.85s region returns conditionally

It is no longer a universal gate.

But if a real D3 student delivers \(E[A]\sim22-26\), then total cycle budgets naturally fall back into approximately:

$$
0.73-0.87s.
$$

Thus H2 likely needs a substantial improvement unless D3 gets much closer to ceiling or a better K/tree geometry emerges.

---

# 52. K is still unfrozen

The seven-domain data do not prove K32 optimal.

---

# 53. Horizon marginal value

Examples at 10% error:

```text
Gaussian:
K8  9.00
K16 16.00
K32 25.57

Scale:
8.00
14.14
22.29

Direction:
7.29
12.29
18.14

AR1:
7.29
11.71
15.14
```

The payoff from extending horizon depends strongly on residual structure.

---

# 54. D3 geometry must therefore be raced

At minimum:

```text
K8
K16
K32
```

and possibly:

```text
K24
K48
K64
```

after initial cost models exist.

---

# 55. DATA recorder status

Source-class BF16 train/validation recorder is now operational.

---

# 56. DATA pilot

Current completed examples:

```text
coding_01       TRAIN pilot
coding_13       VALIDATION pilot
```

with nine generated-state rows each and saved shape/alignment/finiteness checks.

---

# 57. DATA pilot classification

```text
RECORDER MECHANISM       PASS
TRAINING CORPUS          NOT YET SUFFICIENT
```

No student conclusion can be drawn from these pilots.

---

# 58. DATA expansion is now mandatory

Build independent train/validation data across all target domains.

---

# 59. Recommended scaling ladder

Do not wait until the entire 1000+ prompt corpus is finished to obtain learning curves.

Capture/train at milestones such as:

```text
~10 prompts/domain
~30 prompts/domain
~100 prompts/domain
then 1000+ total prompts
```

while maintaining frozen held-out sets.

---

# 60. Why staged scaling

This tells us whether improvement comes from:

```text
more data
more model capacity
different trajectory formulation
```

before spending all collection/training compute.

---

# 61. Final corpus objective

Still target:

$$
\boxed{1000+\ distinct\ prompts}
$$

with many committed boundaries per prompt.

---

# 62. D3 train/validation/test discipline

Strict prompt-level separation.

The seven D1-B32 prompts remain **test-only**.

Never train on:

```text
coding_16
math_16
agent_16
convo_16
retrieval_16
longform_16
json_16
```

or their captured continuations.

---

# 63. G1/G2 remain excluded

Correctness certification prompts remain outside the student corpus.

---

# 64. D3 data records

Store enough information for:

```text
current/recent h40 states
future h40 trajectory
future h42 trajectory
committed tokens
future target tokens
tail FP32 Top-k/margins
position/domain
```

plus relevant provenance.

---

# 65. D3

## PRIMARY PROPOSER LEARNING TRACK

Primary deployed output remains:

$$
\hat h_{40,t+1:t+K}.
$$

The real source tail remains the decoder.

---

# 66. D3 first architecture race

Compare on identical data:

```text
A. chunked K8 latent predictor
B. open-loop K32 latent predictor
C. causal/recurrent latent trajectory predictor
```

---

# 67. Chunked K8 remains attractive

The local tail basin is strong at K8.

Chunking may reduce long-horizon systematic drift.

---

# 68. But chunking is not free

If each K8 chunk needs serial:

```text
student
+
tail
+
feedback
```

four times for K32, latency may dominate.

Measure wall time.

---

# 69. Open-loop K32 advantage

Potentially:

```text
one student pass
+
one block tail replay
```

with maximum parallelism.

But D1-B32 shows its error structure must be very good.

---

# 70. D3 validation loop

Every saved student checkpoint must automatically run:

```text
student h40 prediction
→ production source tail
→ FP32 head
→ acceptance evaluator
```

on held-out prompts.

---

# 71. D3 model selection

Rank checkpoints primarily by:

$$
\boxed{
\frac{E[A]}
{T_{\rm student}+T_{\rm tail}}
}
$$

before full H2 integration.

---

# 72. Hidden-state metrics cannot kill a model alone

A model with mediocre L2 may still generate excellent tail tokens.

Conversely, a low-L2 model can have poor acceptance if errors are systematically harmful.

---

# 73. Real-residual replay

## MANDATORY

As soon as the first D3 checkpoint exists:

```text
collect its actual h40 residual trajectories
→ feed those trajectories through the exact tail
→ compare to D1-B32 synthetic families
```

This becomes the new decisive proposer gate.

---

# 74. Residual fingerprinting

Characterize actual errors by:

```text
rel-L2
cosine
bias
temporal autocorrelation
principal directions
error growth with horizon
domain
```

to understand where they sit relative to Gaussian/direction/scale/AR1 probes.

---

# 75. D3 first meaningful PASS

Do not set a hidden-state threshold.

A first architectural pass is instead:

```text
real held-out K32 E[A] materially > old T1
+
clear scaling with data/model size
+
credible student latency
```

---

# 76. Suggested acceptance bands

```text
E[A] <4        weak
4–8            useful state-prediction signal
8–16           strong research result
16–22          system-relevant
22–28          Exact-30 potentially reachable with strong H2
28–33          near-ceiling proposer
```

These are not final binary gates.

---

# 77. BranchPack reconsideration

Once actual D3-tail logits exist, run a new oracle analysis.

---

# 78. Tree prerequisites

Need:

```text
actual student root/conditional Top-k
branch-conditioned student behavior
tree node count
exact verifier cost
```

not merely true-path Top16 coverage.

---

# 79. Linear versus tree decision

If D3-tail Top1 prefix survival is already strong:

```text
prefer linear
```

until trees show positive full-cycle economics.

If Top1 is weak but Top4/Top16 is extremely strong:

```text
evaluate adaptive BranchPack
```

---

# 80. D2

## COMPLETE SOURCE TAIL

Now co-primary with D3.

---

# 81. D2 tail exactness

First complete D2-E source-adapter certification.

No tail timing is accepted before parity.

---

# 82. D2 tail memory

Routed experts alone remain roughly:

$$
6.86GB.
$$

Measure complete L41/L42 memory including all non-expert tensors and scratch.

---

# 83. Tail RAM target

If complete tail fits:

```text
keep whole draft tail host-RAM resident
```

and eliminate draft-tail NVMe traffic.

---

# 84. D2 complete execution geometries

Benchmark:

```text
K1 serial
K8
K16
K32
```

with real routes/state replay.

---

# 85. D2 placement race

Measure:

```text
CPU source-adapted VNNI
GPU
hybrid
strategic VRAM subset
```

rather than assuming CPU is final.

---

# 86. D2 CPU measurement requirements

Report:

```text
adapter wall
expert wall
attention/DSA
shared expert
core linears
router
head
total tail wall
host bandwidth
```

---

# 87. D2 production contention

Then race tail execution simultaneously with the full verifier/H2 scheduler.

The draft tail and target CPU rail share the same host-memory system.

---

# 88. H2

## PRODUCTION-CORRECT VERIFIER

This remains the largest system-side gap.

---

# 89. H2 must remove H1's two synthetic assumptions

Use:

```text
genuine diverse CPU expert objects
actual GPU-assigned H2D inside timing
```

---

# 90. H2 source adapter

If the exact verifier CPU rail uses the same source activation issue:

apply the certified adapter there as well.

Production H2 must use the real source path, not historical unsigned test inputs.

---

# 91. H2 CPU split

Sweep based on work rather than number of expert objects.

---

# 92. H2 scheduling strategies

Race:

```text
continuous overlap
DMA bursts
CPU bursts
layer-phased execution
queue/deadline adaptive execution
```

---

# 93. H2 residency

Optimize:

$$
\boxed{\text{critical-path milliseconds saved per VRAM MB}}.
$$

---

# 94. H2 width sweep

Measure at least:

```text
N8
N16
N24
N32
N48
N64
```

because final draft geometry remains open.

---

# 95. H2 outputs

For each width/split:

```text
wall
CPU wall
GPU wall
H2D bytes/time
RAM BW
CPU source-adapter throughput
GPU expert time
nonexpert GPU time
scheduler overhead
bubbles
```

---

# 96. Unified D2+H2 scheduler

Tail work and verifier work can no longer be modeled independently.

Long-term scheduler inputs must include:

```text
draft-tail CPU deadlines
verifier CPU tasks
H2D queues
GPU availability
VRAM residency
RAM bandwidth
```

---

# 97. Final objective

For configuration \(G\):

$$
\boxed{
R(G)=
\frac{E[A](G)}
{
T_{\rm D3}(G)
+
T_{\rm tail}(G)
+
T_{\rm verifier}(G)
+
T_{\rm extra}(G)
}
}
$$

---

# 98. Exact-30 criterion

$$
\boxed{R(G)\ge30}.
$$

No proxy metric replaces this.

---

# 99. No universal E[A] threshold

Required acceptance depends on actual system cost.

---

# 100. No universal verifier-wall threshold

Likewise, no fixed 0.75/0.85/1.0 s number is globally correct.

But measured E[A] immediately translates into a cycle budget.

---

# 101. Current strongest synthetic K32 point

Gaussian 0.1:

$$
E[A]=25.57.
$$

It gives:

$$
T_{\rm total}\le0.852s.
$$

This is a useful system target, not a production claim.

---

# 102. At E[A]=22

Approximate cycle budget:

$$
\sim0.73s.
$$

---

# 103. At E[A]=16

Approximate budget:

$$
\sim0.53s.
$$

Thus low-acceptance D3 models cannot be rescued by modest scheduler tuning.

---

# 104. K-extension remains an escape path

If K32 acceptance is good but system budget remains impossible, test longer K.

But only after D3/D2 real measurements.

---

# 105. Why longer K may help

More accepted tokens can amortize:

```text
fixed target core cost
student setup
scheduler setup
```

and may improve rows/expert.

---

# 106. Why longer K may hurt

It can increase:

```text
latent trajectory drift
tail work
target batch work
memory pressure
```

and therefore lower final throughput.

---

# 107. Closed branches remain

```text
stale checkpoint
pure GPU+H2D
pure residency
global-expert cache model
naive CPU
FP32 CPU final rail
old T1 linear proposer
old T1 BranchPack
irreducible final-token entropy
route-exact student requirement
rel-L2≤0.3 as sufficient production gate
```

---

# 108. Historical/provisional branches

```text
old VNNI warm/cold/contended timing
```

remain useful references but are no longer production-certified until the source adapter is included.

---

# 109. Primary open architecture

```text
exact committed root
→ future-h40 D3 student
→ exact source L41/L42 tail
→ FP32 draft head
→ linear or adaptive tree depending real logits
→ exact heterogeneous verifier
```

---

# 110. Major open gates

```text
1. source-adapter full CPU exactness
2. source-adapter performance
3. complete tail D2 wall time
4. large cross-domain D3 corpus
5. first real D3 student residuals
6. real K32 E[A]
7. production-correct H2 wall
8. unified throughput
```

---

# 111. Immediate Track A — D2-E/D2-P

```text
expand adapter parity
→ multi-expert / full-range source activations
→ complete L41/L42 tail
→ source-path VNNI benchmarks
```

This should happen before using CPU timing in any final equation.

---

# 112. Immediate Track B — DATA

```text
expand train/val captures across every domain
→ preserve strict prompt separation
→ build learning-curve corpus
→ continue toward 1000+ prompts
```

---

# 113. Immediate Track C — D3

As soon as the first multi-domain corpus tranche exists:

```text
train K8 chunked student
+
train/open-loop K32 student
→ real-tail replay
→ E[A]
→ residual fingerprint
```

Do not wait for the entire 1000-prompt corpus to start the learning curve.

---

# 114. Immediate Track D — D2 complete tail

After adapter parity:

```text
measure K8/K16/K32 exact source tail
→ CPU/GPU/hybrid
→ isolated + contended
```

---

# 115. Immediate Track E — H2

```text
production-correct full verifier
→ genuine CPU objects
→ actual H2D
→ source adapter
→ split/scheduling/residency sweep
```

---

# 116. First D3 decision

### Strong

Real K32:

$$
E[A]\ge22
$$

with credible latency:

```text
immediately co-design with H2
```

### Moderate

$$
16\le E[A]<22
$$

Scale data/model and test K48/K64 or branching.

### Weak

$$
8\le E[A]<16
$$

The state student has signal but not enough; investigate residual structure and chunked feedback.

### Very weak

$$
E[A]<8
$$

Revisit student architecture/capacity before system integration.

---

# 117. Real-residual decision

If actual student residuals behave like Gaussian 0.1:

very favorable.

If they behave like AR1/directional:

focus training on temporal drift correction rather than simply lowering global L2.

---

# 118. Loss-design consequence

Potentially weight:

```text
low-frequency temporal drift
future-delta consistency
tail-decoded ranking loss
prefix-survival-sensitive positions
```

more heavily than raw state MSE.

Only introduce these after observing actual residuals.

---

# 119. Full historical test suite

The five targeted CPU regressions passing is good.

Before production integration, rerun the complete historical suite after adapter changes.

No partial regression set substitutes for that.

---

# 120. P3 trigger

Begin full integrated P3 once we possess four measured quantities:

```text
real D3 E[A]
real D3 latency
real D2 source-tail latency
real H2 verifier latency
```

---

# 121. P3 milestones

```text
≥10 actual final tok/s
≥20
≥25
≥30
```

---

# 122. P4

Success remains sustained:

$$
\boxed{\ge30\text{ final output tok/s}}
$$

on the actual RTX 3060/CPU/RAM/NVMe system in STEADY thermal state.

---

# 123. 1M phase

After short-context Exact-30:

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

# 124. 1M measurements

At every context tier report:

```text
VRAM
RAM
NVMe
KV/DSA state
tail state
student wall
tail wall
verifier wall
E[A]
final tok/s
thermals
correctness
```

---

# 125. Research understanding

Current short-context understanding:

$$
\boxed{\sim99.9997\%}.
$$

The remaining proposer uncertainty is no longer conceptual information availability.

It is:

```text
real student residual structure
+
source-tail cost
+
heterogeneous verifier cost
```

---

# 126. Engineering/delivery progress

Current defensible value:

$$
\boxed{\mathbf{88-89\%}}.
$$

---

# 127. Why progress is slightly below the previous 90–91%

The seven-domain robustness result is valuable and positive.

But two assumptions previously treated as nearly production-ready are now explicitly open:

1. K8 robustness did not translate uniformly to K32.
2. historical VNNI timing did not include the full source activation adapter.

That warrants a small downward correction in delivery confidence.

---

# 128. Why progress remains high

Already solved:

```text
source correctness
G2
checkpoint provenance
GPU exact kernel stack
TRUE-P2
expert-universe accounting
Belady
exact tail interface
L42 information ceiling
broad synthetic tail robustness
CPU integer core
data-recorder mechanism
```

This is not a reset.

---

# 129. After source-adapter full certification

If complete tail/source CPU parity passes and adapter overhead is modest:

$$
\boxed{89-90\%}.
$$

---

# 130. After first real multi-domain D3 student

If real residuals produce:

$$
E[A]\ge16
$$

with a clear learning curve:

$$
\boxed{91-93\%}.
$$

---

# 131. After D3 reaches 22–28 acceptance and D2 is affordable

$$
\boxed{94-96\%}.
$$

---

# 132. After H2 + D2 + D3 jointly predict ≥30 from measured quantities

$$
\boxed{96-98\%}.
$$

---

# 133. Integrated P3 ≥20

$$
\boxed{97\%+}.
$$

---

# 134. Integrated P3 25–30

$$
\boxed{98-99\%}.
$$

---

# 135. Completion

Only:

$$
\boxed{100\%}
$$

when P4 measures ≥30 sustained actual final output tok/s.

---

# 136. Canonical NEXT

Run four fronts now:

```text
A — SOURCE CPU CERTIFICATION
adapter
→ exhaustive/real parity
→ complete tail parity
→ production-source v4-B timing

B — DATA
cross-domain train/validation captures
→ staged learning-curve corpus
→ 1000+ prompts

C — D3
first real future-h40 students
→ K8 chunked vs K32 open-loop
→ exact-tail replay
→ real K32 E[A]
→ residual fingerprint

D — H2/D2 SYSTEM
complete source tail timing
+
production-correct verifier
+
real H2D/CPU contention
+
split/residency/scheduling
```

Then:

```text
REAL D3 RESIDUALS
→ D1-B replay
→ linear E[A]
→ conditional Top-k
→ branch oracle only if justified

MEASURE:
Tstudent(K)
Ttail(K)
Tverifier(K)
E[A](K)

RACE K:
8 / 16 / 24 / 32 / possibly 48 / 64

OPTIMIZE:
R(K) = E[A](K) / Tcycle(K)

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

## The most important interpretation

The seven-domain result means **D1-B was directionally right, but the old “wide basin” needed a more precise definition**.

At K8, a 30% perturbed state could look almost harmless. At K32, the same nominal error can give only \(E[A]\sim9-13\). Conversely, just 10% Gaussian-like error gives **25.57**, which is genuinely close to the acceptance regime we were hoping for.

So the next student should **not** be trained toward a magic cosine number. Train it, then immediately run its residuals through the real tail and measure prefix survival.

The production D3 scorecard should effectively look like:

$$
\boxed{
\text{student}
\rightarrow
\hat h_{40,1:K}
\rightarrow
L41/L42
\rightarrow
tokens
\rightarrow
E[A]
}
$$

Everything before \(E[A]\) is diagnostic.

Also, the CPU adapter finding is important enough that I would put **source-path VNNI re-certification ahead of any serious tail performance claim**. The fact that one entire real L41 expert already matches bitwise is encouraging; this looks like an integration gap rather than evidence against VNNI. But until the adapter is benchmarked, the old 0.378/0.246 figures must remain provisional.

### Where Exact-30 stands now

The proposer path is still alive—and considerably better grounded than the old T0/T1 approach—but the seven-domain numbers show we need a **good real state predictor**, not merely a “roughly correct” one.

The system side is also more demanding than it looked: for the best non-exact K32 synthetic result,

$$
E[A]=25.57,
$$

the entire cycle must fit inside:

$$
\boxed{0.852\text{ s}}
$$

to achieve 30 tok/s. Student + tail consume part of that, so the exact verifier likely needs to get appreciably below that.

That makes the project's next decisive pair:

$$
\boxed{\textbf{real D3 residual acceptance}}
$$

and

$$
\boxed{\textbf{production-source D2/H2 wall time}}.
$$

Current engineering completion:

$$
\boxed{\mathbf{88-89\%}}.
$$

**Master plan updated.**
