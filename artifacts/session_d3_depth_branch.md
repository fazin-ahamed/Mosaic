# Session Report: Depth-Branching Investigation (post-SCALE1000)

**Date:** 2026-09-19
**Status:** COMPLETE. Depth branching PASSED mechanism, then SATURATED.
**Headline:** E[A] 2.200 -> 2.771 with zero training/capture; remaining gap to
4.0 is now dominated by state-drift (DepthScale) + root miss, not branching.

---

## 1. Rank-at-cliff (diagnosis)

HIST1001 + A2 + ps splice on frozen SCALE-DEV3, true-token rank per rollout
position, chain-alive regime:

| depth | greedy | rank<4 | rank<8 | rank<16 | median |
|---|---|---|---|---|---|
| 1 root | .643 | .886 | .900 | .943 | 0 |
| 2 | .600 | .756 | .800 | .867 | 0 |
| 3 (cliff) | .185 | .444 | .630 | .667 | 5 |

Cliff at depth 3. Locally rank<8 in 63% of alive chains -> looked branchable.
Artifact: `rank_at_cliff.json`.

## 2. Depth-branch probe (gold-free, B=1 reproduces greedy 2.200)

Branch d2+d3, global beam:

| B | E[A] | dEA | S3 | S4 | nodes |
|---|---|---|---|---|---|
| 1 | 2.200 | - | .071 | .071 | 3 |
| 2 | 2.343 | +.143 | .129 | .100 | 7 |
| 4 | 2.457 | +.257 | .171 | .129 | 21 |
| 8 | 2.771 | +.571 | .257 | .186 | 73 |

Gate dEA_B8>=0.50 -> **PASS** (not STRONG >=1.0, not capability 4.0).
Recovered 0.571/1.8 = 31.7% of the original gap. Artifact: `depth_branch_probe.json`.

## 3. Selective-depth race — d3 is the efficient lever

| mode | B8 E[A] | dEA | nodes |
|---|---|---|---|
| d2-only | 2.357 | +.157 | 17 |
| **d3-only** | **2.671** | **+.471** | **10** |
| d2+d3 | 2.771 | +.571 | 73 |

**d3-only captures 83% of the gain at 14% of the nodes.** d2 branching is nearly
worthless (+0.10 for 7x nodes). ABP2 branches at d3, not d2.

## 4. Beam-loss attribution (d2+d3 B8, 70 prompts)

```
ROOT_MISS          25   (root greedy wrong = S1 miss .357; separate axis)
S3_HIT             18   (reached depth 3)
UNRANKABLE         19   (true y3 rank>=16: state drift)
PREFIX_LOST         5   (d2 pruning)
TOKEN_NOT_EXPANDED  2   (local K3 too small)
CHILD_PRUNED        1   (beam scoring)
```

Of 45 root-correct prompts, 27 miss depth 3. Of those 27:
**19 UNRANKABLE (70%) = state drift, branching CANNOT fix**; only 8 branch-fixable.
The auto-pointer said "ROOT_MISS -> better root" but that conflates the S1 axis
with the depth axis. Correct reading: depth residual is state drift -> DepthScale.
Artifact: `selective_depth.json`.

## 5. ABP2-minimal — confidence-gated d3 (margin threshold)

Root/d2 greedy, d3 width=8 iff top1-top2 logp margin < THR:

| THR | E[A] | S3 | nodes | branch frac |
|---|---|---|---|---|
| 0 (never) | 2.200 | .071 | 3.0 | .000 |
| 0.5 | 2.357 | .114 | 4.5 | .214 |
| 1.0 | 2.457 | .171 | 6.0 | .429 |
| 2.0 | 2.600 | .200 | 7.5 | .643 |
| 4.0 | 2.657 | .229 | 9.4 | .914 |
| always | 2.671 | .243 | 10.0 | 1.000 |

Clean quality/cost frontier. THR~2-4 trims ~25% nodes for -0.01 to -0.07 E[A].
But absolute node count is already tiny (10), so gating is a minor refinement.
Artifact: `abp2_d3.json`.

## 6. Verdict on branching

```text
depth branching                  PASS then SATURATED
best gold-free E[A]              2.771 (d2+d3 B8) / 2.671 (d3-only B8)
efficient policy                 d3-only, width 8, optional margin gate
gap to E[A]=4 recovered          ~32% (0.571 of 1.8)
remaining depth residual         70% UNRANKABLE (state drift)
branch-fixable residual          ~8/27 cases (small)
```

Branching, with no training and no capture, closed a third of the gap. It is now
saturated: further tree engineering buys little. The remaining path to E[A]=4:

1. **DepthScale-1001** (PRIMARY next): attacks the 19-case state-drift bucket.
   Cost: os/chain predicted-state teacher regen on 1001 prompts (multi-hour,
   capture-class) — now justified by the attribution.
2. **Root work** (25 ROOT_MISS caps everything): deferred root-representation axis.
3. **ABP2 production cost**: measure MoE expert-union of the d3 tree before
   committing (node count != verifier cost).

## 7. Node cost caveat

All node counts are candidate-tree statistics, NOT production verifier cost.
Before ABP2 promotion: measure |union of (layer,expert)| touched by the d3 tree
via the H2 profiler. A 10-node d3 tree with high expert overlap may be nearly
free; with disjoint experts it may not.
