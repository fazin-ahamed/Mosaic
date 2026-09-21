# Session Report: DepthScale-1001 — CAPABILITY GATE CROSSED

**Date:** 2026-09-20
**Status:** COMPLETE. E[A] >= 4 reached on SCALE-DEV3 (development).
**Headline:** continuation retrain on the 1001-prompt union lifts greedy
E[A] 2.200 -> 3.300 and, combined with frozen d3 branching, reaches
**E[A] = 4.014 (B4) / 4.057 (B8)** — first time the capability gate is met.

---

## 1. What was done

The depth cliff residual was localized (branching saturated; 19/27 depth-3
misses were UNRANKABLE state drift). DepthScale attacks that drift by
retraining ONLY the continuation model on more data. Root (HIST1001) and
branch policy frozen; data is the only lever.

- **Teacher augmentation:** os/chain h40 rollouts added to the 917 gold-only
  scale teachers (scale210/700/1000). Cheap A2 rollouts, no SourceTail —
  **2.6 min for 917 records** (not the multi-hour "capture-class" job assumed;
  `build_examples` consumes only h40 state variants + gold decisions).
- **ps1001 training:** BridgeNet H1024/D3 on the 1001-prompt union (96,096 rows
  = 32,032 gold + 64,064 os/chain), frozen ps recipe (800 steps, gold_frac 0.25,
  strong pos-weight, full-vocab CE+margin+KL). FP32 out-of-core mmap backend,
  cgroup 16/20G, 50-step checkpoints. ~65 s/step, ~14.2 h wall, no OOM/reboot.

## 2. Result (gold-free depth-2/3 beam probe, frozen HIST1001 + d3 branching)

| B | ps_old E[A] | ps1001 E[A] | dE[A] | ps1001 S3 | ps1001 S4 |
|---|---|---|---|---|---|
| 1 (greedy) | 2.200 | **3.300** | +1.100 | .300 | .243 |
| 2 | 2.343 | 3.743 | +1.400 | .457 | .300 |
| 4 | 2.457 | **4.014** | +1.557 | .514 | .357 |
| 8 | 2.771 | **4.057** | +1.286 | .543 | .357 |

- Greedy continuation alone (B=1) jumped +1.100 E[A] — the data-only effect.
- S3 (depth-3 survival) .071 -> .300 greedy, -> .543 at B8: the UNRANKABLE
  state-drift bucket collapsed, exactly the hypothesis.
- Branching now compounds: B4 already clears 4.0; B8 adds little over B4
  (branching near-saturated once continuation is strong).
- ps1001 B8 trim20 = 3.286, deepest 16, accepted 45/70. Root S1 unchanged .643
  (root frozen, as designed).

## 3. Gate status

```text
DepthScale mechanism (dEA>=0.25)     PASS (by a wide margin)
DepthScale strong (dEA>=0.50)        PASS
combined strong (E[A]>=3.5)          PASS (3.743 at B2)
CAPABILITY (E[A]>=4)                 PASS on SCALE-DEV3 (4.014 B4 / 4.057 B8)
```

## 4. CRITICAL caveat — this is development, not promotion

**SCALE-DEV3 is BURNED.** It was consumed for the SCALE1000 scaling decision
and is now reused here for mechanism/capability development. E[A]=4 on a burned
set is a strong development signal, NOT a capability promotion. Per roadmap:

- This does NOT open BLIND.
- Promotion requires E[A] >= 4 on a FRESH **CAP-DEV70** (70 prompts, 10/domain,
  fully disjoint from all train corpora, DEV21, SCALE-DEV70/2/3, G1/G2, BLIND).
- Only after CAP-DEV70 PASS: freeze root + ps1001 + branch policy + verifier +
  evaluator, then BLIND once.

## 5. Node-cost caveat unchanged

B4/B8 node counts (21/73) are candidate-tree statistics, NOT production verifier
wall. Before any P3 integration: measure the d3 tree's MoE expert-union via the
H2 profiler. B4 reaching 4.014 at 21 nodes is materially cheaper than B8's 73
nodes for +0.04 E[A] — the efficient production candidate is likely **d3 B4**.

## 6. Artifacts

- `ps1001.pt` — continuation checkpoint (H1024/D3, 1001-corpus)
- `depth_branch_probe_ps1001.json` — ps1001 probe (this result)
- `depth_branch_probe.json` — ps_old baseline
- `d3_depthscale_augment.py`, `d3_depthscale_train.py`, `d3_depthscale_finish.sh`
- teacher aug dirs: `d3_scale{,700,1000}_teacher_aug/`

## 7. Next

1. **Author + capture CAP-DEV70** (fresh, disjoint). This is now justified — a
   development candidate plausibly at capability exists. Capture cost: source
   recording for 70 prompts via SafeCapture v2 (val-only).
2. **Freeze the candidate** (HIST1001 + ps1001 + d3 branch policy) and evaluate
   on CAP-DEV70. Efficient variant d3 B4; quality variant d3 B8.
3. **Measure d3-tree expert-union cost** (H2 profiler) in parallel.
4. If CAP-DEV70 E[A] >= 4 with robust breadth -> freeze all -> BLIND once.

## 8. Classification

```text
DepthScale-1001                      PASS (data-only, +1.1 greedy)
combined HIST1001+ps1001+d3 branch   E[A] 4.014 (B4) / 4.057 (B8) on SCALE-DEV3
capability on BURNED dev set         MET (development signal, not promotion)
capability promotion                 PENDING fresh CAP-DEV70
BLIND                                SEALED
next lever                           CAP-DEV70 capture + frozen-candidate eval
engineering                          advanced; capability path now credible
```
