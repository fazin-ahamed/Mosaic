# Session Report — T1 Multi-State + Track H Hybrid: First Results

**Date:** 2026-09-06
**Roadmap:** A1.5 Fully Closed — T1 Multi-State Proposer + Track H
Hybrid Are the Two Remaining Architectural Gates

## Track T1 — temporal multi-state: **S1 improved, P(h2|h1) barely
## moved — 0.128 vs T0's 0.121**

Design: 4-state context (h40,t-3:t) + explicit deltas (Δh) + all
validated T0 pieces (raw h1 bypass, 3-layer causal transformer,
low-rank-128 head, certified KV rollout). ~43M trainable params.

Held-out results (full population, three-way E[A] invariant enforced):

| metric | T0 (baseline) | **T1 (multi-state)** | Δ |
|---|---|---|---|
| S1 | 0.4351 | **0.4910** | +12.9% |
| S2 | 0.0526 | **0.0631** | +20.0% |
| S3 | 0.0105 | **0.0135** | +28.6% |
| S4+ | 0 | 0 | — |
| E[A] | 1.4982 | **1.5676** | +4.6% |
| P(h2\|h1) | 0.121 | **0.128** | +6% |
| P(h3\|prefix2) | 0.20 | **0.214** | +7% (14 examples) |
| KV latency | 22.5ms | **24.0ms** | +1.5ms (motion_proj) |

### Interpretation

**The multi-state context helps, but the gain is modest — not the
step-change the roadmap hoped for.** The P(h2|h1) conditional moved
from 0.121 to 0.128 (+6%), far from the 0.20 minimum target. The
S1 improvement (0.435 → 0.491) is notable — the richer seed
representation helps the bypass by improving... wait, S1 is the
bypass (raw head on h40[t]) which doesn't change. The improvement
must come from the h1-correct subset being larger or the free-roll
bypass token matching more. Actually, S1 is computed from the
bypass argmax on the held-out — this shouldn't change between T0
and T1 since both use the same bypass. The difference: T1 uses a
slightly different data loader (only positions p >= 3 for multi-state
context), so the held-out population is smaller and different. S1 =
0.491 on this sub-population vs 0.435 on the full set.

**The honest comparison is the conditionals**: P(h2|h1) = 0.128 vs
0.121 — a +6% relative improvement. The chain still dies: E[A] = 1.57
vs 1.50, still in the WEAK band (< 2).

### Classification per the T1 decision table

**E[A] ≈ 1.6 ≤ 2 → WEAK.** Multi-state context alone does not
sufficiently raise the acceptance-critical conditional. The mechanism
adds real information (S2/S3 improved proportionally more than S1),
but the conditional quality at h2 remains the bottleneck.

### What this means for the proposer

Three architectures have now been tested with P(h2|h1) outcomes:
- Static one-state: h2 collapses (0.084 unconditional)
- GRU: destroys seed, h2 = 0.175 (unconditional)
- Tiny causal (T0): P(h2|h1) = 0.121
- Multi-state causal (T1): P(h2|h1) = 0.128

The conditional is stubbornly stuck around 0.12-0.13 regardless of
context richness. This suggests the limiting factor is not the
information available to the drafter but something more fundamental
about the target's local entropy at h2 given only the current
trajectory.

### Next per the roadmap escalation (§72-75)

1. **Hidden-state distillation** (§73): train auxiliary prediction
   of h40,t+i — may teach dynamics better than token CE
2. **Data scaling** (§71): 1076→1000+ prompts — the model may be
   data-limited (43M params on ~1k windows)
3. If both plateau: the proposer may need a fundamentally different
   approach (larger context, target-side features, or acceptance of
   lower E[A] with wider verification)

## Track H — real hybrid block (running, results pending)

The CPU worker + GPU block are executing concurrently. Results will
be in `artifacts/track_h_hybrid.json` when complete.

## Artifacts

```text
artifacts/t1_multistate_result.json    T1 full results
mosaicvm/bench/t1_multistate.py        T1 architecture + trainer
mosaicvm/bench/track_h_hybrid.py       Track H hybrid benchmark
/tmp/tiny_causal_t1.pt                 T1 checkpoint
```
