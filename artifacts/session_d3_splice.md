# Session Report — DD1+ps Splice: MECHANISM GATE PASS

**Date:** 2026-09-12
**Roadmap:** DD-1 Reaches A2 Breadth — DD1+ps Splice Is NEXT
(§17–§28)

## Splice result (21 DEV21, zero retraining, canonical stats)

| metric | ps alone | DD1+ps splice | Δ |
|---|---|---|---|
| mean E[A] | 2.667 | **2.762** | +0.095 |
| S1 (count) | 8/21 | **9/21** | +1 |
| S4 | 0.143 | 0.143 | 0 |
| deepest | 15 | **15** | retained |
| trim20 | 1.539 | **1.692** | +0.153 |
| LOO-min | 2.000 | **2.100** | +0.100 |
| accepted | 8/21 | **9/21** | +1 |

Bootstrap CI: [1.62, 4.29]. Conditional survival: C2 =
P(S2|S1) = **0.889**; C3 = 0.500; C4 = 0.750.

DD-1 standalone S1 inside splice harness: **9/21** (exact
reproduction verified).

## Mechanism gate: **PASS**

S1 count ≥ 9: **9/21** ✓ (equals the A2 baseline)
S4 > 0: **0.143** ✓
Mean competitive: 2.762 > 2.667 ✓

**This is the first proposer in the project combining A2-level
first-position breadth (S1 = 9/21) with Bridge-level depth
(deepest = 15, S4 = 3/21).** Every prior architecture traded
one for the other.

C2 = 0.889: when DD-1's first token is accepted, the ps
continuation survives to position 2 89% of the time — the
bridge's continuation is essentially unaffected by receiving
DD-1's token instead of its own.

## Wall measurements (§21)

| component | ms/position |
|---|---|
| DD-1 first decision | **9.55** |
| ps continuation (A2+bridge+head) | **11.94** |
| first-position total | 21.49 |

Subsequent positions cost only the ps rate (11.94 ms). The DD-1
specialist adds ~9.55 ms only at position 0.

## Per-domain mean E[A]

| domain | mean |
|---|---|
| math | 7.667 |
| convo | 4.000 |
| json | 2.667 |
| retrieval | 1.667 |
| longform | 1.333 |
| agent | 1.000 |
| coding | 1.000 |

## Architecture (frozen DD1+ps baseline)

```text
exact h40[S-1] + root token → DD-1 (H1024) → token_1
A2(h40[S-1], root) → h40[S]                (recurrence)
if token_1 correct:
  A2(h40[S], token_1) → h40[S+1] → ps → token_2  (continuation)
```

## What this does NOT mean

- E[A] = 2.762 does not cross the ≥ 4 capability gate
- trim20 = 1.692 is still below A2's 1.762
- n = 21; needs BLIND-ADJ eventually
- DD-1 train S1 = 1.0 (memorization); val S1 = 9/21 is the
  measurement

## NEXT: DD-1 H512 capacity race (roadmap §31)

Train S1 = 1.0 strongly suggests H1024 overfits. Race H512
(same input, objective, teacher, steps, evaluator) to reduce
the train/val gap and potentially improve generalization.

## Artifacts

```text
artifacts/d3_splice_eval.json         full splice + canonical stats
mosaicvm/bench/d3_splice_eval.py      splice evaluator
artifacts/d3_dd1/dd1_h40/dd1.pt       DD-1 (h40, spliced)
artifacts/d3_bridge_r3b/bridge_ps/bridge.pt  ps (frozen)
```
