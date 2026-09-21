# Session Report — SCALE700: STRONG PASS on Fresh SCALE-DEV2
# (ΔH=+9); SCALE1000 Authorized; Capability Gate Still Open

**Date:** 2026-09-16
**Roadmap:** SCALE210 PASSED — Fresh SCALE700 Generalization
Phase Complete (§19–§23 of the SCALE700 setup)

The leftover ledger string
`STRONG_PASS_SCALE700_AUTHORIZED` is the **measured label**
from `d3_scale_ledger.py` (written when strong S30 fires).
SCALE700 already ran. Per setup §19–§23, ΔH≥6 strong →
**1000+**, not another SCALE700. This report authorizes
SCALE1000 and does not reuse that string as a next-step.

## Predeclared verdict (S27–§30)

Primary pairing, frozen before training: HIST1_210 →
HIST1_700 on SCALE-DEV2 (fresh 70-prompt authority, disjoint
from all 700 training prompts and from SCALE-DEV70 / DEV21):

$$
G=11,\quad L=2,\quad \Delta H=+9
$$

$$
S_1:\ 40/70\ \rightarrow\ 49/70\ (0.571\rightarrow 0.700)
$$

- **mechanism S28 PASS:** ΔH≥4, ΔE[A]=+0.186>0, S4=0.0429>0
- **robust S29 PASS:** Δtrim20=+0.286, ΔLOO=+0.188,
  domains_net_gain=5 (≥2)
- **strong S30 PASS:** ΔH≥6 and ΔE[A]/Δtrim20/ΔLOO all >0
- plateau S32 false, fail S33 false
- **Measured ledger verdict:** `STRONG_PASS_SCALE700_AUTHORIZED`
- **Consequence:** SCALE1000 authorized. Capability gate
  E[A]≥4 still OPEN → BLIND-ADJ remains sealed.

Secondary pairing WS1_210 → WS1_700 on the same set:
G=12, L=2, ΔH=+10, S1 39/70→49/70. Same four gates PASS.
Same leftover ledger string. Same next step.

## Full numbers (SCALE-DEV2, n=70, authority)

| model | splice S1 | mean E[A] | trim20 | LOO | Top4 | NLL | r@16 | unreach |
|---|---|---|---|---|---|---|---|---|
| HIST1_210 (frozen baseline) | 40/70 | 2.157 | 1.929 | 2.087 | .757 | 3.63 | .814 | 11 |
| **HIST1_700** | **49/70** | **2.343** | **2.214** | **2.275** | **.800** | **2.07** | **.900** | **5** |
| WS1_210 (frozen baseline) | 39/70 | 2.114 | 1.881 | 2.044 | .743 | 3.32 | .786 | 12 |
| WS1_700 | 49/70 | 2.343 | 2.214 | 2.275 | .800 | 2.31 | .886 | 7 |

Boot CI95 mean E[A]: HIST210 [1.886, 2.486] vs HIST700
[2.071, 2.643] — intervals overlap. ΔH and trim20/LOO still
pass the predeclared gates; do not read the CI overlap as a
gate fail. S4 stayed 0.0429 (deepest 6 on both). Depth did
not move; this is a breadth result.

### Per-domain standalone S1 (HIST1_210 → HIST1_700)

| domain | 210 | 700 | Δ |
|---|---|---|---|
| coding | .50 | .80 | **+.30** |
| json | .40 | .70 | **+.30** |
| math | .70 | .80 | +.10 |
| agent | .50 | .60 | +.10 |
| retrieval | .60 | .70 | +.10 |
| convo | .90 | .90 | 0 |
| longform | .40 | .40 | 0 |

Largest S1 lifts are **coding and json**, not agent.
Longform S1 is unchanged at 0.4 because G and L cancelled
inside the domain (longform_055 converted; longform_047
lost). Convo was already 0.9 and stayed there.

HIST700 G: agent_055, coding_053/055/056, json_048/052/054,
longform_055, math_052/056, retrieval_048.
HIST700 L: longform_047, math_053.

## Continuity overlays (not authority)

| set | model | splice S1 | mean E[A] | trim20 | deepest |
|---|---|---|---|---|---|
| SCALE-DEV70 (now a development-decisions set) | HIST1_700 | 48/70 | 2.214 | 2.143 | 5 |
| legacy DEV21 | HIST1_700 | 11/21 | 2.810 | 1.846 | 15 |
| legacy DEV21 | WS1_700 | 10/21 | 2.714 | 1.692 | 15 |

SCALE-DEV70 at 48/70 sits next to HIST1_210's historical
41/70 on that same set (session_d3_scale210.md). That is
continuity, not a paired ledger — SCALE-DEV70 informed
SCALE700 design and is no longer a fresh holdout.

Legacy DEV21 HIST1_700 = 11/21 (standalone S1 0.524). Prior
HIST-1 DEV comparator was 10/21. Depth preserved (deepest 15,
S4 0.143). After 20+ architecture decisions iterated against
DEV21, this set is not authority. Both numbers reported; no
DEV-based action.

## Census (SCALE-DEV2)

HIST1_210 → HIST1_700: recall@16 .814→.900; unreachable
11→5 (agent_052, longform_049/051/052, math_055).
16 rerankable misses remain (rank 2–64). Coverage moved with
quality. Five prompts still sit outside Top-512.

Teacher audit: 490/490 gold records, overlap_checks all
empty (`train_vs_prior_manifests`, `train_texts_vs_prior`,
`train_vs_scale_dev2`), `all_empty_required: true`.
Manifest sha256 `62ecaf2bc6b8c070…12be687` unchanged from
setup.

Training: 22,400 rows (orig84 2,688 + scale126 4,032 +
scale490 15,680). Frozen HIST-1 recipe (H1024/D3/aux256,
prev-aux, CE+margin+KL, AdamW 1e-3, cosine 600). Train
argmax reached 1.000 by step 250; step 500 L=0.201. The
gain is generalization on a fresh holdout, not train-set
memorization of SCALE-DEV2.

## What this does not authorize

- **E[A]≥4 still OPEN.** Best measured mean on the authority
  set is 2.343. Legacy DEV21 mean 2.810 is a 21-prompt
  continuity number with a wide CI [1.67, 4.38] — not a
  capability pass.
- **BLIND-ADJ remains sealed.** Setup: freeze + one-shot
  BLIND only after E[A]≥4 on a frozen architecture.
- **Not a production P3/P4 result.** No final-output tok/s
  measured. Engineering progress stays 92–93%.
- **Not a ranking / architecture change.** Inputs, capacity,
  and the HIST-1 loss are the SCALE210 recipe at 700
  prompts.
- **SCALE-DEV2 is now burned** as a development-decisions
  set, same way SCALE-DEV70 was after SCALE210. SCALE1000
  needs a new fresh 70-prompt holdout before any 1000-scale
  training.

## Next

1. **SCALE1000** (authorized by strong S30). New fresh
   holdout first (70 prompts, disjoint from 700 train +
   SCALE-DEV2 + SCALE-DEV70 + DEV21 + G1/G2). Then 300 new
   train prompts (7×~43, or a documented 7×100−700 split)
   under the same frozen recipe. Pair HIST1_700 vs HIST1_1000
   (and WS) on that new holdout. Same gates.
2. Capability gate E[A]≥4 remains the BLIND trigger, not
   another scale step.
3. H2 production verifier stays co-primary and independent.

## Artifacts

```text
artifacts/d3_scale700_train/ledger_hist210_vs_hist700.json  PRIMARY
artifacts/d3_scale700_train/ledger_ws210_vs_ws700.json      secondary
artifacts/d3_scale700_train/hist700.pt                      85e63a3cb763080c
artifacts/d3_scale700_train/ws700.pt                        747ae851aed994dd
artifacts/d3_scale700_train/{hist,ws}700_scaledev2.json
artifacts/d3_scale700_train/census_{hist,ws}700_scaledev2.json
artifacts/d3_scale700_train/hist700_scaledev70.json
artifacts/d3_scale700_train/{hist,ws}700_legacy_dev21.json
artifacts/d3_scale700_train/baseline_{hist,ws}210_scaledev2.json
artifacts/d3_scale700_teacher/audit.json                    ae58ef49609ce02f
artifacts/d3_scale700/prompts_scale700.json                 62ecaf2bc6b8c070
```
