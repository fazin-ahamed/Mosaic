# Session Report: OFDM — FAIL (catastrophic forgetting) + semantics finding

**Date:** 2026-09-20
**Status:** ps_old OFDM CLOSED. Also: a correctness finding that questions
whether ExposureBank (as scoped) is distinct from the closed SCALE recipe.

---

## 1. OFDM result — regression, not repair

Census PASSED huge: ps_old (clean-unseen on 917 later prompts) failed
81.5%/position; 4531 states survived the ps1001 rank>=4 relevance filter;
all 7 domains, 898 prompts. But the 100-step repair REGRESSED ps1001:

| set | ps1001 B4 | ofdm B4 | delta |
|---|---|---|---|
| CAP-DEV70 | 3.757 | 3.300 | **-0.457** |
| SCALE-DEV3 | 4.014 | 3.057 | **-0.957** |

Per-domain (CAP-DEV70 B4): convo -1.10, json -0.90, math -0.60, retrieval
-0.60; weak domains unmoved (agent +0.10, coding -0.10, longform 0.00).
Strong domains damaged, weak untouched = classic off-policy interference.

**Cause:** the 4531 states are ps_old-*trajectory* states. ps_old (84-prompt
model) lives on a different, worse state manifold. ps1001 never reaches those
states at inference, so CE-to-gold on them dragged ps1001 toward ps_old's
behavior on states ps1001 was already handling. The rank>=4 filter tested
"ps1001 fails when force-fed s", not "ps1001 reaches s". Only the latter matters.

**Verdict: ps_old OFDM CLOSED.** "Unseen is not enough — states must also be
on the current policy's manifold."

## 2. Evaluator bug (must fix regardless)

`d3_depth_branch_probe.py` reports dEA vs the candidate's OWN B1, not vs the
frozen ps1001+B4 baseline. The finish script printed verdict=PASS while the
model had regressed. Every future repair gate MUST compute
dEA = candidate_B4 - 3.757 (frozen ps1001 CAP-DEV70 B4), with hash asserts on
root/continuation-baseline/branch/eval-set. Fix before any further repair run.

## 3. Semantics finding — does ExposureBank differ from SCALE1000?

Acceptance in the eval harness is `draft == toks[t+1]` where toks is the GOLD
greedy continuation, and E[A] = accepted-gold-prefix length + 1. The verifier
resets on first mismatch. Therefore:

- The ONLY states that affect E[A] are **gold-prefix states** (all prior
  drafts correct). ps1001's own *wrong-prefix* draft states are irrelevant to
  accepted length — the cycle ends at first reject.
- The drift that matters is purely **A2's cumulative state-prediction error
  along the gold token prefix**, which on unseen prompts is larger. The `chain`
  teacher variant already models exactly this (A2 recursive, gold-token driver).

Consequence: "capture fresh prompts, augment os/chain, train ps" is
**mechanically the SCALE1000 recipe** — which already trained ps1001 on
gold+os+chain of 1001 prompts and PLATEAUED at the capability gate. An
ExposureBank built that way is SCALE-more in disguise.

The only genuinely distinct lever ExposureBank could add is **first-reject
FOCUSING**: train preferentially on the gold-prefix chain states where ps1001
mispredicts on fresh prompts, vs SCALE1000's uniform gold:os:chain mixture.
That is a narrow, unproven marginal hypothesis over a closed recipe — and OFDM
already showed that naive "focus on failure states" regresses (off-manifold),
so even the focusing variant is not clearly safe.

## 4. Open question for the operator

ExposureBank costs ~3.5h source capture for 35 fresh prompts. Given the
semantics above, its hypothesis reduces to "first-reject-focused chain-state
training on fresh prompts beats uniform SCALE training." Options:

- **A. Proceed with ExposureBank** but explicitly as first-reject-FOCUSED
  chain-state repair (not uniform), 35-prompt yield-gated, compare vs frozen
  3.757. Accept it may replicate the SCALE plateau.
- **B. Skip to Hydra.** The residual is depth-drift; sequentially-dependent
  H2/H3/H4 heads change the continuation *capacity/conditioning*, a lever
  neither SCALE nor focused-chain training exercises. OFDM/BAPM/AAL together
  have arguably already shown the current ps *architecture+data recipe* is at
  its ceiling.
- **C. One chronological-holdout model** (train ps on first ~700, mine the 301
  later at near-ps quality) — cheaper on-manifold-ish test, no capture, but
  still a weaker-model trajectory (partial OFDM flaw).

## 5. State

```text
BAPM                    CLOSED (margin not the gap)
AAL-2 on train          VACUOUS (memorized)
ps_old OFDM             CLOSED (off-manifold forgetting)
ps1001                  FROZEN BEST, fresh B4 3.757
evaluator baseline bug  MUST FIX
BLIND                   SEALED
engineering             92-93%
```
