"""DSpark analysis from full-stream draft records — POST-AUDIT semantics.

Operates on already-captured per-block records — no model required.

Record schema (one dict per draft block):
    target              list[int]   sampled-path target tokens (D0 geometry)
    target_greedy_path  list[int]   greedy-path target tokens (may be short:
                                    collected only while the greedy draft
                                    matched — SURVIVOR-CONDITIONED)
    sampled             list[int]   T=1.0 sampled draft
    greedy              list[int]   argmax draft
    topk_ids            list[list[int]]  greedy-chain candidates per depth
    topk_scores         list[list[float]]
    confidence          list[float]
    bonus               int
    rng_seed            int

Classifications (per roadmap §20-24):
  D0  VALID — sampled chain vs its own target trajectory
  D1  VALID — greedy chain vs greedy-path target (survivor-consistent:
      target_greedy_path extends exactly as far as the greedy draft
      matched, so first-mismatch is exact; deeper agreement is
      unreachable by construction and does not bias the prefix length)
  D2  depth-0 valid (path-independent); depths 1-4 are
      CONDITIONAL_ON_GREEDY_PREFIX_SURVIVAL (survivor-conditioned
      diagnostics, not unbiased tree coverage)
  D3  the score-mass path surrogate is retained ONLY as
      D3_SCORE_MASS_PATH_SURROGATE — an exploratory diagnostic, NOT a
      measured BranchPack tree (no branch-conditioned candidates; N is
      mass, not unique tree nodes). True trees: dspark_real_tree.py.
"""
from __future__ import annotations

BLOCK_SIZE = 5
TOPK_COVER = (1, 2, 4, 8, 16)
REAL_BUDGETS = (8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160)


def accepted_prefix(draft, target):
    """First-mismatch accepted depth. draft/target are equal-length lists."""
    k = 0
    n = min(len(draft), len(target))
    while k < n and draft[k] == target[k]:
        k += 1
    return k


def d0_sampled(records):
    """D0: source-default sampled T=1.0 chain (VALID)."""
    depths = [accepted_prefix(r["sampled"], r["target"]) for r in records]
    return _depth_stats(depths, label="D0_SOURCE_DEFAULT_SAMPLED_CHAIN")


def d1_greedy(records):
    """D1: greedy DSpark chain (VALID).

    Uses target_greedy_path when present. That path is collected by
    feeding the greedy draft's own tokens, so agreement is exact up to
    the first mismatch and the prefix length is unbiased.
    """
    depths = []
    for r in records:
        tgt = r.get("target_greedy_path") or r["target"]
        depths.append(accepted_prefix(r["greedy"], tgt))
    return _depth_stats(depths, label="D1_GREEDY_CHAIN")


def d2_coverage(records, ks=TOPK_COVER):
    """D2: P(target in greedy-chain top-k) at each depth.

    Path-aligned (target_greedy_path). Depth 0 is path-independent and
    unconditional. Depths >= 1 are scored only when the greedy chain
    matched through depth d-1 — SURVIVOR-CONDITIONED diagnostics.
    """
    hits = {k: [0] * BLOCK_SIZE for k in ks}
    n_at = [0] * BLOCK_SIZE
    for r in records:
        target = r.get("target_greedy_path") or r["target"]
        topk_ids = r["topk_ids"]
        greedy = r["greedy"]
        for d in range(BLOCK_SIZE):
            if d >= len(target) or d >= len(topk_ids):
                break
            if d > 0 and accepted_prefix(greedy, target) < d:
                break               # greedy chain died before depth d
            n_at[d] += 1
            ids = topk_ids[d]
            t = target[d]
            for k in ks:
                if t in ids[:k]:
                    hits[k][d] += 1
    out = {
        "label": "D2_TOPK_COVERAGE",
        "conditioning": ("depth0=UNCONDITIONAL; depths1+="
                         "CONDITIONAL_ON_GREEDY_PREFIX_SURVIVAL"),
        "n_at_depth": n_at,
        "P": {},
    }
    for k in ks:
        out["P"][str(k)] = [
            round(hits[k][d] / n_at[d], 4) if n_at[d] else None
            for d in range(BLOCK_SIZE)
        ]
    return out


def d3_score_mass_surrogate(records, budgets=None):
    if budgets is None:
        budgets = REAL_BUDGETS
    """D3_SCORE_MASS_PATH_SURROGATE — EXPLORATORY DIAGNOSTIC ONLY.

    NOT a measured BranchPack tree:
      * candidates are greedy-chain top-k, not branch-conditioned
      * N is redistributed particle mass, not a unique-node budget
    Retained per roadmap §24 for continuity with the 2026-08-29 run.
    True trees live in dspark_real_tree.py.
    """
    per_n = {}
    for n in budgets:
        depths = [_surrogate_accepted(r, n) for r in records]
        per_n[str(n)] = _depth_stats(
            depths, label=f"D3_SCORE_MASS_PATH_SURROGATE_N{n}")
    return {"label": "D3_SCORE_MASS_PATH_SURROGATE",
            "warning": ("not a true tree: greedy-chain candidates + mass "
                        "budget; see dspark_real_tree.py for D3_REAL"),
            "budgets": per_n}


def _surrogate_accepted(record, budget):
    """Score-mass allocation along the target path (legacy surrogate)."""
    target = record["target"]
    topk_ids = record["topk_ids"]
    topk_scores = record.get("topk_scores")
    if topk_scores is None:
        topk_scores = [[0.0] * len(ids) for ids in topk_ids]
    remaining = float(budget)
    accepted = 0
    for d in range(min(BLOCK_SIZE, len(target), len(topk_ids))):
        if remaining < 1:
            break
        ids = topk_ids[d]
        scores = topk_scores[d][:len(ids)]
        mx = max(scores) if scores else 0.0
        exps = [pow(2.718281828, s - mx) for s in scores]
        z = sum(exps) or 1.0
        weights = [e / z for e in exps]
        conf = record.get("confidence")
        if conf is not None and d < len(conf):
            c = max(0.0, min(1.0, float(conf[d])))
            onehot = [1.0 if i == 0 else 0.0 for i in range(len(weights))]
            weights = [c * o + (1 - c) * w
                       for o, w in zip(onehot, weights)]
            z = sum(weights) or 1.0
            weights = [w / z for w in weights]
        alloc = [int(remaining * w) for w in weights]
        drift = int(remaining) - sum(alloc)
        if drift != 0 and alloc:
            alloc[0] = max(0, alloc[0] + drift)
        hit = False
        t = target[d]
        for tok, a in zip(ids, alloc):
            if tok == t and a >= 1:
                hit = True
                break
        if not hit:
            break
        accepted += 1
        remaining = float(sum(a for tok, a in zip(ids, alloc) if tok == t))
    return accepted


def _depth_stats(depths, label):
    n = len(depths)
    if n == 0:
        return {"label": label, "n_blocks": 0, "E[A]": None}
    hist = {str(i): depths.count(i) for i in range(BLOCK_SIZE + 1)}
    return {
        "label": label,
        "n_blocks": n,
        "E[A]": round(sum(depths) / n, 3),
        "hist": hist,
        "P(depth>=d)": {
            str(d): round(sum(1 for x in depths if x >= d) / n, 4)
            for d in range(1, BLOCK_SIZE + 1)
        },
        "tokens_per_block_incl_bonus": round(sum(depths) / n + 1, 3),
    }


def analyze(records, budgets=None):
    """D0/D1/D2 + the reclassified D3 surrogate. True trees: real_tree()."""
    if budgets is None:
        budgets = (16, 32, 48, 64, 80, 96, 112, 124, 128, 144, 160)
    return {
        "n_records": len(records),
        "D0": d0_sampled(records),
        "D1": d1_greedy(records),
        "D2": d2_coverage(records),
        "D3_surrogate": d3_score_mass_surrogate(records, budgets),
    }
