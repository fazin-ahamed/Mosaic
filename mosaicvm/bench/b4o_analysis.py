"""B4-O analysis: fixed-width shallow-tree oracle from replay records.

Consumes b4o_records{,2}_{pred}_{domain}.pt (ranks of the true token at
each ladder level, Top-16 ids/scores) and computes:

  1. conditional coverage ladders  c_j(W) = P(true in Top-W at level j
     | path correct through j-1)   for W in {1,2,4,8,16}
  2. E[A]_oracle(D, W) = sum_{d<=D} prod_{j<=d} c_j(W)
  3. validation: W=1 ladder over B3-S-matched anchors (1-8) vs the
     B3-S hold/rank64 D1 E[A] per domain
  4. cost model: full W-ary tree node counts N(D,W), T_proposal =
     N * t_draft, T_verify(N) from measured P2 points, R_oracle_est
  5. rank-64 branch-value gate at equal node budget
  6. expert-object estimate from the routing corpus frontier

Classification: BLACKWELL_SANDBOX_ONLY + DRAFTER_NUMERICS_DEVIATION_
RECORDED (fp16 dataset states; deep-anchor position effect documented).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

OUT = Path("mosaicvm/bench/rfix_out")
DOMAINS = ["coding", "agent_tool_use", "math_reasoning",
           "general_conversation", "long_context_retrieval",
           "structured_json", "long_generation"]
PREDS = ["hold", "rank64"]
WIDTHS = [1, 2, 4, 8, 16]
DEPTHS = [1, 2, 3, 5, 10, 16, 30]
B3S_MATCH_ANCHORS = set(range(1, 9))     # dataset indices 1..8

# B4O_VERIFY_BLOCK_MEASURED: measured incomplete-target verification
# block times (seconds) from artifacts/true_p2.json (block_s at K
# tokens, Blackwell sandbox). NOT TRUE_P2: exact DSA + 0xFF semantics
# remain open, so these are P2_COREBLOCK-class timings.
B4O_VERIFY_BLOCK_MEASURED = [(56, 1.17098), (63, 1.25036), (96, 1.48077)]
T_DRAFT_MS = 25.52                        # measured on sandbox
# NOTE: the roadmap's "26.3/29.1/31.0 oracle/s" are R = E[A]/t with the
# PERFECT-drafter branchpack E[A] (28.3/38.0/49.9) — not raw rates.
# routing corpus: per-layer unique-expert union p50 vs window N
ROUTE_UNION = [(64, 86), (80, 95), (112, 112), (160, 127)]
TOTAL_OBJECTS = 256 * 43                  # expert-layer objects


def load_ranks(pred, dom, phase):
    pref = "b4o_records" if phase == 1 else "b4o_records2"
    p = OUT / f"{pref}_{pred}_{dom}.pt"
    if not p.exists():
        return None, None
    rec = torch.load(p, map_location="cpu")
    return rec["ranks"], rec["anchors"]


def merge(pred, dom):
    """Merge phase-1 (deep) + phase-2 (early) records."""
    all_ranks, all_anchors = [], []
    for phase in (2, 1):
        r, a = load_ranks(pred, dom, phase)
        if r is None:
            continue
        all_ranks.extend(r)
        all_anchors.extend(a)
    return all_ranks, all_anchors


def coverage_ladder(ranks, width):
    """Conditional coverage c_j(W) over a set of ladders."""
    n = len(ranks)
    alive = [True] * n
    cs = []
    for j in range(len(ranks[0])):
        n_alive = sum(alive)
        if n_alive == 0:
            break
        hits = sum(1 for k in range(n)
                   if alive[k] and 0 <= ranks[k][j] < width)
        cs.append(hits / n_alive)
        alive = [alive[k] and 0 <= ranks[k][j] < width
                 for k in range(n)]
    return cs


def ea_from_ladder(cs, D):
    ea, surv = 0.0, 1.0
    for j in range(min(D, len(cs))):
        surv *= cs[j]
        ea += surv
    return ea


def per_ladder_depth(ranks, width, D_cap=30):
    """Per-ladder accepted depth at width W (empirical, capped)."""
    out = []
    for r in ranks:
        d = 0
        for j in range(min(D_cap, len(r))):
            if 0 <= r[j] < width:
                d += 1
            else:
                break
        out.append(d)
    return out


def t_verify_s(N):
    """T_verify(N) in seconds, linear fit of measured block times."""
    if N <= 0:
        return 0.0
    import numpy as np
    xs = np.array([x for x, _ in B4O_VERIFY_BLOCK_MEASURED], dtype=float)
    ys = np.array([t for _, t in B4O_VERIFY_BLOCK_MEASURED], dtype=float)
    b, a = np.polyfit(xs, ys, 1)
    return float(a + b * N)


def union_estimate(N):
    """Per-layer unique experts (p50) estimate for an N-node window."""
    if N <= 0:
        return 0
    # fit union ~ alpha * N^beta on the corpus points
    import numpy as np
    xs = np.log([x for x, _ in ROUTE_UNION])
    ys = np.log([y for _, y in ROUTE_UNION])
    beta, loga = np.polyfit(xs, ys, 1)
    alpha = math.exp(loga)
    return int(round(alpha * (N ** beta)))


def node_count(W, D):
    """Verification-batch nodes: level-1..D of the full W-ary tree."""
    if W == 1:
        return D
    return W * (W ** D - 1) // (W - 1)


def internal_count(W, D):
    """Drafter calls needed: root + level-1..D-1 nodes."""
    if W == 1:
        return D
    return (W ** D - 1) // (W - 1)


def main():
    print("=" * 78)
    print("B4-O FIXED-WIDTH SHALLOW-TREE ORACLE")
    print("=" * 78)

    # ---- load + merge ----
    data = {}
    for pred in PREDS:
        for dom in DOMAINS:
            ranks, anchors = merge(pred, dom)
            if ranks:
                data[(pred, dom)] = (ranks, anchors)
    print(f"\nloaded: {len(data)} (pred, domain) record sets")

    # ---- 1+2: coverage ladders + E[A]_oracle (all anchors) ----
    print("\n" + "-" * 78)
    print("CONDITIONAL COVERAGE LADDERS c_j(W) — all anchors (aggregate)")
    print("-" * 78)
    agg = {}
    for pred in PREDS:
        all_r = [r for dom in DOMAINS for r in data.get((pred, dom), ([], []))[0]]
        if not all_r:
            continue
        agg[pred] = {}
        for W in WIDTHS:
            cs = coverage_ladder(all_r, W)
            agg[pred][W] = cs

    for pred in PREDS:
        print(f"\n{pred}:")
        hdr = "level j:  " + "".join(f"{'W='+str(w):>8}" for w in WIDTHS)
        print(hdr)
        for j in range(8):
            row = f"  j={j:<5}"
            for W in WIDTHS:
                cs = agg[pred][W]
                row += f"{cs[j]:>8.3f}" if j < len(cs) else f"{'—':>8}"
            print(row)

    print("\n" + "-" * 78)
    print("E[A]_oracle(D, W) — aggregate, all anchors")
    print("-" * 78)
    print(f"{'pred':<8}{'D':>4}{'N':>6}" + "".join(f"{'W='+str(w):>8}" for w in WIDTHS))
    for pred in PREDS:
        for D in DEPTHS:
            row = f"{pred:<8}{D:>4}"
            best_n = min(node_count(w, D) for w in WIDTHS)
            row += f"{best_n:>6}"
            for W in WIDTHS:
                cs = agg[pred][W]
                row += f"{ea_from_ladder(cs, D):>8.3f}"
            print(row)

    # ---- 3: validation vs B3-S (matched anchors 1-8, phase-2 only) ----
    print("\n" + "-" * 78)
    print("VALIDATION: W=1 ladder, B3-S-matched anchors (1-8) vs B3-S D1 E[A]")
    print("-" * 78)
    B3S = {"coding": 2.75, "agent_tool_use": 2.12, "math_reasoning": 1.75,
           "general_conversation": 0.88, "long_context_retrieval": 1.12,
           "structured_json": 1.88, "long_generation": 1.25}
    for pred in PREDS:
        print(f"\n{pred}:")
        print(f"{'domain':<26}{'oracle W=1':>12}{'B3-S D1':>10}{'diff':>8}")
        tot_o, tot_b, n_d = 0.0, 0.0, 0
        for dom in DOMAINS:
            r2, a2 = load_ranks(pred, dom, 2)
            if not r2:
                continue
            sel = [r for r, a in zip(r2, a2) if a in B3S_MATCH_ANCHORS]
            if not sel:
                continue
            depths = per_ladder_depth(sel, 1)
            o = sum(depths) / len(depths)
            b = B3S[dom]
            tot_o += o
            tot_b += b
            n_d += 1
            print(f"{dom:<26}{o:>12.3f}{b:>10.2f}{o-b:>+8.3f}")
        if n_d:
            print(f"{'MEAN':<26}{tot_o/n_d:>12.3f}{tot_b/n_d:>10.2f}"
                  f"{(tot_o-tot_b)/n_d:>+8.3f}")

    # ---- 4: cost model + rates ----
    print("\n" + "-" * 78)
    print("COST MODEL (B4O_VERIFY_BLOCK_MEASURED, P2_COREBLOCK-class)")
    print("  R_verify = E[A]/T_verify (zero overhead)")
    print("  T_cycle_est = T_verify + T_prop (two T_prop variants)")
    print(f"  (t_draft={T_DRAFT_MS} ms/call; T_verify: linear fit of "
          f"measured block_s)")
    print("-" * 78)
    print(f"{'pred':<8}{'W':>3}{'D':>3}{'N':>5}{'calls':>6}{'E[A]':>7}"
          f"{'T_ver':>7}{'R_ver':>7}{'R_cycU':>7}{'R_cycB':>7}"
          f"{'E[A]req30':>9}{'u/ly':>6}")
    rows = []
    for pred in PREDS:
        for W in [1, 2, 4, 8, 16]:
            for D in [1, 2, 3, 5, 10, 30]:
                N = node_count(W, D)
                if N > 160 or N == 0:
                    continue
                calls = internal_count(W, D)
                cs = agg[pred][W]
                ea = ea_from_ladder(cs, D)
                tv = t_verify_s(N)
                r_ver = ea / tv if tv > 0 else 0
                tp_batched = D * T_DRAFT_MS / 1e3
                tp_unbatched = calls * T_DRAFT_MS / 1e3
                r_cyc_b = ea / (tp_batched + tv)
                r_cyc_u = ea / (tp_unbatched + tv)
                u = union_estimate(N)
                rows.append((r_ver, pred, W, D, N, calls, ea, tv,
                             r_ver, r_cyc_u, r_cyc_b, u))
    rows.sort(reverse=True)
    for _, pred, W, D, N, calls, ea, tv, r_ver, r_cu, r_cb, u in rows[:26]:
        req30 = 30.0 * tv
        print(f"{pred:<8}{W:>3}{D:>3}{N:>5}{calls:>6}{ea:>7.3f}"
              f"{tv:>7.3f}{r_ver:>7.2f}{r_cu:>7.2f}{r_cb:>7.2f}"
              f"{req30:>9.1f}{u:>6}")

    # ---- 5: rank-64 branch-value gate at equal N ----
    print("\n" + "-" * 78)
    print("RANK-64 BRANCH-VALUE GATE (§86): r64 vs hold at equal N")
    print("-" * 78)
    for W in [2, 4, 8, 16]:
        for D in [1, 2, 3]:
            N = node_count(W, D)
            if N > 160 or 'hold' not in agg or 'rank64' not in agg:
                continue
            eh = ea_from_ladder(agg['hold'][W], D)
            er = ea_from_ladder(agg['rank64'][W], D)
            tp = internal_count(W, D) * T_DRAFT_MS / 1e3
            tv = t_verify_s(N)
            rh = eh / (tp + tv)
            rr = er / (tp + tv)
            print(f"W={W:<3}D={D}  N={N:<4} hold E[A]={eh:.3f} "
                  f"r64 E[A]={er:.3f}  R: hold {rh:.1f}/s  r64 {rr:.1f}/s  "
                  f"{'R64' if rr > rh else 'HOLD'} wins")

    # ---- per-domain E[A]_oracle table (W=4, D=2) ----
    print("\n" + "-" * 78)
    print("PER-DOMAIN E[A]_oracle (W=4, D=2; N=20) — all anchors")
    print("-" * 78)
    for pred in PREDS:
        row = f"{pred:<10}"
        for dom in DOMAINS:
            ranks = data.get((pred, dom), ([], []))[0]
            if not ranks:
                row += f"{'—':>9}"
                continue
            cs = coverage_ladder(ranks, 4)
            row += f"{ea_from_ladder(cs, 2):>9.3f}"
        print(row)
    print(f"{'':<10}" + "".join(f"{d[:8]:>9}" for d in DOMAINS))

    # save artifact
    art = {
        "phase": "NanoSpark-B4O-fixed-width-oracle",
        "classification": "BLACKWELL_SANDBOX_ONLY + "
                          "DRAFTER_NUMERICS_DEVIATION_RECORDED",
        "ladders": {p: {str(w): agg[p][w] for w in WIDTHS}
                    for p in agg},
        "cost_model": {
            "classification": "B4O_VERIFY_BLOCK_MEASURED "
                             "(P2_COREBLOCK-class; NOT TRUE_P2 — "
                             "exact DSA + 0xFF still open)",
            "t_draft_ms": T_DRAFT_MS,
            "verify_block_points_s": B4O_VERIFY_BLOCK_MEASURED,
            "route_union_points": ROUTE_UNION,
            "rate_definitions": {
                "R_verify": "E[A]/T_verify (zero overhead)",
                "R_cycle_unbatched": "E[A]/(T_verify + calls*t_draft)",
                "R_cycle_batched": "E[A]/(T_verify + D*t_draft)",
            },
        },
    }
    with open(OUT / "b4o_oracle.json", "w") as f:
        json.dump(art, f, indent=1)
    print(f"\nwrote {OUT / 'b4o_oracle.json'}")


if __name__ == "__main__":
    main()
