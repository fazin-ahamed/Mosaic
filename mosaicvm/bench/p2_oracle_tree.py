"""Track P2: root Top-k sweep + conditional Top-k + oracle tree.

NO TRAINING. Uses the frozen T1 checkpoint and existing held-out data.

Measurements:
  1. Root (bypass) Top-k: 1,2,3,4,6,8,16,32
  2. Conditional h_k Top-k given true prefix (teacher-forced logits
     at every position — this is what a tree containing the correct
     path would see)
  3. Uniform-width beam survival S_k(w) for w=1,2,4,8,16
  4. Oracle tree E[A] under node budgets (best-first allocation)
  5. L36/L38/L40/L42 root quality comparison (from recorded states)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from tiny_causal import load_frozen, raw_head_logits, K, VOCAB, DEV
from t1_multistate import (TinyCausalMS, load_split_ms, HC, HIDDEN,
                           CTX)

PROMPTS = Path("artifacts/stage1_prompts.json")
CKPT_T1 = Path("/tmp/tiny_causal_t1.pt")
OUT = Path("artifacts/p2_oracle_tree.json")


def main():
    torch.manual_seed(0)
    spec = json.loads(PROMPTS.read_text())
    splits = spec["splits"]
    Xm_ho, Xc_ho, y_ho = load_split_ms(splits["test"])
    fr = load_frozen()
    Xc_ho, Xm_ho = Xc_ho.to(DEV), Xm_ho.to(DEV)
    y_ho = y_ho.to(DEV)
    N = Xc_ho.shape[0]
    print(f"held-out N={N}")

    model = TinyCausalMS(fr["embed.weight"].to(DEV)).to(DEV)
    model.load_state_dict(torch.load(CKPT_T1, map_location=DEV))
    model.eval()

    # ---------- 1. root Top-k (bypass) ----------
    with torch.no_grad():
        bl = raw_head_logits(Xc_ho, fr)          # [N, V]
        b_rank = []
        for i in range(N):
            true_t = y_ho[i, 0]
            logit = bl[i]
            rank = int((logit > logit[true_t]).sum().item())
            b_rank.append(rank)
    b_rank = torch.tensor(b_rank)
    root_topk = {}
    for k in (1, 2, 3, 4, 6, 8, 16, 32):
        root_topk[k] = round(
            (b_rank < k).float().mean().item(), 4)
    print("root Top-k:", root_topk)

    # ---------- 2. teacher-forced logits at every position ----------
    # For the tree containing the correct path, each node on the
    # correct path sees the TRUE prefix. Teacher-forced logits give
    # exactly this.
    all_ranks = torch.zeros(N, K, dtype=torch.long, device=DEV)
    with torch.no_grad():
        for i0 in range(0, N, 8):
            lg = model(Xc_ho[i0:i0 + 8], Xm_ho[i0:i0 + 8],
                       y_ho[i0:i0 + 8])
            for j in range(lg.shape[0]):
                for k in range(K):
                    true_t = y_ho[i0 + j, k]
                    r = int((lg[j, k] > lg[j, k, true_t]).sum().item())
                    all_ranks[i0 + j, k] = r

    # ---------- 3. conditional Top-k at each horizon ----------
    cond_topk = {}
    for h in (1, 2, 3, 4, 8, 16, 32):
        row = {}
        for k in (1, 2, 4, 8, 16):
            row[k] = round(
                (all_ranks[:, h - 1] < k).float().mean().item(), 4)
        cond_topk[f"h{h}"] = row
    print("conditional Top-k (unconditional on prefix):")
    for h, r in cond_topk.items():
        print(f"  {h}: {r}")

    # ---------- 4. beam survival S_k(w) ----------
    # Tree structure: BYPASS at position 0 (root), MODEL at 1+.
    # Combined rank matrix: bypass rank at col 0, model rank at 1+
    comb_ranks = all_ranks.clone()
    comb_ranks[:, 0] = b_rank.to(DEV)   # bypass ranks at h1
    beam = {}
    for w in (1, 2, 3, 4, 6, 8, 16, 32):
        surv = {}
        for k in (1, 2, 3, 4, 8, 16, 32):
            ok = (comb_ranks[:, :k] < w).all(dim=1)
            surv[k] = round(ok.float().mean().item(), 4)
        beam[f"w{w}"] = surv
    print("beam survival:")
    for w, s in beam.items():
        print(f"  {w}: {s}")

    # ---------- 5. oracle E[A] under node budgets ----------
    # A tree with uniform width w per position has total nodes
    # ~= sum_{k=0}^{D-1} w^k. For budget B, find best (w, D).
    # Also: variable-width oracle (greedy: at each position, use
    # the width that maximizes marginal accepted-depth gain).
    budgets = (16, 24, 32, 48, 64, 80, 112, 224, 512, 2048, 8192)
    oracle = {}
    for B in budgets:
        best_ea, best_cfg = 0, None
        for w in (1, 2, 3, 4, 6, 8, 16, 32):
            for D in range(1, K + 1):
                if w == 1:
                    nodes = D
                else:
                    nodes = (w * (w ** min(D, 6) - 1)) // (w - 1) \
                        + max(0, D - 6) * w
                if nodes > B:
                    continue
                ea = 1 + sum(
                    (comb_ranks[:, :k] < w).all(dim=1)
                    .float().mean().item()
                    for k in range(1, min(D, K)))
                if ea > best_ea:
                    best_ea = ea
                    best_cfg = (w, D, nodes)
        oracle[B] = {
            "best_E_A": round(best_ea, 4),
            "width": best_cfg[0] if best_cfg else None,
            "depth": best_cfg[1] if best_cfg else None,
            "nodes": best_cfg[2] if best_cfg else None,
        }
        print(f"budget {B:4d}: best E[A]={best_ea:.4f} "
              f"(w={best_cfg[0]}, D={best_cfg[1]}, "
              f"nodes={best_cfg[2]})" if best_cfg else
              f"budget {B:4d}: no feasible tree")

    # ---------- 6. variable-width oracle (greedy) ----------
    # At each position, choose the width that maximizes E[A] per node
    # spent. This is an upper bound on any tree from these logits.
    # Simple approach: for each position, find the minimum width that
    # includes the true token; the "cost" of position k is that width.
    # A tree with per-position width w_k has the true path iff
    # rank[k] < w_k at every k. The greedy oracle picks w_k = rank+1
    # for each position (minimum to include truth).
    min_w = (all_ranks + 1)  # [N, K] minimum width to capture truth
    var_oracle = {}
    for D in (4, 8, 16, 32):
        # E[A] if the tree uses width = max(min_w[:, :k]) at each
        # position k (variable width per example is not a single
        # tree, but gives the information-theoretic ceiling)
        ea = 1 + sum(
            ((all_ranks[:, :k] < min_w[:, :k].max(dim=1).values
              .unsqueeze(1).expand(-1, k)).all(dim=1))
            .float().mean().item()
            if False else
            # simpler: true path survives iff rank[k] < w_k where
            # w_k is the tree's width at position k. The MINIMAL
            # tree that captures example i needs width max(min_w[i,:k])
            # at each position. The "oracle tree" with budget B
            # allocates widths optimally across positions.
            # For now: report the distribution of min_w.
            0 for _ in range(1))
        # report: P(path survives to depth D with per-position
        # minimal width) = P(rank[k] < w_k for all k<=D) where
        # w_k is determined by the tree geometry
        # For the ceiling: if the tree can adapt width per position,
        # the binding constraint is the MAXIMUM rank along the path
        max_rank = all_ranks[:, :D].max(dim=1).values
        surv = {}
        for w in (1, 2, 4, 8, 16, 32):
            surv[f"maxrank<w"] = None
        # simpler: report P(max rank along path < w) for various w
        ceiling = {}
        for w in (1, 2, 4, 8, 16, 32):
            ea_w = 1 + sum(
                (comb_ranks[:, :k].max(dim=1).values < w)
                .float().mean().item()
                for k in range(1, D + 1))
            ceiling[f"maxrank<{w}"] = round(ea_w, 4)
        var_oracle[f"depth{D}"] = {
            "max_rank_p50": int(comb_ranks[:, :D].max(dim=1)
                                .values.float().median().item()),
            "max_rank_p90": int(comb_ranks[:, :D].max(dim=1)
                                .values.float().quantile(0.9).item()),
            "E_A_ceiling_by_maxrank": ceiling,
        }
    print("variable oracle (maxrank ceiling):")
    for d, v in var_oracle.items():
        print(f"  {d}: p50={v['max_rank_p50']} "
              f"p90={v['max_rank_p90']} "
              f"ceiling={v['E_A_ceiling_by_maxrank']}")

    # ---------- 7. L36/L38/L40/L42 root comparison ----------
    # From the recorded states at each exit
    root_by_exit = {}
    for exit_L in (36, 38, 40, 42):
        # load states at this exit
        xs, ys2 = [], []
        for cat in splits["test"]:
            f = DATA / f"{cat}.pt" if (DATA / f"{cat}.pt").exists() \
                else None
            if f is None:
                continue
            d = torch.load(f, weights_only=False)
            g = d["gen"]
            st = d["states"][exit_L].float().view(-1, 4, 4096)
            pos = min(len(g), st.shape[0])
            for p in range(CTX - 1, pos):
                if p + K - 1 < len(g):
                    xs.append(st[p])
                    ys2.append(g[p:p + K])
        if not xs:
            continue
        X_e = torch.stack(xs).to(DEV)
        y_e = torch.tensor(ys2, dtype=torch.long).to(DEV)
        with torch.no_grad():
            rl = raw_head_logits(X_e, fr)
            t1_acc = (rl.argmax(-1) == y_e[:, 0]).float().mean().item()
            t2 = rl.topk(2, dim=-1).indices
            t2_cov = (t2 == y_e[:, 0:1]).any(-1).float().mean().item()
            t4 = rl.topk(4, dim=-1).indices
            t4_cov = (t4 == y_e[:, 0:1]).any(-1).float().mean().item()
            t8 = rl.topk(8, dim=-1).indices
            t8_cov = (t8 == y_e[:, 0:1]).any(-1).float().mean().item()
        root_by_exit[f"L{exit_L}"] = {
            "top1": round(t1_acc, 4), "top2": round(t2_cov, 4),
            "top4": round(t4_cov, 4), "top8": round(t8_cov, 4)}
    print("root by exit:", root_by_exit)

    result = {
        "classification": "P2_ROOT_TOPK_AND_ORACLE_TREE",
        "population": f"held-out N={N} (T1 multi-state subset)",
        "root_topk_bypass": {str(k): v
                             for k, v in root_topk.items()},
        "conditional_topk_unconditional": cond_topk,
        "beam_survival": beam,
        "oracle_tree_uniform": {str(b): v
                                for b, v in oracle.items()},
        "variable_oracle_ceiling": var_oracle,
        "root_by_exit": root_by_exit,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT}")


DATA = Path("/tmp/stage1_data")

if __name__ == "__main__":
    main()
