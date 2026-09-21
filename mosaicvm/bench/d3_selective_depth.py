"""Selective-depth race + beam-loss attribution (roadmap Probe-2).

Selective race: branch at {d2}, {d3}, {d2,d3}; B in {2,4,8}; root greedy.
Gold-free tree; gold consulted only after freeze (same contract as
d3_depth_branch_probe). B=1 baseline reproduces greedy E[A]=2.200.

Beam-loss attribution (on the reference d2+d3, B=8 run): for every prompt
where the true depth-3 token is locally rank<8 in the teacher-forced
continuation (i.e. "should be branchable"), classify why B8 failed to
accept through depth 3:

  ROOT_MISS          root greedy != true y1 (chain never starts)
  PREFIX_LOST        true (y1,y2) prefix absent from beam entering d3
  TOKEN_NOT_EXPANDED true prefix present but y3 rank >= local_k (K3 too small)
  CHILD_PRUNED       y3 expanded but (y1,y2,y3) lost global beam prune
  UNRANKABLE         true y3 rank >= 16 (branching cannot fix; DepthScale)
  S3_HIT             accepted through depth 3 (not a failure)

The dominant bucket dictates ABP2: PREFIX_LOST -> wider d2; TOKEN_NOT_EXPANDED
-> wider local K3; CHILD_PRUNED -> better beam scoring; UNRANKABLE -> DepthScale.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_dd1_hist import DDHistNet
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import (load_head, load_embed,
                                      make_bridge_feat)
from mosaicvm.bench.d3_v1 import make_feat, HORIZON
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_ws import load_a2, load_bridge
from mosaicvm.bench.d1b_tail_robustness import prefix_metrics
from mosaicvm.bench.d3_stats import stats as canon_stats

HIST = "artifacts/d3_scale1000_train/hist1001.pt"
A2 = "artifacts/d3_v1_A2/student.pt"
BRIDGE = "artifacts/d3_bridge_r3b/bridge_ps/bridge.pt"
BEAMS = (2, 4, 8)
MODES = {"d2": (2,), "d3": (3,), "d2d3": (2, 3)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="artifacts/d3_scale1000_data")
    p.add_argument("--manifest",
                   default="artifacts/d3_scale1000/prompts_scale1000.json")
    p.add_argument("--split", default="val")
    p.add_argument("--output",
                   default="artifacts/d3_scale1000_train/selective_depth.json")
    a = p.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()

    ck = torch.load(HIST, map_location="cpu", weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"], ck["aux_dim"])
    hist.load_state_dict(ck["model"]); hist.eval()
    xm_mean, xm_std = ck["xm_mean"], ck["xm_std"]
    xa_mean, xa_std = ck["xa_mean"], ck["xa_std"]
    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(Path(A2))
    bridge, b_xm, b_xs = load_bridge(Path(BRIDGE))
    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda(); w_norm_g = w_norm.cuda()

    caps = load_captures(a.data, json.loads(Path(a.manifest).read_text()),
                         None, HORIZON, require_both=False)
    val = [c for c in caps if c["split"] == a.split]
    print(f"val: {len(val)} prompts ({time.time()-t0:.0f}s)", flush=True)

    @torch.inference_mode()
    def hist_logits(mfn, afn):
        z = hist(mfn.unsqueeze(0), afn.unsqueeze(0)).cuda()
        zn = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)
        return (w_head_g @ (zn[0] * w_norm_g))

    @torch.inference_mode()
    def bridge_logits(z):
        zg = z.cuda()
        zn = zg * torch.rsqrt(zg.square().mean() + 1e-6) * w_norm_g
        return (w_head_g @ zn)

    def a2_step(state, prev, driver, cur_tok, t):
        feat = make_feat(embed, state, state - prev, driver, cur_tok, t)
        fn = (feat - a2_xm) / a2_xs
        pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
        return state + pred

    def cont_logits(new_state, driver, cur_tok, t):
        bf = make_bridge_feat(embed, new_state, driver, cur_tok, t)
        bfn = (bf - b_xm) / b_xs
        with torch.inference_mode():
            z = bridge(bfn.unsqueeze(0))[0]
        return bridge_logits(z)

    def lsm(lg):
        return F.log_softmax(lg.float(), dim=-1).detach().cpu()

    def rank_of(lg, tok):
        tl = float(lg[tok])
        return int((lg > tl).sum())

    def root_prefix(c):
        S = c["S"]; toks = c["tokens"]; h = c["h40"].float()
        cur = h[S - 1].flatten(); aux = h[S - 2].flatten()
        mf = make_dd_feat(embed, cur, toks[S], toks[S - 1], S)
        mfn = (mf - xm_mean) / xm_std
        afn = (aux - xa_mean) / xa_std
        lg0 = hist_logits(mfn, afn)
        root = int(lg0.argmax())
        rlp = float(lsm(lg0)[root])
        state0 = h[S - 1].flatten(); prev0 = h[S - 2].flatten()
        new_state0 = a2_step(state0, prev0, toks[S], toks[S - 1], S)
        return ((root,), rlp, new_state0, state0, toks[S], root)

    def run_beam(c, B, branch_depths):
        S = c["S"]; K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]
        beam = [root_prefix(c)]
        expansions = 1
        for depth in range(2, min(4, K + 1)):
            local_k = B if depth in branch_depths else 1
            cand = []
            for pref in beam:
                new_state = a2_step(pref[2], pref[3], pref[5], pref[4],
                                    S + depth - 1)
                lg = cont_logits(new_state, pref[5], pref[4], S + depth - 1)
                ls = lsm(lg)
                kk = min(local_k, ls.numel())
                vals, idxs = ls.topk(kk)
                for v, tok in zip(vals.tolist(), idxs.tolist()):
                    tok = int(tok)
                    cand.append((pref[0] + (tok,), pref[1] + float(v),
                                 new_state, pref[2], pref[5], tok))
                    expansions += 1
            cand.sort(key=lambda x: x[1], reverse=True)
            beam = cand[:B]
        # gold-free freeze; now score longest gold prefix, greedy tail
        def gpl(drafts):
            n = 0
            for j, tk in enumerate(drafts):
                if tk != toks[S + j + 1]:
                    break
                n += 1
            return n
        best = max(beam, key=lambda p: (gpl(p[0]), p[1]))
        drafts = list(best[0])
        state, prev, cur_tok, driver = best[2], best[3], best[4], best[5]
        for j in range(len(drafts), K):
            t = S + j
            ns = a2_step(state, prev, driver, cur_tok, t)
            lg = cont_logits(ns, driver, cur_tok, t)
            tok = int(lg.argmax())
            drafts.append(tok)
            cur_tok, driver = driver, tok
            prev, state = state, ns
        hits = [drafts[j] == toks[S + j + 1] for j in range(K)]
        m = prefix_metrics([int(x) for x in hits], K)
        return m, beam, expansions

    def teacher_forced_d3(c):
        """Follow TRUE tokens y1,y2; return (r3 rank of y3, reached)."""
        S = c["S"]; K = min(HORIZON, c["h40"].shape[0] - S)
        if K < 3:
            return None, False
        toks = c["tokens"]; h = c["h40"].float()
        state = h[S - 1].flatten(); prev = h[S - 2].flatten()
        cur_tok = toks[S - 1]; driver = toks[S]
        r3 = None
        for j in range(3):
            t = S + j
            ns = a2_step(state, prev, driver, cur_tok, t)
            if j == 2:
                lg = cont_logits(ns, driver, cur_tok, t)
                r3 = rank_of(lg, toks[t + 1])
            # teacher force: next driver is the TRUE token
            cur_tok = driver; driver = toks[t + 1]
            prev = state; state = ns
        return r3, True

    # ---- selective-depth race ----
    results = {"classification": "D3_SELECTIVE_DEPTH",
               "n_prompts": len(val), "by_mode": {}}
    # baseline B1 (no branching, identical across modes)
    base_rows = []
    for c in val:
        m, _, _ = run_beam(c, 1, set())
        base_rows.append({"category": c["category"],
                          "cl_loop": {"E_A": m["E_A"], "Sk": m["Sk"]}})
    ea1 = canon_stats(base_rows, "cl_loop", k=32)["mean"]
    results["baseline_B1_EA"] = ea1
    print(f"B1 baseline E[A]={ea1} ({time.time()-t0:.0f}s)", flush=True)
    print(f"{'mode':>6} {'B':>2} {'E[A]':>6} {'dEA':>7} {'S3':>6} "
          f"{'S4':>6} {'nodes':>6}")

    for mode, bd in MODES.items():
        results["by_mode"][mode] = {}
        for B in BEAMS:
            rows = []; nodes = 0
            for c in val:
                m, _, ex = run_beam(c, B, set(bd))
                rows.append({"category": c["category"],
                             "cl_loop": {"E_A": m["E_A"], "Sk": m["Sk"]}})
                nodes += ex
            cs = canon_stats(rows, "cl_loop", k=32)
            mn = nodes / len(val)
            results["by_mode"][mode][str(B)] = {
                "mean_EA": cs["mean"], "delta_EA": round(cs["mean"] - ea1, 4),
                "S2": cs["S2"], "S3": cs["S3"], "S4": cs["S4"],
                "trim20": cs["trim20"], "mean_nodes": round(mn, 1),
                "GainPerNode_incremental": round(
                    (cs["mean"] - ea1) / max(mn - 3.0, 1e-9), 4)}
            print(f"{mode:>6} {B:>2} {cs['mean']:>6.3f} "
                  f"{cs['mean']-ea1:>+7.3f} {cs['S3']:>6.3f} "
                  f"{cs['S4']:>6.3f} {mn:>6.1f}  ({time.time()-t0:.0f}s)",
                  flush=True)

    # ---- beam-loss attribution on reference d2d3 @ B=8 ----
    print("\n=== beam-loss attribution (d2+d3, B=8) ===", flush=True)
    B = 8; bd = {2, 3}
    buckets = Counter()
    branchable = 0
    for c in val:
        S = c["S"]; K = min(HORIZON, c["h40"].shape[0] - S)
        if K < 3:
            buckets["SHORT"] += 1
            continue
        toks = c["tokens"]
        y1, y2, y3 = toks[S + 1], toks[S + 2], toks[S + 3]
        m, beam, _ = run_beam(c, B, bd)
        root_draft = beam[0][0][0] if False else None
        # recompute root draft
        rp = root_prefix(c); root_draft = rp[5]
        r3, reached = teacher_forced_d3(c)
        # entering-d3 beam == beam after d2 prune; final beam has len-3 drafts
        prefixes3 = [p[0] for p in beam]
        true_prefix_present = any(pp[:2] == (y1, y2) for pp in prefixes3)
        child_present = any(pp[:3] == (y1, y2, y3) for pp in prefixes3)
        s3_hit = (m["Sk"][2] == 1)

        if s3_hit:
            buckets["S3_HIT"] += 1
            continue
        # failures below
        if r3 is not None and r3 < 8:
            branchable += 1
        if root_draft != y1:
            buckets["ROOT_MISS"] += 1
        elif r3 is not None and r3 >= 16:
            buckets["UNRANKABLE"] += 1
        elif not true_prefix_present:
            buckets["PREFIX_LOST"] += 1
        elif r3 is not None and r3 >= B:  # local_k = B = 8
            buckets["TOKEN_NOT_EXPANDED"] += 1
        else:
            buckets["CHILD_PRUNED"] += 1

    results["attribution_d2d3_B8"] = dict(buckets)
    results["attribution_branchable_d3_misses"] = branchable
    print(json.dumps(dict(buckets), indent=1))
    print(f"branchable (r3<8) d3 misses: {branchable}")

    # ABP2 pointer
    fail = {k: v for k, v in buckets.items()
            if k not in ("S3_HIT", "SHORT")}
    if fail:
        dom = max(fail, key=fail.get)
        rx = {"PREFIX_LOST": "widen d2 beam",
              "TOKEN_NOT_EXPANDED": "widen local K3 (decouple from B)",
              "CHILD_PRUNED": "improve beam scoring (calibration)",
              "UNRANKABLE": "DepthScale (state drift; branching cannot fix)",
              "ROOT_MISS": "root branching / better root"}
        results["dominant_failure"] = dom
        results["abp2_pointer"] = rx.get(dom, "?")
        print(f"DOMINANT: {dom} -> {rx.get(dom)}")

    Path(a.output).write_text(json.dumps(results, indent=1))
    print(f"wrote {a.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
