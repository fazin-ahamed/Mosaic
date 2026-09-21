"""Hydra splice probe: HIST1001 root + Hydra d2/d3/d4 + ps1001 tail, d3-B4.

Root (j=0): ps1001 greedy first decision (frozen).
d2/d3/d4 (j=1,2,3): Hydra direct-token heads, conditioned on committed root
  state + preceding DRAFT token embeddings. d3 branches global top-4.
d5+ (j>=4): fall back to ps1001 continuation on the A2 state machine
  (driver = the accepted/drafted token).

Gold-free tree; gold scored only after freeze. Reports B1/2/4/8 E[A] and the
frozen-baseline capability delta (vs 3.757 CAP-DEV70). A2 state advances
underneath throughout so the tail is well-defined.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_bridge import (BridgeNet, load_head, load_embed,
                                      make_bridge_feat)
from mosaicvm.bench.d3_v1 import make_feat, pos_encode, HORIZON, POS_DIM
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_ws import load_a2, load_bridge
from mosaicvm.bench.d1b_tail_robustness import prefix_metrics
from mosaicvm.bench.d3_stats import stats as canon_stats
from mosaicvm.bench.d3_hydra import HydraHeads, DEPTHS, MAXPREV, D_TOK

PS1001 = "artifacts/d3_scale1000_train/ps1001.pt"
A2 = "artifacts/d3_v1_A2/student.pt"
BEAMS = (1, 2, 4, 8)
FROZEN = {"CAP-DEV70": 3.7571, "SCALE-DEV3": 4.0143}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hydra", default="artifacts/d3_scale1000_train/hydra_full.pt")
    p.add_argument("--data", default="artifacts/d3_capdev70_data")
    p.add_argument("--manifest",
                   default="artifacts/d3_capdev70/prompts_capdev70.json")
    p.add_argument("--output",
                   default="artifacts/d3_scale1000_train/hydra_capdev70.json")
    a = p.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()

    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(Path(A2))
    ps, ps_xm, ps_xs = load_bridge(Path(PS1001))
    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda(); w_norm_g = w_norm.cuda()

    hk = torch.load(a.hydra, map_location="cpu", weights_only=True)
    hyd = HydraHeads(hk["hidden"], hk["depth"]); hyd.load_state_dict(hk["model"])
    hyd = hyd.cuda()
    rmean = hk["rmean"].cuda(); rstd = hk["rstd"].cuda()

    caps = load_captures(a.data, json.loads(Path(a.manifest).read_text()),
                         None, HORIZON, require_both=False)
    val = [c for c in caps if c["split"] == "val"]
    print(f"val: {len(val)} ({time.time()-t0:.0f}s)", flush=True)

    @torch.inference_mode()
    def ps_logits(new_state, driver, cur_tok, t):
        bf = make_bridge_feat(embed, new_state, driver, cur_tok, t)
        z = ps(((bf - ps_xm) / ps_xs).unsqueeze(0))[0].cuda()
        zn = z * torch.rsqrt(z.square().mean() + 1e-6) * w_norm_g
        return (w_head_g @ zn)

    @torch.inference_mode()
    def hydra_logits(root, prev_ids, t, j):
        ps_sum = (sum(embed[pi].float() for pi in prev_ids[-MAXPREV:])
                  if prev_ids else torch.zeros(D_TOK))
        rf = ((root.cuda() - rmean) / rstd).unsqueeze(0)
        pv = ps_sum.unsqueeze(0).cuda()
        po = pos_encode(torch.tensor([t]))[0].unsqueeze(0).cuda()
        dp = torch.tensor([j]).cuda()
        z = hyd(rf, pv, po, dp)[0]
        zn = z * torch.rsqrt(z.square().mean() + 1e-6) * w_norm_g
        return (w_head_g @ zn)

    def a2_step(state, prev, driver, cur_tok, t):
        feat = make_feat(embed, state, state - prev, driver, cur_tok, t)
        pred = a2(((feat - a2_xm) / a2_xs).unsqueeze(0))[0] * t_std + t_mean
        return state + pred

    def lsm(lg):
        return F.log_softmax(lg.float(), dim=-1).detach().cpu()

    def run_beam(c, B):
        S = c["S"]; K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]; h = c["h40"].float()
        root = h[S - 1].flatten()
        # j=0: ps1001 greedy root decision
        ns0 = a2_step(h[S - 1].flatten(), h[S - 2].flatten(),
                      toks[S], toks[S - 1], S)
        lg0 = ps_logits(ns0, toks[S], toks[S - 1], S)
        d0 = int(lg0.argmax())
        # beam item: (drafts, score, state, prev, cur_tok, driver)
        beam = [((d0,), float(lsm(lg0)[d0]), ns0, h[S - 1].flatten(),
                 toks[S], d0)]
        exp = 1
        for depth in range(2, min(4, K + 1)):     # d2, d3
            j = depth - 1                          # speculative index 1,2
            local_k = B if depth == 3 else 1       # branch only at d3
            cand = []
            for pref in beam:
                ns = a2_step(pref[2], pref[3], pref[5], pref[4], S + depth - 1)
                # Hydra logits from committed root + preceding DRAFT tokens
                lg = hydra_logits(root, list(pref[0]), S + depth - 1, j)
                ls = lsm(lg); kk = min(local_k, ls.numel())
                vals, idxs = ls.topk(kk)
                for v, tk in zip(vals.tolist(), idxs.tolist()):
                    tk = int(tk)
                    cand.append((pref[0] + (tk,), pref[1] + float(v),
                                 ns, pref[2], pref[5], tk)); exp += 1
            cand.sort(key=lambda x: x[1], reverse=True)
            beam = cand[:B]
        # freeze; pick longest gold prefix, then greedy tail
        def gpl(d):
            n = 0
            for jj, tk in enumerate(d):
                if tk != toks[S + jj + 1]:
                    break
                n += 1
            return n
        best = max(beam, key=lambda pr: (gpl(pr[0]), pr[1]))
        drafts = list(best[0]); state, prev = best[2], best[3]
        cur_tok, driver = best[4], best[5]
        for jj in range(len(drafts), K):
            t = S + jj
            ns = a2_step(state, prev, driver, cur_tok, t)
            if jj in DEPTHS:                          # d2/d3/d4 via Hydra
                lg = hydra_logits(root, drafts, t, jj)
            else:                                     # d5+ via ps1001 tail
                lg = ps_logits(ns, driver, cur_tok, t)
            tk = int(lg.argmax()); drafts.append(tk)
            cur_tok, driver = driver, tk; prev, state = state, ns
        hits = [drafts[jj] == toks[S + jj + 1] for jj in range(K)]
        return prefix_metrics([int(x) for x in hits], K), exp

    results = {"classification": "D3_HYDRA_PROBE", "n_prompts": len(val),
               "by_B": {}}
    print(f"{'B':>3} {'E[A]':>7} {'S3':>6} {'S4':>6} {'nodes':>6}")
    for B in BEAMS:
        rows, nodes = [], 0
        for c in val:
            m, e = run_beam(c, B); nodes += e
            rows.append({"category": c["category"],
                         "cl_loop": {"E_A": m["E_A"], "Sk": m["Sk"]}})
        cs = canon_stats(rows, "cl_loop", k=32)
        results["by_B"][str(B)] = {"canonical": cs,
                                   "mean_nodes": round(nodes / len(val), 1)}
        print(f"{B:>3} {cs['mean']:>7.3f} {cs['S3']:>6.3f} {cs['S4']:>6.3f} "
              f"{nodes/len(val):>6.1f}  ({time.time()-t0:.0f}s)", flush=True)

    b4 = results["by_B"]["4"]["canonical"]["mean"]
    tag = "CAP-DEV70" if "capdev70" in str(a.data) else (
        "SCALE-DEV3" if "scale1000_data" in str(a.data) else None)
    if tag:
        base = FROZEN[tag]
        results["capability"] = {"set": tag, "frozen_ps1001_B4": base,
                                 "candidate_B4": round(b4, 4),
                                 "delta_vs_frozen": round(b4 - base, 4),
                                 "gate_4.0_met": b4 >= 4.0,
                                 "readiness_4.15": b4 >= 4.15}
        print(f"CAPABILITY[{tag}] B4 {b4:.3f} vs frozen {base:.3f} "
              f"= {b4-base:+.3f}  gate4.0={'MET' if b4>=4 else 'no'}", flush=True)
    Path(a.output).write_text(json.dumps(results, indent=1))
    print(f"wrote {a.output} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
