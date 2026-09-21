"""Rank-at-cliff probe (decides branching vs continuation lever).

Mirrors the exact HIST1001 + A2 + ps splice rollout used by
d3_scale_eval (the one that produced C2=.600 C3=.185 on SCALE-DEV3),
but at each rollout position logs the TRUE token's rank in the draft
distribution. This discriminates the two failure modes at the depth-3
cliff:

  true_rank in {1..3}  at pos 3  -> branching / adaptive tree rescues it
  true_rank >> 3       at pos 3  -> continuation state has drifted;
                                    need a better depth model, not a tree

Read-only. No training, no capture. HIST1001 first decision, then the
frozen A2 recurrence + ps bridge continuation.

Per position we report rank conditioned on the chain still being alive
(all previous drafts correct) — that is the regime a real speculative
tree/continuation actually operates in.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

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

HIST = "artifacts/d3_scale1000_train/hist1001.pt"
A2 = "artifacts/d3_v1_A2/student.pt"
BRIDGE = "artifacts/d3_bridge_r3b/bridge_ps/bridge.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="artifacts/d3_scale1000_data")
    p.add_argument("--manifest",
                   default="artifacts/d3_scale1000/prompts_scale1000.json")
    p.add_argument("--split", default="val")
    p.add_argument("--output",
                   default="artifacts/d3_scale1000_train/rank_at_cliff.json")
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

    def rank_of(lg, tok):
        tl = float(lg[tok])
        return int((lg > tl).sum())

    # rank_by_pos[d] = list of true-token ranks at depth d, conditioned
    # on the chain being alive through d-1 (all prior drafts correct)
    rank_alive = defaultdict(list)
    rank_all = defaultdict(list)

    for c in val:
        S = c["S"]; K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]; h = c["h40"].float()
        cur = h[S - 1].flatten(); aux = h[S - 2].flatten()
        mf = make_dd_feat(embed, cur, toks[S], toks[S - 1], S)
        mfn = (mf - xm_mean) / xm_std
        afn = (aux - xa_mean) / xa_std
        lg0 = hist_logits(mfn, afn)
        dd1_draft = int(lg0.argmax())

        state = h[S - 1].flatten(); prev = h[S - 2].flatten()
        cur_tok = toks[S - 1]; driver = toks[S]
        alive = True
        for j in range(K):
            t = S + j
            true_tok = toks[t + 1]
            feat = make_feat(embed, state, state - prev, driver, cur_tok, t)
            fn = (feat - a2_xm) / a2_xs
            pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
            new_state = state + pred
            if j == 0:
                lg = lg0; draft = dd1_draft
            else:
                bf = make_bridge_feat(embed, new_state, driver, cur_tok, t)
                bfn = (bf - b_xm) / b_xs
                with torch.inference_mode():
                    z = bridge(bfn.unsqueeze(0))[0]
                lg = bridge_logits(z); draft = int(lg.argmax())
            rk = rank_of(lg, true_tok)
            rank_all[j + 1].append(rk)
            if alive:
                rank_alive[j + 1].append(rk)
            if draft != true_tok:
                alive = False
            cur_tok = driver; driver = draft
            prev = state; state = new_state

    def bucket(ranks):
        n = len(ranks)
        if n == 0:
            return None
        import statistics as st
        b = {"n": n,
             "rank0": sum(r == 0 for r in ranks) / n,      # greedy hit
             "rank<2": sum(r < 2 for r in ranks) / n,
             "rank<4": sum(r < 4 for r in ranks) / n,
             "rank<8": sum(r < 8 for r in ranks) / n,
             "rank<16": sum(r < 16 for r in ranks) / n,
             "rank<64": sum(r < 64 for r in ranks) / n,
             "median": st.median(ranks),
             "mean": round(sum(ranks) / n, 1)}
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in b.items()}

    out = {"classification": "D3_RANK_AT_CLIFF", "model": "hist1001+A2+ps",
           "n_prompts": len(val),
           "rank_alive": {d: bucket(rank_alive[d]) for d in sorted(rank_alive)},
           "rank_all": {d: bucket(rank_all[d]) for d in sorted(rank_all)}}
    Path(a.output).write_text(json.dumps(out, indent=1))

    print("\n=== TRUE-TOKEN RANK, chain alive (branching-relevant regime) ===")
    print(f"{'depth':>5} {'n':>4} {'rank0':>7} {'<2':>6} {'<4':>6} "
          f"{'<8':>6} {'<16':>7} {'<64':>7} {'med':>5} {'mean':>7}")
    for d in sorted(rank_alive):
        b = bucket(rank_alive[d])
        if b:
            print(f"{d:>5} {b['n']:>4} {b['rank0']:>7} {b['rank<2']:>6} "
                  f"{b['rank<4']:>6} {b['rank<8']:>6} {b['rank<16']:>7} "
                  f"{b['rank<64']:>7} {b['median']:>5} {b['mean']:>7}")
    print(f"wrote {a.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
