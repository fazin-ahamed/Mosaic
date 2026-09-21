"""ABP2-minimal: confidence-gated depth-3 branching.

Selective race showed d3-only B8 captures 83% of the d2+d3 gain at 14%
of the nodes. This tests whether branching ONLY when the depth-3
distribution is uncertain recovers the +0.471 at fewer nodes than
fixed d3-B8 (2.671 E[A], 10 nodes).

Policy: root greedy, d2 greedy, then at d3 pick branch width from the
entering-prefix local distribution:
    margin = logp(top1) - logp(top2)   (nats)
    width = 8 if margin < THR else 1
Gold-free; gold consulted only after freeze. Sweep THR.
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
THRS = [0.0, 0.5, 1.0, 2.0, 4.0, 1e9]  # 1e9 == always branch (=fixed B8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="artifacts/d3_scale1000_data")
    p.add_argument("--manifest",
                   default="artifacts/d3_scale1000/prompts_scale1000.json")
    p.add_argument("--split", default="val")
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--output",
                   default="artifacts/d3_scale1000_train/abp2_d3.json")
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

    def run(c, thr, W):
        """root greedy, d2 greedy, d3 conditional width. Returns (m, nodes)."""
        S = c["S"]; K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]; h = c["h40"].float()
        cur = h[S - 1].flatten(); aux = h[S - 2].flatten()
        mf = make_dd_feat(embed, cur, toks[S], toks[S - 1], S)
        mfn = (mf - xm_mean) / xm_std
        afn = (aux - xa_mean) / xa_std
        lg0 = hist_logits(mfn, afn)
        root = int(lg0.argmax())
        state = h[S - 1].flatten(); prev = h[S - 2].flatten()
        cur_tok = toks[S - 1]; driver = toks[S]
        nodes = 1
        drafts = [root]
        # advance the committed greedy state to consume the root token
        ns = a2_step(state, prev, driver, cur_tok, S)
        cur_tok, driver = driver, root
        prev, state = state, ns

        if K < 2:
            hits = [drafts[j] == toks[S + j + 1] for j in range(K)]
            return prefix_metrics([int(x) for x in hits], K), nodes

        # depth 2 greedy
        ns = a2_step(state, prev, driver, cur_tok, S + 1)
        lg2 = cont_logits(ns, driver, cur_tok, S + 1)
        d2 = int(lg2.argmax()); nodes += 1
        drafts.append(d2)
        cur_tok, driver = driver, d2
        prev, state = state, ns

        if K < 3:
            hits = [drafts[j] == toks[S + j + 1] for j in range(K)]
            return prefix_metrics([int(x) for x in hits], K), nodes

        # depth 3 conditional branch
        ns = a2_step(state, prev, driver, cur_tok, S + 2)
        lg3 = cont_logits(ns, driver, cur_tok, S + 2)
        ls3 = lsm(lg3)
        top2 = ls3.topk(2).values.tolist()
        margin = top2[0] - top2[1]
        width = W if margin < thr else 1
        vals, idxs = ls3.topk(width)
        # build width prefixes; each greedy-continues to K
        branch = []
        for v, tok in zip(vals.tolist(), idxs.tolist()):
            tok = int(tok); nodes += 1
            branch.append((drafts + [tok], ns, state, driver, tok))
        # greedy tail each, pick longest gold prefix after freeze
        def tail(pre):
            dfs, st, pv, cur_t, drv = list(pre[0]), pre[1], pre[2], pre[3], pre[4]
            for j in range(len(dfs), K):
                t = S + j
                nst = a2_step(st, pv, drv, cur_t, t)
                lg = cont_logits(nst, drv, cur_t, t)
                tk = int(lg.argmax())
                dfs.append(tk)
                cur_t, drv = drv, tk
                pv, st = st, nst
            return dfs
        best = None; best_gpl = -1
        for pre in branch:
            full = tail(pre)
            n = 0
            for j, tk in enumerate(full):
                if tk != toks[S + j + 1]:
                    break
                n += 1
            if n > best_gpl:
                best_gpl = n; best = full
        hits = [best[j] == toks[S + j + 1] for j in range(K)]
        return prefix_metrics([int(x) for x in hits], K), nodes

    results = {"classification": "D3_ABP2_D3_CONDITIONAL",
               "width": a.width, "n_prompts": len(val), "by_thr": {}}
    print(f"{'THR':>6} {'E[A]':>7} {'S3':>6} {'S4':>6} {'nodes':>6} "
          f"{'brchfrac':>8}")
    for thr in THRS:
        rows = []; nodes = 0; nb = 0
        for c in val:
            m, nd = run(c, thr, a.width)
            rows.append({"category": c["category"],
                         "cl_loop": {"E_A": m["E_A"], "Sk": m["Sk"]}})
            nodes += nd
            if nd > 3:  # more than root+d2+1 => branched at d3
                nb += 1
        cs = canon_stats(rows, "cl_loop", k=32)
        mn = nodes / len(val)
        bf = nb / len(val)
        results["by_thr"][str(thr)] = {
            "mean_EA": cs["mean"], "S3": cs["S3"], "S4": cs["S4"],
            "trim20": cs["trim20"], "mean_nodes": round(mn, 2),
            "branch_fraction": round(bf, 3)}
        label = "always" if thr > 1e8 else f"{thr}"
        print(f"{label:>6} {cs['mean']:>7.3f} {cs['S3']:>6.3f} "
              f"{cs['S4']:>6.3f} {mn:>6.2f} {bf:>8.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    Path(a.output).write_text(json.dumps(results, indent=1))
    print(f"wrote {a.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
