"""DD-1 HIST evaluator: standalone boundary S1 + splice with ps."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "."); sys.path.insert(0, "mosaicvm/bench")
from mosaicvm.bench.d3_dd1_hist import DDHistNet, D_STATE
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import (BridgeNet, load_head,
                                      load_embed, make_bridge_feat)
from mosaicvm.bench.d3_v1 import (DeltaNet, make_feat,
                                  HORIZON, pos_encode)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d1b_tail_robustness import prefix_metrics
from mosaicvm.bench.d3_stats import stats as canon_stats


def load_hist(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = DDHistNet(ck["hidden"], ck["depth"], ck["aux_dim"])
    m.load_state_dict(ck["model"]); m.eval()
    return (m, ck["xm_mean"], ck["xm_std"],
            ck["xa_mean"], ck["xa_std"], ck["hist_mode"],
            ck["state_field"])


def load_a2(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = DeltaNet(ck["hidden"], ck["depth"], ck["rank"],
                 aux_out=ck.get("h42_aux_weight", 0) > 0
                 or "aux.weight" in ck["model"])
    m.load_state_dict(ck["model"]); m.eval()
    return (m, ck["target_mean"], ck["target_std"],
            ck["x_mean"], ck["x_std"])


def load_bridge(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = BridgeNet(ck["hidden"], ck["depth"])
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["x_mean"], ck["x_std"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--a2", type=Path,
                   default=Path("artifacts/d3_v1_A2/student.pt"))
    p.add_argument("--bridge", type=Path,
                   default=Path("artifacts/d3_bridge_r3b/bridge_ps/bridge.pt"))
    a = p.parse_args()
    torch.set_num_threads(4)

    t0 = time.time()
    hist, xm_mean, xm_std, xa_mean, xa_std, hmode, sfield = \
        load_hist(a.ckpt)
    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(a.a2)
    bridge, b_xm, b_xs = load_bridge(a.bridge)
    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda(); w_norm_g = w_norm.cuda()

    caps = load_captures(a.data,
                         json.loads(Path("artifacts/stage1_prompts.json").read_text()),
                         None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    print(f"val: {len(val)} ({time.time()-t0:.0f}s)", flush=True)

    @torch.inference_mode()
    def head_argmax(z):
        zg = z.cuda()
        zn = zg * torch.rsqrt(zg.square().mean() + 1e-6) * w_norm_g
        return int((w_head_g @ zn).argmax())

    def get_aux(c, S):
        cur = c[sfield][S-1].float().flatten()
        prev = c[sfield][S-2].float().flatten()
        if hmode == "zero":
            return torch.zeros(D_STATE)
        elif hmode == "prev":
            return prev
        else:
            return cur - prev

    # ---- standalone boundary eval ----
    correct = 0; topk = {1:0, 2:0, 4:0, 8:0, 16:0}
    for c in val:
        S = c["S"]; toks = c["tokens"]
        cur = c[sfield][S-1].float().flatten()
        aux = get_aux(c, S)
        mf = make_dd_feat(embed, cur, toks[S], toks[S-1], S)
        mfn = (mf - xm_mean) / xm_std
        afn = (aux - xa_mean) / xa_std
        z = hist(mfn.unsqueeze(0), afn.unsqueeze(0))[0]
        draft = head_argmax(z)
        true = toks[S+1]
        correct += draft == true
        lg_z = z.cuda()
        zn = lg_z * torch.rsqrt(lg_z.square().mean() + 1e-6) * w_norm_g
        lg = w_head_g @ zn
        for k in topk:
            topk[k] += int(true in lg.topk(k).indices.tolist())
    n = len(val)
    standalone = {"S1": round(correct/n, 4),
                  "Top_k": {str(k): round(v/n, 4) for k, v in sorted(topk.items())}}
    print(f"standalone: S1={correct}/{n} Top4={topk[4]}/{n}", flush=True)

    # ---- splice eval ----
    rows = []
    dd1_hits = 0
    for c in val:
        S = c["S"]; K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]; h = c["h40"].float()
        # DD-1 first decision
        cur = c[sfield][S-1].float().flatten()
        aux = get_aux(c, S)
        mf = make_dd_feat(embed, cur, toks[S], toks[S-1], S)
        mfn = (mf - xm_mean) / xm_std
        afn = (aux - xa_mean) / xa_std
        dd1_z = hist(mfn.unsqueeze(0), afn.unsqueeze(0))[0]
        dd1_draft = head_argmax(dd1_z)
        # A2 + bridge continuation
        state = h[S-1].flatten(); prev = h[S-2].flatten()
        cur_tok = toks[S-1]; driver = toks[S]
        hits = []
        for j in range(K):
            t = S + j
            feat = make_feat(embed, state, state - prev, driver, cur_tok, t)
            fn = (feat - a2_xm) / a2_xs
            pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
            new_state = state + pred
            if j == 0:
                draft = dd1_draft
            else:
                bf = make_bridge_feat(embed, new_state, driver, cur_tok, t)
                bfn = (bf - b_xm) / b_xs
                z = bridge(bfn.unsqueeze(0))[0]
                draft = head_argmax(z)
            hits.append(draft == toks[t+1])
            cur_tok = driver; driver = draft
            prev = state; state = new_state
        dd1_hits += hits[0]
        m = prefix_metrics([int(x) for x in hits], K)
        rows.append({"category": c["category"], "m": m})
    # canonical stats
    stat_rows = [{"category": r["category"],
                  "cl_loop": {"E_A": r["m"]["E_A"], "Sk": r["m"]["Sk"]}}
                 for r in rows]
    cs = canon_stats(stat_rows, "cl_loop", k=32)
    c2 = cs["S2"]/cs["S1"] if cs["S1"]>0 else 0
    c3 = cs["S3"]/cs["S2"] if cs["S2"]>0 else 0
    c4 = cs["S4"]/cs["S3"] if cs["S3"]>0 else 0
    result = {
        "classification": "D3_DD1_HIST_EVAL",
        "hist_mode": hmode,
        "standalone": standalone,
        "dd1_splice_S1_count": f"{dd1_hits}/{len(val)}",
        "canonical_stats": cs,
        "conditional_survival": {"C2": round(c2,4), "C3": round(c3,4), "C4": round(c4,4)},
        "mechanism_gate": {
            "S1_count": f"{dd1_hits}/{len(val)}",
            "S1_gate": ">= 9",
            "S1_passed": dd1_hits >= 9,
            "S4": cs["S4"], "S4_passed": cs["S4"] > 0,
            "overall": "PASS" if dd1_hits >= 9 and cs["S4"] > 0 else "FAIL"},
        "checkpoint": str(a.ckpt.resolve()),
    }
    a.output.write_text(json.dumps(result, indent=1))
    print(f"splice: mean={cs['mean']} S1={dd1_hits}/{len(val)} "
          f"S4={cs['S4']} deep={cs['deepest']} "
          f"gate={result['mechanism_gate']['overall']}")
    print(f"wrote {a.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
