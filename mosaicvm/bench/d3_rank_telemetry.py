"""Ranking telemetry pass (roadmap S21/S25). READ-ONLY.

For each checkpoint (HIST-1 baseline + LOW/MED/HIGH rank arms),
over the 21 DEV roots:

  target rank / logit / NLL
  hardest-wrong token / logit
  target-vs-wrong margin
  Top1/2/4/8/16 membership
  per-domain aggregation
  DD standalone wall (ms/first-position) and VRAM

The frozen acceptance evaluator is untouched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_dd1_hist import DDHistNet, D_STATE
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import load_head, load_embed
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON

CKPTS = {
    "HIST-1": "artifacts/d3_dd1_hist/dd1_hist_prev/dd1_hist.pt",
    "LOW": "artifacts/d3_dd1_rank/rank_LOW/dd1_hist.pt",
    "MED": "artifacts/d3_dd1_rank/rank_MED/dd1_hist.pt",
    "HIGH": "artifacts/d3_dd1_rank/rank_HIGH/dd1_hist.pt",
}
OUT = Path("artifacts/d3_dd1_rank_telemetry.json")


def main():
    torch.set_num_threads(4)
    t0 = time.time()

    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda()
    w_norm_g = w_norm.cuda()
    torch.cuda.reset_peak_memory_stats()

    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    print(f"{len(val)} DEV prompts ({time.time()-t0:.0f}s)",
          flush=True)

    results = {}
    for name, path in CKPTS.items():
        ck = torch.load(path, map_location="cpu",
                        weights_only=True)
        model = DDHistNet(ck["hidden"], ck["depth"],
                          ck["aux_dim"])
        model.load_state_dict(ck["model"])
        model.eval()
        xm_mean, xm_std = ck["xm_mean"], ck["xm_std"]
        xa_mean, xa_std = ck["xa_mean"], ck["xa_std"]
        sfield = ck["state_field"]

        per_prompt = {}
        # warmup for wall timing
        c0 = val[0]; S0 = c0["S"]
        _ = model(((make_dd_feat(embed,
             c0[sfield][S0-1].float().flatten(),
             c0["tokens"][S0], c0["tokens"][S0-1], S0)
             - xm_mean) / xm_std).unsqueeze(0),
            ((c0[sfield][S0-2].float().flatten() - xa_mean)
             / xa_std).unsqueeze(0))

        wall_ms = []
        with torch.inference_mode():
            for c in val:
                S = c["S"]; toks = c["tokens"]
                cur = c[sfield][S-1].float().flatten()
                aux = c[sfield][S-2].float().flatten()
                mf = make_dd_feat(embed, cur, toks[S],
                                  toks[S-1], S)
                mfn = (mf - xm_mean) / xm_std
                afn = (aux - xa_mean) / xa_std
                tw = time.perf_counter()
                z = model(mfn.unsqueeze(0),
                          afn.unsqueeze(0))[0]
                zg = z.cuda()
                zn = zg * torch.rsqrt(
                    zg.square().mean() + 1e-6) * w_norm_g
                lg = w_head_g @ zn
                torch.cuda.synchronize()
                wall_ms.append(
                    (time.perf_counter() - tw) * 1000)

                true = toks[S+1]
                tl = float(lg[true])
                pred = int(lg.argmax())
                nll = float(F.cross_entropy(
                    lg.unsqueeze(0).float(),
                    torch.tensor([true]).cuda()))
                # hardest wrong: target-masked argmax
                lgm = lg.clone()
                lgm[true] = float("-inf")
                hw_tok = int(lgm.argmax())
                hw_l = float(lg[hw_tok])
                margin = tl - hw_l
                rank = int((lg > tl).sum().item()) + 1
                topk = {str(k): int(true in
                           lg.topk(k).indices.tolist())
                        for k in (1, 2, 4, 8, 16)}
                per_prompt[c["category"]] = {
                    "hit": pred == true,
                    "target_rank": rank,
                    "target_logit": round(tl, 4),
                    "hardest_wrong_token": hw_tok,
                    "hardest_wrong_logit": round(hw_l, 4),
                    "margin": round(margin, 4),
                    "nll": round(nll, 4),
                    "topk": topk,
                }

        # aggregate
        n = len(per_prompt)
        agg = {
            "S1": sum(v["hit"] for v in
                      per_prompt.values()) / n,
            "mean_margin": sum(v["margin"] for v in
                               per_prompt.values()) / n,
            "mean_nll": sum(v["nll"] for v in
                            per_prompt.values()) / n,
            "mean_target_rank": sum(v["target_rank"]
                                    for v in
                                    per_prompt.values()) / n,
            "topk": {k: sum(v["topk"][k] for v in
                            per_prompt.values()) / n
                     for k in ("1", "2", "4", "8", "16")},
        }
        # per-domain
        dom = {}
        for cat, v in per_prompt.items():
            d = cat.rsplit("_", 1)[0]
            dom.setdefault(d, []).append(v)
        agg["per_domain"] = {
            d: {"S1": sum(v["hit"] for v in vs) / len(vs),
                "mean_margin": round(sum(
                    v["margin"] for v in vs) / len(vs), 4),
                "mean_nll": round(sum(
                    v["nll"] for v in vs) / len(vs), 4)}
            for d, vs in sorted(dom.items())}
        agg["dd_wall_ms"] = {
            "mean": round(sum(wall_ms) / len(wall_ms), 3),
            "min": round(min(wall_ms), 3),
            "max": round(max(wall_ms), 3)}
        agg["vram_mb"] = round(torch.cuda.max_memory_allocated()
                               / 1e6, 1)
        results[name] = {"per_prompt": per_prompt,
                         "aggregate": agg}
        print(f"{name}: S1={agg['S1']:.3f} "
              f"mean_margin={agg['mean_margin']:.3f} "
              f"mean_NLL={agg['mean_nll']:.3f} "
              f"rank={agg['mean_target_rank']:.1f} "
              f"wall={agg['dd_wall_ms']['mean']}ms "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.reset_peak_memory_stats()

    # C2 denominator-artifact check: absolute S2/S1 counts
    c2_check = {}
    for name in CKPTS:
        ev = {"HIST-1":
              "artifacts/d3_dd1_hist/dd1_hist_prev_eval.json",
              "LOW": "artifacts/d3_dd1_rank/rank_LOW_eval.json",
              "MED": "artifacts/d3_dd1_rank/rank_MED_eval.json",
              "HIGH": "artifacts/d3_dd1_rank/rank_HIGH_eval.json"
              }[name]
        e = json.loads(Path(ev).read_text())
        cs = e["canonical_stats"]
        c2_check[name] = {
            "S1_count": round(cs["S1"] * 21),
            "S2_count": round(cs["S2"] * 21),
            "S3_count": round(cs["S3"] * 21),
            "C2_reported": e["conditional_survival"]["C2"]}
    results["c2_denominator_check"] = c2_check

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nC2 denominator check (S2 count / S1 count):")
    for name, v in c2_check.items():
        print(f"  {name}: S1={v['S1_count']} S2={v['S2_count']}"
              f" -> C2={v['C2_reported']}")
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
