"""DTR-0: candidate coverage census (roadmap S11-S15).
Zero training.

For every DEV21 root, compute the exact full-vocabulary
target rank under BOTH frozen models (HIST-1 full-head path
and WS-1 rank-256 path). Report:

  Recall@M for M = 1,2,4,8,16,32,64,128,256,512
  per-prompt exact ranks (persisted)
  per-domain rank distributions
  union coverage (target in WS-1 top-M OR HIST-1 top-M)
  the rerankable set (miss with rank <= 64) per model

Decision rule (S14-S15): prefer M<=64; acceptable M<=128; if
no candidate set up to M=256 materially exceeds current
breadth, reranking cannot be the principal solution.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_dd1_hist import DDHistNet
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import load_head, load_embed
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON

HIST1 = Path("artifacts/d3_dd1_hist/dd1_hist_prev/dd1_hist.pt")
BASIS = Path("artifacts/d3_lrdl0/basis_r256.pt")
WS1 = Path("artifacts/d3_ws/ws1.pt")
OUT = Path("artifacts/d3_dtr0/census.json")
MS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    Path("artifacts/d3_dtr0").mkdir(parents=True,
                                    exist_ok=True)

    ck = torch.load(HIST1, map_location="cpu",
                    weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"], ck["aux_dim"])
    hist.load_state_dict(ck["model"])
    hist.eval()
    xm_mean, xm_std = ck["xm_mean"], ck["xm_std"]
    xa_mean, xa_std = ck["xa_mean"], ck["xa_std"]

    ws1 = torch.load(WS1, map_location="cpu",
                     weights_only=True)
    Q = ws1["Q"].cuda()
    bk = torch.load(BASIS, map_location="cpu",
                    weights_only=True)
    E_g = bk["E_r"].cuda()

    w_head, w_norm = load_head()
    w_head_g = w_head.cuda()
    w_norm_g = w_norm.cuda()
    embed = load_embed()

    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    print(f"{len(val)} DEV prompts ({time.time()-t0:.0f}s)",
          flush=True)

    per_prompt = {}
    with torch.inference_mode():
        for c in val:
            S = c["S"]; toks = c["tokens"]
            cur = c["h40"][S - 1].float().flatten()
            aux = c["h40"][S - 2].float().flatten()
            mf = make_dd_feat(embed, cur, toks[S],
                              toks[S - 1], S)
            mfn = (mf - xm_mean) / xm_std
            afn = (aux - xa_mean) / xa_std
            z = hist(mfn.unsqueeze(0), afn.unsqueeze(0))
            zg = z.cuda()
            zr = zg * torch.rsqrt(
                zg.square().mean(-1, keepdim=True) + 1e-6)
            # HIST-1 full-head path
            lg_h = w_head_g @ (zr[0] * w_norm_g)
            # WS-1 path
            lg_w = ((zr @ Q.t()) @ E_g.t())[0]
            true = toks[S + 1]
            rh = int((lg_h > lg_h[true]).sum().item()) + 1
            rw = int((lg_w > lg_w[true]).sum().item()) + 1
            # union: target in HIST-1 top-M or WS-1 top-M
            h_top = set(lg_h.topk(512).indices.tolist())
            w_top = set(lg_w.topk(512).indices.tolist())
            # union rank: min(rank_h, rank_w) is NOT the same
            # as union-set membership; compute smallest M such
            # that target ∈ h_top[:M] ∪ w_top[:M]
            union_rank = None
            h_ord = lg_h.argsort(descending=True).tolist()
            w_ord = lg_w.argsort(descending=True).tolist()
            h_pos = h_ord.index(true)
            w_pos = w_ord.index(true)
            union_rank = min(max(h_pos, w_pos) + 1,
                            max(w_pos, h_pos) + 1)
            # precise: target in union of top-M sets iff
            # h_pos < M or w_pos < M
            lo, hi = 1, 512
            while lo < hi:
                mid = (lo + hi) // 2
                if h_pos < mid or w_pos < mid:
                    hi = mid
                else:
                    lo = mid + 1
            union_rank = lo
            per_prompt[c["category"]] = {
                "true_token": true,
                "rank_hist1": rh, "rank_ws1": rw,
                "hit_hist1": rh == 1, "hit_ws1": rw == 1,
                "union_rank_at_M": union_rank,
                "h_pos": h_pos, "w_pos": w_pos,
            }
            print(f"  {c['category']}: HIST-1 r{rh} "
                  f"WS-1 r{rw}", flush=True)

    n = len(per_prompt)

    def recall(key, M):
        return sum(1 for v in per_prompt.values()
                   if v[key] <= M) / n

    result = {"classification": "D3_DTR0_CENSUS",
              "n": n, "M_grid": list(MS),
              "recall_hist1": {str(M): round(
                  recall("rank_hist1", M), 4) for M in MS},
              "recall_ws1": {str(M): round(
                  recall("rank_ws1", M), 4) for M in MS},
              "recall_union": {str(M): round(
                  recall("union_rank_at_M", M), 4)
                  for M in MS},
              "per_prompt": per_prompt}

    # per-domain rank distribution (WS-1)
    dom = {}
    for cat, v in per_prompt.items():
        dom.setdefault(cat.rsplit("_", 1)[0],
                       []).append(v["rank_ws1"])
    result["per_domain_rank_ws1"] = {
        d: sorted(r) for d, r in sorted(dom.items())}

    # rerankable set: miss with WS-1 rank <= 64
    rerankable = {c: v["rank_ws1"] for c, v in
                  per_prompt.items()
                  if v["rank_ws1"] > 1 and v["rank_ws1"] <= 64}
    unreachable = {c: v["rank_ws1"] for c, v in
                   per_prompt.items()
                   if v["rank_ws1"] > 64}
    result["rerankable_misses_ws1"] = rerankable
    result["unreachable_beyond_64"] = unreachable

    # branch decision inputs
    r16 = result["recall_ws1"]["16"]
    r64 = result["recall_ws1"]["64"]
    r256 = result["recall_ws1"]["256"]
    result["branch_inputs"] = {
        "recall16": r16, "recall64": r64,
        "recall256": r256,
        "current_S1": 10 / 21,
        "reranker_ceiling_at_16": r16,
        "reranker_ceiling_at_64": r64,
        "note": "ceiling = recall; reranker converts a "
                "subset of retrieved non-top-1 targets"}

    OUT.write_text(json.dumps(result, indent=1))
    print("\n=== DTR-0 CENSUS ===")
    print(f"{'M':>5} {'HIST-1':>8} {'WS-1':>8} {'union':>8}")
    for M in MS:
        print(f"{M:>5} {result['recall_hist1'][str(M)]:>8.3f}"
              f" {result['recall_ws1'][str(M)]:>8.3f}"
              f" {result['recall_union'][str(M)]:>8.3f}")
    print(f"\nrerankable misses (rank<=64): "
          f"{json.dumps(rerankable)}")
    print(f"unreachable (>64): "
          f"{json.dumps(unreachable)}")
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
