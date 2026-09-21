"""Candidate census for SCALE-DEV2 (roadmap S10, S25 analogue).

Exact full-vocab target ranks under a given model on a given
capture corpus/split; Recall@M grid, rerankable/unreachable
sets. Models: hist210 | ws210 | hist700 | ws700 (registry
shared with d3_scale_eval).
"""
from __future__ import annotations

import argparse
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
from mosaicvm.bench.d3_scale_eval import MODELS

MS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   choices=list(MODELS))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()

    m = MODELS[a.model]
    ck = torch.load(m["ckpt"] if m["kind"] == "hist"
                    else m["hist"],
                    map_location="cpu", weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"],
                     ck["aux_dim"])
    hist.load_state_dict(ck["model"]); hist.eval()
    Q = None
    if m["kind"] == "ws":
        ws = torch.load(m["ws"], map_location="cpu",
                        weights_only=True)
        Q = ws["Q"].cuda()
        E_g = torch.load(
            "artifacts/d3_lrdl0/basis_r256.pt",
            map_location="cpu",
            weights_only=True)["E_r"].cuda()
    else:
        w_head, w_norm = load_head()
        w_head_g = w_head.cuda(); w_norm_g = w_norm.cuda()
    embed = load_embed()

    caps = load_captures(str(a.data),
                         json.loads(
                             a.manifest.read_text()),
                         None, HORIZON, require_both=False)
    val = [c for c in caps if c["split"] == a.split]
    print(f"[{a.model}] {a.split}: {len(val)} prompts",
          flush=True)

    ranks = {}
    with torch.inference_mode():
        for c in val:
            S = c["S"]; toks = c["tokens"]
            mf = make_dd_feat(
                embed, c["h40"][S - 1].float().flatten(),
                toks[S], toks[S - 1], S)
            mfn = (mf - ck["xm_mean"]) / ck["xm_std"]
            afn = ((c["h40"][S - 2].float().flatten()
                    - ck["xa_mean"]) / ck["xa_std"])
            z = hist(mfn.unsqueeze(0),
                     afn.unsqueeze(0)).cuda()
            zr = z * torch.rsqrt(
                z.square().mean(-1, keepdim=True) + 1e-6)
            if Q is not None:
                lg = ((zr @ Q.t()) @ E_g.t())[0]
            else:
                lg = w_head_g @ (zr[0] * w_norm_g)
            true = toks[S + 1]
            ranks[c["category"]] = int(
                (lg > lg[true]).sum().item()) + 1

    n = len(ranks)
    result = {"classification": "D3_CENSUS",
              "model": a.model, "split": a.split,
              "n": n,
              "recall": {str(M): round(
                  sum(1 for r in ranks.values()
                      if r <= M) / n, 4)
                  for M in MS},
              "per_prompt_rank": ranks,
              "rerankable_misses": sorted(
                  c for c, r in ranks.items()
                  if 1 < r <= 64),
              "unreachable": {c: r for c, r in
                              sorted(ranks.items())
                              if r > 64}}
    a.output.parent.mkdir(parents=True,
                          exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1))
    print(f"recall@16={result['recall']['16']} "
          f"@64={result['recall']['64']} | "
          f"rerankable {len(result['rerankable_misses'])} "
          f"unreachable {len(result['unreachable'])} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
