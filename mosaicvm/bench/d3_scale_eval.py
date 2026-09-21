"""SCALE evaluation harness (roadmap S17, S25-S27).

Evaluates any frozen first-decision model on any capture
corpus/split:

  --model hist84 | ws84 | hist210 | ws210
  --data artifacts/d3_scale_data
  --manifest artifacts/d3_scale/prompts_scale.json
  --split val
  --tag SCALE-DEV70

Standalone telemetry (hit, exact rank, margin, NLL, Top-k,
per-domain, wall) + mandatory splice through frozen A2 + ps
with canonical d3_stats. Reuses the frozen d3_ws eval helpers.
"""
from __future__ import annotations

import argparse
import hashlib
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
from mosaicvm.bench.d3_v1 import (DeltaNet, make_feat,
                                  HORIZON)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_ws import (eval_telemetry, splice_eval,
                                  load_a2, load_bridge)

MODELS = {
    "hist84": {"kind": "hist",
               "ckpt": "artifacts/d3_dd1_hist/"
                       "dd1_hist_prev/dd1_hist.pt"},
    "ws84": {"kind": "ws",
             "hist": "artifacts/d3_dd1_hist/"
                     "dd1_hist_prev/dd1_hist.pt",
             "ws": "artifacts/d3_ws/ws1.pt"},
    "hist210": {"kind": "hist",
                "ckpt": "artifacts/d3_scale210/"
                        "hist210.pt"},
    "ws210": {"kind": "ws",
              "hist": "artifacts/d3_scale210/hist210.pt",
              "ws": "artifacts/d3_scale210/ws210.pt"},
    "hist700": {"kind": "hist",
                "ckpt": "artifacts/d3_scale700_train/"
                        "hist700.pt"},
    "ws700": {"kind": "ws",
              "hist": "artifacts/d3_scale700_train/"
                      "hist700.pt",
              "ws": "artifacts/d3_scale700_train/"
                    "ws700.pt"},
    "hist1001": {"kind": "hist",
                 "ckpt": "artifacts/d3_scale1000_train/"
                         "hist1001.pt"},
    "ws1001": {"kind": "ws",
               "hist": "artifacts/d3_scale1000_train/"
                       "hist1001.pt",
               "ws": "artifacts/d3_scale1000_train/"
                     "ws1001.pt"},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   choices=list(MODELS))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--tag", default="")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()

    spec = a.manifest
    m = MODELS[a.model]

    if m["kind"] == "hist":
        ck = torch.load(m["ckpt"], map_location="cpu",
                        weights_only=True)
        hist = DDHistNet(ck["hidden"], ck["depth"],
                         ck["aux_dim"])
        hist.load_state_dict(ck["model"]); hist.eval()
        Q = None
    else:
        ck = torch.load(m["hist"], map_location="cpu",
                        weights_only=True)
        hist = DDHistNet(ck["hidden"], ck["depth"],
                         ck["aux_dim"])
        hist.load_state_dict(ck["model"]); hist.eval()
        ws = torch.load(m["ws"], map_location="cpu",
                        weights_only=True)
        Q = ws["Q"].cuda()
    xm_mean, xm_std = ck["xm_mean"], ck["xm_std"]
    xa_mean, xa_std = ck["xa_mean"], ck["xa_std"]
    sfield = ck["state_field"]

    w_head, w_norm = load_head()
    w_head_g = w_head.cuda(); w_norm_g = w_norm.cuda()
    embed = load_embed()
    if Q is not None:
        bk = torch.load(
            "artifacts/d3_lrdl0/basis_r256.pt",
            map_location="cpu", weights_only=True)
        E_g = bk["E_r"].cuda()

    caps = load_captures(str(a.data),
                         json.loads(spec.read_text()),
                         None, HORIZON, require_both=False)
    val = [c for c in caps if c["split"] == a.split]
    print(f"[{a.model}] {a.split}: {len(val)} prompts "
          f"({time.time()-t0:.0f}s)", flush=True)

    eval_telemetry.xm_mean = xm_mean
    eval_telemetry.xm_std = xm_std
    eval_telemetry.xa_mean = xa_mean
    eval_telemetry.xa_std = xa_std
    splice_eval.xm_mean = xm_mean
    splice_eval.xm_std = xm_std
    splice_eval.xa_mean = xa_mean
    splice_eval.xa_std = xa_std

    @torch.inference_mode()
    def z_rms_of(mfn, afn):
        z = hist(mfn.unsqueeze(0), afn.unsqueeze(0))
        zg = z.cuda()
        return zg * torch.rsqrt(
            zg.square().mean(-1, keepdim=True) + 1e-6)

    if Q is None:
        @torch.inference_mode()
        def logits_fn(mfn, afn):
            zr = z_rms_of(mfn, afn)
            return (w_head_g @ (zr[0] * w_norm_g))
    else:
        @torch.inference_mode()
        def logits_fn(mfn, afn):
            zr = z_rms_of(mfn, afn)
            return ((zr @ Q.t()) @ E_g.t())[0]

    pp, agg = eval_telemetry(logits_fn, val, embed, t0,
                             a.model)
    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(
        Path("artifacts/d3_v1_A2/student.pt"))
    bridge, b_xm, b_xs = load_bridge(
        Path("artifacts/d3_bridge_r3b/bridge_ps/bridge.pt"))
    dd1_hits, cs, cond = splice_eval(
        val, embed, logits_fn, a2, t_mean, t_std, a2_xm,
        a2_xs, bridge, b_xm, b_xs, w_head_g, w_norm_g, t0)
    print(f"[{a.model}] splice: mean={cs['mean']} "
          f"S1={dd1_hits}/{len(val)} S4={cs['S4']} "
          f"deep={cs['deepest']}", flush=True)

    result = {
        "classification": "D3_SCALE_EVAL",
        "model": a.model, "tag": a.tag, "split": a.split,
        "data": str(a.data),
        "manifest_sha256": hashlib.sha256(
            spec.read_bytes()).hexdigest(),
        "standalone": agg, "per_prompt": pp,
        "dd1_splice_S1_count": f"{dd1_hits}/{len(val)}",
        "canonical_stats": cs,
        "conditional_survival": cond,
        "checkpoint_sha256": hashlib.sha256(
            Path(m["ckpt"] if m["kind"] == "hist"
                 else m["ws"]).read_bytes()).hexdigest(),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1))
    print(f"wrote {a.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
