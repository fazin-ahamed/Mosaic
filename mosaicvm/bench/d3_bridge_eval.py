"""R2-5 evaluator: closed-loop Decision Bridge proposer.

Production loop (roadmap S32-S35): frozen A2 h40 recurrence +
learned bridge -> z -> frozen production norm + FP32 head ->
draft token; the draft feeds back as the next driver. NO source
L41/L42 tail in the draft path.

Modes per validation prompt (same DEV21 set / convention as
d3_v1_eval):
  one_step  gold h40_{t-1} + true driver -> A2 -> bridge -> head
  cl_loop   chained A2 states; driver = own bridge draft
  tf_loop   chained A2 states; true tokens

Also measures the proposer wall per position (student + bridge +
head, GPU head) — the tail-free latency that competes with the
13.5 ms/position exact-tail marginal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_bridge import (make_bridge_feat, BridgeNet,
                                      load_head, load_embed)
from mosaicvm.bench.d3_v1 import DeltaNet
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d1b_tail_robustness import (prefix_metrics,
                                                prompt_mean_ci)

HORIZON = 32


def load_a2(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = DeltaNet(ck["hidden"], ck["depth"], ck["rank"],
                     aux_out=ck.get("h42_aux_weight", 0) > 0
                     or "aux.weight" in ck["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    return (model, ck["target_mean"], ck["target_std"],
            ck["x_mean"], ck["x_std"])


def load_bridge(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = BridgeNet(ck["hidden"], ck["depth"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["x_mean"], ck["x_std"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge", type=Path,
                   default=Path("artifacts/d3_bridge_r2/bridge.pt"))
    p.add_argument("--a2", type=Path,
                   default=Path("artifacts/d3_v1_A2/student.pt"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path,
                   default=Path("artifacts/stage1_prompts.json"))
    a = p.parse_args()
    torch.set_num_threads(4)

    a2, t_mean, t_std, x_mean, x_std = load_a2(a.a2)
    bridge, b_mean, b_std = load_bridge(a.bridge)
    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda()
    w_norm_g = w_norm.cuda()

    from mosaicvm.bench.d3_v1 import make_feat

    @torch.inference_mode()
    def draft(z):
        """z [4096] cpu -> draft token via GPU head (frozen norm
        + fp32 head). Move z to GPU first, everything GPU-side."""
        zg = z.cuda()
        zn = zg * torch.rsqrt(zg.square().mean() + 1e-6)
        zn = zn * w_norm_g
        lg = w_head_g @ zn
        return int(lg.argmax())

    captures = load_captures(a.data,
                             json.loads(a.manifest.read_text()),
                             None, HORIZON)
    val = [c for c in captures if c["split"] == "val"]
    print(f"val prompts: {len(val)}", flush=True)

    rows = []
    lat_ms = []
    t0 = time.time()
    for c in val:
        S = c["S"]
        K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]
        h = c["h40"].float()

        # exact root sanity: committed h40[S-1] through A2+bridge
        # is NOT guaranteed — the root draft comes from the exact
        # committed h42 (boundary reuse): assert that instead.
        root_draft = toks[S]     # exact by capture construction

        def run(mode):
            state = h[S - 1].flatten()
            prev = h[S - 2].flatten()
            cur_tok = toks[S - 1]
            driver = root_draft
            preds, hits, zs = [], [], []
            for j in range(K):
                t = S + j
                if mode == "one_step":
                    state_in = h[t - 1].flatten()
                    delta_in = h[t - 1].flatten() - h[t - 2].flatten()
                    cur_in = toks[t - 1]
                    drv_in = toks[t]
                else:
                    state_in = state
                    delta_in = state - prev
                    cur_in = cur_tok
                    drv_in = driver
                # A2 predicts h40_t
                feat = make_feat(embed, state_in, delta_in, drv_in,
                                 cur_in, t)
                feat = (feat - x_mean) / x_std
                with torch.no_grad():
                    pred = a2(feat.unsqueeze(0))[0] * t_std + t_mean
                new_state = state_in + pred          # [16384]
                # bridge -> z -> draft
                bf = make_bridge_feat(embed, new_state, drv_in,
                                      cur_in, t)
                bf = (bf - b_mean) / b_std
                with torch.no_grad():
                    z = bridge(bf.unsqueeze(0))[0]
                d = draft(z)
                hits.append(d == toks[t + 1])
                preds.append(new_state)
                zs.append(z)
                cur_tok = drv_in
                driver = d if mode == "cl_loop" else toks[t + 1]
                prev = state_in
                state = new_state
            return preds, hits, zs

        row = {"category": c["category"], "K": K}
        preds_os = None
        for mode in ("one_step", "cl_loop", "tf_loop"):
            preds, hits, zs = run(mode)
            row[mode] = prefix_metrics([int(x) for x in hits], K)
            if mode == "one_step":
                preds_os = preds
        # A2 state quality vs gold h40 (from the one_step run)
        gold = h[S:S + K].flatten(1)
        pf = torch.stack(preds_os)
        row["os_h40_rel_l2"] = float(
            ((pf - gold).norm(dim=1)
             / gold.norm(dim=1).clamp_min(1e-6)).mean())
        rows.append(row)
        print(f"{c['category']} ({time.time()-t0:.0f}s) "
              f"os {row['one_step']['E_A']} "
              f"cl {row['cl_loop']['E_A']} "
              f"tf {row['tf_loop']['E_A']}", flush=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.with_suffix(".partial.json").write_text(
            json.dumps({"complete": False, "rows": rows}, indent=2))

    # proposer wall (student + bridge + GPU head per position)
    c = val[0]
    S = c["S"]
    h = c["h40"].float()
    toks = c["tokens"]
    for _ in range(3):
        feat = make_feat(embed, h[S - 1].flatten(),
                         h[S - 1].flatten() - h[S - 2].flatten(),
                         toks[S], toks[S - 1], S)
        with torch.no_grad():
            pred = a2(((feat - x_mean) / x_std).unsqueeze(0))[0
                       ] * t_std + t_mean
        bf = make_bridge_feat(embed, h[S - 1].flatten() + pred,
                              toks[S], toks[S - 1], S)
        with torch.no_grad():
            z = bridge(((bf - b_mean) / b_std).unsqueeze(0))[0]
        draft(z)
    torch.cuda.synchronize()
    for rep in range(20):
        t = time.perf_counter()
        feat = make_feat(embed, h[S - 1].flatten(),
                         h[S - 1].flatten() - h[S - 2].flatten(),
                         toks[S], toks[S - 1], S)
        with torch.no_grad():
            pred = a2(((feat - x_mean) / x_std).unsqueeze(0))[0
                       ] * t_std + t_mean
        bf = make_bridge_feat(embed, h[S - 1].flatten() + pred,
                              toks[S], toks[S - 1], S)
        with torch.no_grad():
            z = bridge(((bf - b_mean) / b_std).unsqueeze(0))[0]
        d = draft(z)
        torch.cuda.synchronize()
        lat_ms.append((time.perf_counter() - t) * 1e3)
    lat = round(sorted(lat_ms)[len(lat_ms) // 2], 3)

    aggregate = {}
    for mode in ("one_step", "cl_loop", "tf_loop"):
        vals = [r[mode]["E_A"] for r in rows]
        aggregate[mode] = {
            "n_prompts": len(rows), "E_A": sum(vals) / len(vals),
            "exploratory_prompt_ci95": prompt_mean_ci(vals),
            "per_prompt_E_A": vals,
            "S1": sum(r[mode]["Sk"][0] for r in rows) / len(rows),
            "S2": sum(r[mode]["Sk"][1] for r in rows) / len(rows),
            "S4": sum(r[mode]["Sk"][3] for r in rows) / len(rows),
        }
    result = {
        "classification": "D3_R2_DECISION_BRIDGE_EVAL",
        "production_certified": False,
        "student_pass": False,
        "draft_path": "frozen A2 h40 recurrence + learned bridge "
                      "+ frozen norm + FP32 head (NO source tail)",
        "proposer_wall_ms_per_position": lat,
        "vs_source_tail_marginal_ms": 13.5,
        "bridge": str(a.bridge.resolve()),
        "a2": str(a.a2.resolve()),
        "rows": rows,
        "aggregate": aggregate,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
    }
    a.output.write_text(json.dumps(result, indent=2))
    a.output.with_suffix(".partial.json").unlink(missing_ok=True)
    print(json.dumps({"aggregate": {m: v["E_A"] for m, v in
                                    aggregate.items()},
                      "proposer_wall_ms": lat}, indent=2))


if __name__ == "__main__":
    main()
