"""LRDL evaluator: standalone boundary telemetry (S31) +
mandatory splice through frozen A2 + ps (S32).

First decision: LRDL q → logits = E_r @ q (frozen basis).
Continuation: unchanged A2 recurrence + ps bridge.
Canonical statistics via d3_stats.
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

from mosaicvm.bench.d3_lrdl import LRDLNet, RANK
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import (BridgeNet, load_head,
                                      load_embed,
                                      make_bridge_feat)
from mosaicvm.bench.d3_v1 import (DeltaNet, make_feat,
                                  HORIZON, pos_encode)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d1b_tail_robustness import prefix_metrics
from mosaicvm.bench.d3_stats import stats as canon_stats

import os


def load_lrdl(path):
    ck = torch.load(path, map_location="cpu",
                    weights_only=True)
    m = LRDLNet(ck["hidden"], ck["depth"], ck["aux_dim"],
                ck["rank"])
    m.load_state_dict(ck["model"]); m.eval()
    return (m, ck["xm_mean"], ck["xm_std"],
            ck["xa_mean"], ck["xa_std"], ck["state_field"],
            ck["rank"])


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
    p.add_argument("--ckpt", type=Path,
                   default=Path("artifacts/d3_lrdl0/"
                                "lrdl0_r256/lrdl0.pt"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--a2", type=Path,
                   default=Path("artifacts/d3_v1_A2/student.pt"))
    p.add_argument("--bridge", type=Path,
                   default=Path("artifacts/d3_bridge_r3b/"
                                "bridge_ps/bridge.pt"))
    a = p.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()

    (lrdl, xm_mean, xm_std, xa_mean, xa_std, sfield,
     ck_rank) = load_lrdl(a.ckpt)
    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(a.a2)
    bridge, b_xm, b_xs = load_bridge(a.bridge)
    embed = load_embed()
    # frozen basis (rank matched to checkpoint) + ps head
    basis_path = Path(f"artifacts/d3_lrdl0/"
                      f"basis_r{ck_rank}.pt")
    bk = torch.load(basis_path, map_location="cpu",
                    weights_only=True)
    assert bk["rank"] == ck_rank, "basis rank mismatch"
    E_g = bk["E_r"].cuda()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda()
    w_norm_g = w_norm.cuda()
    torch.cuda.reset_peak_memory_stats()

    caps = load_captures(a.data,
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    print(f"val: {len(val)} ({time.time()-t0:.0f}s)",
          flush=True)

    @torch.inference_mode()
    def lrdl_logits(mfn, afn):
        q = lrdl(mfn.unsqueeze(0), afn.unsqueeze(0))[0].cuda()
        return q @ E_g.t()

    @torch.inference_mode()
    def head_argmax(z):
        zg = z.cuda()
        zn = zg * torch.rsqrt(zg.square().mean() + 1e-6
                              ) * w_norm_g
        return int((w_head_g @ zn).argmax())

    # ---- standalone boundary telemetry (S31) ----
    per_prompt = {}
    wall_ms = []
    # warmup
    c0 = val[0]; S0 = c0["S"]
    _ = lrdl_logits(
        (make_dd_feat(embed, c0[sfield][S0-1].float().flatten(),
                      c0["tokens"][S0], c0["tokens"][S0-1], S0)
         - xm_mean) / xm_std,
        (c0[sfield][S0-2].float().flatten() - xa_mean) / xa_std)
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
            lg = lrdl_logits(mfn, afn)
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - tw) * 1000)
            true = toks[S+1]
            tl = float(lg[true])
            pred = int(lg.argmax())
            nll = float(F.cross_entropy(
                lg.unsqueeze(0).float(),
                torch.tensor([true]).cuda()))
            lgm = lg.clone(); lgm[true] = float("-inf")
            hw = int(lgm.argmax())
            margin = tl - float(lg[hw])
            rank = int((lg > tl).sum().item()) + 1
            per_prompt[c["category"]] = {
                "hit": pred == true, "target_rank": rank,
                "target_logit": round(tl, 4),
                "hardest_wrong_token": hw,
                "margin": round(margin, 4),
                "nll": round(nll, 4),
                "topk": {str(k): int(true in
                           lg.topk(k).indices.tolist())
                         for k in (1, 2, 4, 8, 16)},
            }
    n = len(per_prompt)
    agg = {
        "S1": sum(v["hit"] for v in per_prompt.values()) / n,
        "mean_margin": round(sum(v["margin"] for v in
                                 per_prompt.values()) / n, 4),
        "mean_nll": round(sum(v["nll"] for v in
                              per_prompt.values()) / n, 4),
        "mean_target_rank": round(sum(v["target_rank"]
                                      for v in
                                      per_prompt.values()) / n, 2),
        "topk": {k: round(sum(v["topk"][k] for v in
                              per_prompt.values()) / n, 4)
                 for k in ("1", "2", "4", "8", "16")},
        "dd_wall_ms": {"mean": round(sum(wall_ms) / len(wall_ms),
                                     3),
                       "min": round(min(wall_ms), 3),
                       "max": round(max(wall_ms), 3)},
        "vram_mb": round(torch.cuda.max_memory_allocated()
                         / 1e6, 1),
    }
    dom = {}
    for cat, v in per_prompt.items():
        dom.setdefault(cat.rsplit("_", 1)[0], []).append(v)
    agg["per_domain"] = {
        d: {"S1": round(sum(v["hit"] for v in vs) / len(vs), 4),
            "mean_margin": round(sum(v["margin"] for v in vs)
                                 / len(vs), 4),
            "mean_nll": round(sum(v["nll"] for v in vs)
                              / len(vs), 4)}
        for d, vs in sorted(dom.items())}
    print(f"standalone: S1={agg['S1']:.4f} "
          f"Top4={agg['topk']['4']} NLL={agg['mean_nll']} "
          f"wall={agg['dd_wall_ms']['mean']}ms "
          f"VRAM={agg['vram_mb']}MB", flush=True)

    # ---- splice (S32) ----
    torch.cuda.reset_peak_memory_stats()
    rows = []
    dd1_hits = 0
    splice_t0 = time.perf_counter()
    for c in val:
        S = c["S"]; K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]; h = c["h40"].float()
        cur = c[sfield][S-1].float().flatten()
        aux = c[sfield][S-2].float().flatten()
        mf = make_dd_feat(embed, cur, toks[S], toks[S-1], S)
        mfn = (mf - xm_mean) / xm_std
        afn = (aux - xa_mean) / xa_std
        with torch.inference_mode():
            lg = lrdl_logits(mfn, afn)
        dd1_draft = int(lg.argmax())
        state = h[S-1].flatten(); prev = h[S-2].flatten()
        cur_tok = toks[S-1]; driver = toks[S]
        hits = []
        for j in range(K):
            t = S + j
            feat = make_feat(embed, state, state - prev,
                             driver, cur_tok, t)
            fn = (feat - a2_xm) / a2_xs
            pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
            new_state = state + pred
            if j == 0:
                draft = dd1_draft
            else:
                bf = make_bridge_feat(embed, new_state,
                                      driver, cur_tok, t)
                bfn = (bf - b_xm) / b_xs
                with torch.inference_mode():
                    z = bridge(bfn.unsqueeze(0))[0]
                draft = head_argmax(z)
            hits.append(draft == toks[t+1])
            cur_tok = driver; driver = draft
            prev = state; state = new_state
        dd1_hits += hits[0]
        m = prefix_metrics([int(x) for x in hits], K)
        rows.append({"category": c["category"], "m": m})
    splice_wall = (time.perf_counter() - splice_t0) * 1000

    stat_rows = [{"category": r["category"],
                  "cl_loop": {"E_A": r["m"]["E_A"],
                              "Sk": r["m"]["Sk"]}}
                 for r in rows]
    cs = canon_stats(stat_rows, "cl_loop", k=32)
    c2 = cs["S2"] / cs["S1"] if cs["S1"] > 0 else 0
    c3 = cs["S3"] / cs["S2"] if cs["S2"] > 0 else 0
    c4 = cs["S4"] / cs["S3"] if cs["S3"] > 0 else 0

    result = {
        "classification": "D3_LRDL0_EVAL",
        "rank": ck_rank,
        "checkpoint": str(a.ckpt.resolve()),
        "standalone": agg,
        "per_prompt": per_prompt,
        "dd1_splice_S1_count": f"{dd1_hits}/{len(val)}",
        "canonical_stats": cs,
        "conditional_survival": {
            "C2": round(c2, 4), "C3": round(c3, 4),
            "C4": round(c4, 4)},
        "splice_wall_ms_total": round(splice_wall, 1),
        "splice_vram_mb": round(
            torch.cuda.max_memory_allocated() / 1e6, 1),
        "mechanism_gate": {
            "S1_count": f"{dd1_hits}/{len(val)}",
            "gate": ">= 11", "passed": dd1_hits >= 11,
            "S4": cs["S4"], "S4_passed": cs["S4"] > 0},
    }
    a.output.write_text(json.dumps(result, indent=1))
    print(f"splice: mean={cs['mean']} S1={dd1_hits}/{len(val)}"
          f" S4={cs['S4']} deep={cs['deepest']} "
          f"C2={c2:.4f}")
    print(f"wrote {a.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
