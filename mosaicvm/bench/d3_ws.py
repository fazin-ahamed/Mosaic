"""WS-0 / WS-1 warm-start deconfound (roadmap S8-S24, S54).

Question: does the LRDL-0 8/21 regression come from the
low-rank interface itself, or from retraining the trunk?

WS-0 (zero training): frozen HIST-1 trunk + full canonical
normalization; logits = E_r (V_r^T zn) where W_eff =
W_head·diag(w_norm) ≈ E_r V_r^T. No training, no new weights.

WS-1 (q-head only): same frozen trunk + frozen E_r; train
ONLY Q: zn → q ∈ R^256, initialized Q = V_r^T (= WS-0).
Objective: full-vocab CE + 0.5 KL (no hinge). Frozen recipe:
AdamW 1e-3/wd 1e-4, cosine 600, CHUNK 512, clip 1.0, seed
104729.

Integrity (S16): the loaded trunk must reproduce the frozen
HIST-1 DEV telemetry (S1=10/21 and per-prompt margins from
d3_dd1_rank_telemetry.json) before any WS evaluation.

Outputs:
  artifacts/d3_ws/ws0_eval.json     (S17 telemetry)
  artifacts/d3_ws/ws1.pt            (Q + refs)
  artifacts/d3_ws/ws1_eval.json     (S18 telemetry + splice)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_dd1_hist import DDHistNet, D_STATE
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import (load_head, load_embed,
                                      make_bridge_feat)
from mosaicvm.bench.d3_v1 import (DeltaNet, make_feat,
                                  HORIZON)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d1b_tail_robustness import prefix_metrics
from mosaicvm.bench.d3_stats import stats as canon_stats
from mosaicvm.bench.d3_dd1_hist import build_hist_examples

HIST1 = Path("artifacts/d3_dd1_hist/dd1_hist_prev/dd1_hist.pt")
BASIS = Path("artifacts/d3_lrdl0/basis_r256.pt")
FROZEN_TELEM = Path("artifacts/d3_dd1_rank_telemetry.json")
OUTDIR = Path("artifacts/d3_ws")


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
    from mosaicvm.bench.d3_bridge import BridgeNet
    m = BridgeNet(ck["hidden"], ck["depth"])
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["x_mean"], ck["x_std"]


def eval_telemetry(logits_fn, val, embed, t0, tag):
    """S17 telemetry: S1, Top-k, NLL, margin, rank, per prompt."""
    per_prompt = {}
    wall_ms = []
    with torch.inference_mode():
        for c in val:
            S = c["S"]; toks = c["tokens"]
            cur = c["h40"][S - 1].float().flatten()
            aux = c["h40"][S - 2].float().flatten()
            mf = make_dd_feat(embed, cur, toks[S],
                              toks[S - 1], S)
            mfn = (mf - eval_telemetry.xm_mean) \
                / eval_telemetry.xm_std
            afn = (aux - eval_telemetry.xa_mean) \
                / eval_telemetry.xa_std
            tw = time.perf_counter()
            lg = logits_fn(mfn, afn)
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - tw) * 1000)
            true = toks[S + 1]
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
        "wall_ms_mean": round(sum(wall_ms) / len(wall_ms), 3),
    }
    dom = {}
    for cat, v in per_prompt.items():
        dom.setdefault(cat.rsplit("_", 1)[0], []).append(v)
    agg["per_domain"] = {
        d: {"S1": round(sum(v["hit"] for v in vs) / len(vs), 4)}
        for d, vs in sorted(dom.items())}
    hits = sorted(c for c, v in per_prompt.items() if v["hit"])
    print(f"[{tag}] S1={agg['S1']:.4f} Top4={agg['topk']['4']}"
          f" NLL={agg['mean_nll']} wall={agg['wall_ms_mean']}ms",
          flush=True)
    print(f"[{tag}] hits: {hits}", flush=True)
    return per_prompt, agg


def splice_eval(val, embed, first_logits_fn, a2, t_mean, t_std,
                a2_xm, a2_xs, bridge, b_xm, b_xs, w_head_g,
                w_norm_g, t0):
    rows = []; dd1_hits = 0
    @torch.inference_mode()
    def head_argmax(z):
        zg = z.cuda()
        zn = zg * torch.rsqrt(zg.square().mean() + 1e-6
                              ) * w_norm_g
        return int((w_head_g @ zn).argmax())
    for c in val:
        S = c["S"]; K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]; h = c["h40"].float()
        cur = c["h40"][S - 1].float().flatten()
        aux = c["h40"][S - 2].float().flatten()
        mf = make_dd_feat(embed, cur, toks[S], toks[S - 1], S)
        mfn = (mf - splice_eval.xm_mean) / splice_eval.xm_std
        afn = (aux - splice_eval.xa_mean) / splice_eval.xa_std
        with torch.inference_mode():
            lg = first_logits_fn(mfn, afn)
        dd1_draft = int(lg.argmax())
        state = h[S - 1].flatten(); prev = h[S - 2].flatten()
        cur_tok = toks[S - 1]; driver = toks[S]
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
            hits.append(draft == toks[t + 1])
            cur_tok = driver; driver = draft
            prev = state; state = new_state
        dd1_hits += hits[0]
        m = prefix_metrics([int(x) for x in hits], K)
        rows.append({"category": c["category"], "m": m})
    stat_rows = [{"category": r["category"],
                  "cl_loop": {"E_A": r["m"]["E_A"],
                              "Sk": r["m"]["Sk"]}}
                 for r in rows]
    cs = canon_stats(stat_rows, "cl_loop", k=32)
    c2 = cs["S2"] / cs["S1"] if cs["S1"] > 0 else 0
    return dd1_hits, cs, {"C2": round(c2, 4),
                          "C3": round(cs["S3"] / cs["S2"], 4)
                          if cs["S2"] > 0 else 0,
                          "C4": round(cs["S4"] / cs["S3"], 4)
                          if cs["S3"] > 0 else 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("all", "ws0", "ws1"),
                   default="all")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- frozen HIST-1 trunk ----
    ck = torch.load(HIST1, map_location="cpu",
                    weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"], ck["aux_dim"])
    hist.load_state_dict(ck["model"])   # strict
    hist.eval()
    for p_ in hist.parameters():
        p_.requires_grad_(False)
    xm_mean, xm_std = ck["xm_mean"], ck["xm_std"]
    xa_mean, xa_std = ck["xa_mean"], ck["xa_std"]

    # ---- frozen basis ----
    bk = torch.load(BASIS, map_location="cpu",
                    weights_only=True)
    E_r = bk["E_r"]          # [129280, 256]
    V_r = bk["V_r"]          # [4096, 256]
    assert bk["rank"] == 256
    E_g = E_r.cuda()
    V_g = V_r.cuda()

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

    # stash normalization for helpers
    eval_telemetry.xm_mean = xm_mean
    eval_telemetry.xm_std = xm_std
    eval_telemetry.xa_mean = xa_mean
    eval_telemetry.xa_std = xa_std
    splice_eval.xm_mean = xm_mean
    splice_eval.xm_std = xm_std
    splice_eval.xa_mean = xa_mean
    splice_eval.xa_std = xa_std

    # z_rms: RMS-normalized trunk output ONLY.
    # w_norm is NOT applied here: the frozen basis factorizes
    # W_eff = W_head*diag(w_norm) = E_r @ V_r^T, so the
    # low-rank path consumes z_rms directly. w_norm appears
    # only in the canonical full-head comparator.
    @torch.inference_mode()
    def z_rms_of(mfn, afn):
        z = hist(mfn.unsqueeze(0), afn.unsqueeze(0))
        zg = z.cuda()
        return zg * torch.rsqrt(
            zg.square().mean(-1, keepdim=True) + 1e-6
        )                                   # [1, 4096]

    # ---- S16 integrity: canonical full-head path must
    # reproduce the frozen HIST-1 telemetry ----
    frozen = json.loads(FROZEN_TELEM.read_text())["HIST-1"]
    ref = frozen["per_prompt"]

    @torch.inference_mode()
    def canonical_logits(mfn, afn):
        zr = z_rms_of(mfn, afn)[0]          # [4096]
        # form A: W_head @ (w_norm * z_rms)  (eval-harness path)
        lgA = w_head_g @ (zr * w_norm_g)
        # form B: (W_head * w_norm) @ z_rms  (factorized path)
        lgB = (w_head_g * w_norm_g.unsqueeze(0)) @ zr
        # explicit cross-check: any normalization drift in
        # either path surfaces here
        assert torch.allclose(lgA, lgB, atol=1e-3), \
            "canonical-path disagreement"
        return lgA

    integ_hits = []
    max_md = 0.0
    with torch.inference_mode():
        for c in val:
            S = c["S"]; toks = c["tokens"]
            cur = c["h40"][S - 1].float().flatten()
            aux = c["h40"][S - 2].float().flatten()
            mf = make_dd_feat(embed, cur, toks[S],
                              toks[S - 1], S)
            mfn = (mf - xm_mean) / xm_std
            afn = (aux - xa_mean) / xa_std
            lg = canonical_logits(mfn, afn)
            true = toks[S + 1]
            hit = int(lg.argmax()) == true
            integ_hits.append((c["category"], hit))
            tl = float(lg[true])
            lgm = lg.clone(); lgm[true] = float("-inf")
            m = tl - float(lg[lgm.argmax()])
            max_md = max(max_md,
                         abs(m - ref[c["category"]]["margin"]))
    s1_integ = sum(h for _, h in integ_hits)
    integ_pass = (s1_integ == 10
                  and all(h == ref[c]["hit"]
                          for c, h in integ_hits)
                  and max_md < 5e-4)
    print(f"integrity: S1={s1_integ}/21 "
          f"max|Δmargin|={max_md:.2e} "
          f"{'PASS' if integ_pass else 'FAIL'}", flush=True)
    if not integ_pass:
        raise RuntimeError(
            "frozen trunk does not reproduce HIST-1 telemetry")

    # ---- WS-0: zero-training factorization ----
    @torch.inference_mode()
    def ws0_logits(mfn, afn):
        zr = z_rms_of(mfn, afn)             # [1, 4096]
        q = zr @ V_g                        # [1, 256]
        return (q @ E_g.t())[0]

    pp0, agg0 = eval_telemetry(ws0_logits, val, embed, t0,
                               "WS-0")
    json_13_14 = {c: pp0[c] for c in
                  ("json_13", "json_14")}
    print(f"[WS-0] json probes: {json_13_14}", flush=True)
    (OUTDIR / "ws0_eval.json").write_text(json.dumps(
        {"classification": "D3_WS0",
         "integrity": {"S1": s1_integ,
                       "max_margin_delta": max_md,
                       "pass": integ_pass},
         "aggregate": agg0, "per_prompt": pp0,
         "hist1_ckpt_sha256": hashlib.sha256(
             HIST1.read_bytes()).hexdigest(),
         "basis_sha256": hashlib.sha256(
             BASIS.read_bytes()).hexdigest()},
        indent=1))
    if a.stage == "ws0":
        return

    # ---- WS-1: train only Q (init = V_r^T) on cached zn ----
    xm, xa, labels, _, is_boundary = build_hist_examples(
        "h40", "prev")
    n = xm.shape[0]
    xmn = (xm - xm_mean) / xm_std
    xan = (xa - xa_mean) / xa_std
    y_idx = torch.tensor([m["y"] for m in labels])
    top_idx_t = torch.tensor([m["top_idx"] for m in labels])
    p_t = torch.stack([
        F.softmax(torch.tensor(m["top_val"]), dim=0)
        for m in labels])
    y_g = y_idx.cuda(); top_g = top_idx_t.cuda()
    pt_g = p_t.cuda()

    # cache zn for ALL training rows (trunk frozen)
    print(f"caching z_rms for {n} rows...", flush=True)
    zr_cache = torch.empty(n, 4096, device="cuda")
    with torch.inference_mode():
        for i in range(0, n, 512):
            sl = slice(i, min(i + 512, n))
            z = hist(xmn[sl], xan[sl])
            zg = z.cuda()
            zr_cache[sl] = zg * torch.rsqrt(
                zg.square().mean(-1, keepdim=True) + 1e-6)
    print(f"z_rms cached ({time.time()-t0:.0f}s)", flush=True)

    Q = nn.Parameter(V_r.t().clone().cuda())  # [256, 4096]
    P0 = V_r.t().cuda()
    opt = torch.optim.AdamW([Q], lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=600, eta_min=1e-3 * 0.02)
    LAM_KL = 0.5
    CHUNK = 512
    q_drift = []

    for step in range(1, 601):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            q = zr_cache[sl] @ Q.t()          # [B, 256]
            lg = q @ E_g.t()                  # [B, V]
            ce = F.cross_entropy(lg.float(), y_g[sl],
                                 reduction="none")
            l16 = lg.gather(1, top_g[sl])
            logp = F.log_softmax(l16, dim=-1)
            kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log()
                              - logp)).sum(-1)
            part = (ce + LAM_KL * kl).mean() \
                * (q.shape[0] / n)
            part.backward()
            total += part.item()
        torch.nn.utils.clip_grad_norm_([Q], 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 600:
            d = float((Q - P0).norm())
            q_drift.append({"step": step,
                            "Q_minus_P256_fro": round(d, 4)})
            with torch.no_grad():
                accs = []
                for i in range(0, n, 2048):
                    sl = slice(i, min(i + 2048, n))
                    lg = (zr_cache[sl] @ Q.t()) @ E_g.t()
                    accs.append((lg.argmax(-1) == y_g[sl])
                                .float().mean().item())
            print(f"step {step}: L {total:.4f} train-argmax "
                  f"{sum(accs)/len(accs):.3f} "
                  f"||Q-P||={d:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- WS-1 evaluation ----
    Q_d = Q.detach()

    @torch.inference_mode()
    def ws1_logits(mfn, afn):
        zr = z_rms_of(mfn, afn)
        return ((zr @ Q_d.t()) @ E_g.t())[0]

    pp1, agg1 = eval_telemetry(ws1_logits, val, embed, t0,
                               "WS-1")
    probes = {c: {"ws0": pp0[c], "ws1": pp1[c]} for c in
              ("json_13", "json_14")}
    print(f"[WS-1] json probes: "
          f"{json.dumps(probes, indent=1)}", flush=True)

    # ---- splice (mandatory if S1 >= 10; run always for
    # telemetry) ----
    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(
        Path("artifacts/d3_v1_A2/student.pt"))
    bridge, b_xm, b_xs = load_bridge(
        Path("artifacts/d3_bridge_r3b/bridge_ps/bridge.pt"))
    dd1_hits, cs, cond = splice_eval(
        val, embed, ws1_logits, a2, t_mean, t_std, a2_xm,
        a2_xs, bridge, b_xm, b_xs, w_head_g, w_norm_g, t0)
    print(f"[WS-1] splice: mean={cs['mean']} "
          f"S1={dd1_hits}/{len(val)} S4={cs['S4']} "
          f"deep={cs['deepest']}", flush=True)

    result = {
        "classification": "D3_WS1",
        "aggregate": agg1, "per_prompt": pp1,
        "q_drift": q_drift,
        "json_probes": probes,
        "dd1_splice_S1_count": f"{dd1_hits}/{len(val)}",
        "canonical_stats": cs,
        "conditional_survival": cond,
        "gates": {
            "recover_baseline_S1_ge_10":
                agg1["S1"] >= 10 / 21,
            "genuine_signal_S1_ge_11":
                agg1["S1"] >= 11 / 21,
            "regression_S1_le_9": agg1["S1"] <= 9 / 21},
        "hist1_ckpt_sha256": hashlib.sha256(
            HIST1.read_bytes()).hexdigest(),
        "basis_sha256": hashlib.sha256(
            BASIS.read_bytes()).hexdigest(),
    }
    torch.save({"Q": Q.detach().cpu(),
                "hist1_ckpt": str(HIST1),
                "basis": str(BASIS), "rank": 256,
                "lam_kl": LAM_KL, "seed": 104729,
                "hist1_ckpt_sha256": result["hist1_ckpt_sha256"],
                "basis_sha256": result["basis_sha256"]},
               OUTDIR / "ws1.pt")
    (OUTDIR / "ws1_eval.json").write_text(
        json.dumps(result, indent=1))
    print(f"wrote {OUTDIR}/ws1.pt, ws1_eval.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
