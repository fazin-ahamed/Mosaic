"""LRDL-1A: row-scale vocabulary adapter on frozen WS-1
(roadmap S11-S20, S60 steps 3-9).

Frozen: HIST-1 trunk, trained Q256, E256 base, corpus, split,
A2, ps, evaluator, d3_stats. Trained: ONLY a ∈ R^V (129,280)
with vocabulary row scales

    s_i = exp(alpha * tanh(a_i)),  alpha = 0.25,  a = 0 init

so E' = diag(s) E equals E EXACTLY at step 0 (zero-init fail-
closed equivalence check vs WS-1). Objective:

    L = CE_full + 0.5 KL + lam_s * (1/V) sum(log s_i)^2

lam_s derived from a training-only smoke (never DEV).

Because trunk/Q/E are frozen, base logits are constant:
    base = (z_rms @ Q^T) @ E_r^T   [2688, V]
    logits = base * s              (row-scale broadcast)
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

from mosaicvm.bench.d3_dd1_hist import (DDHistNet,
                                        build_hist_examples)
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import load_head, load_embed
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON
from mosaicvm.bench.d3_ws import (eval_telemetry, splice_eval,
                                  load_a2, load_bridge)

HIST1 = Path("artifacts/d3_dd1_hist/dd1_hist_prev/dd1_hist.pt")
BASIS = Path("artifacts/d3_lrdl0/basis_r256.pt")
WS1 = Path("artifacts/d3_ws/ws1.pt")
WS1_EVAL = Path("artifacts/d3_ws/ws1_eval.json")
OUTDIR = Path("artifacts/d3_lrdl1a")
ALPHA = 0.25
V = 129280


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("all", "smoke", "train"),
                   default="all")
    p.add_argument("--lam-s", type=float, default=None,
                   help="override smoke-derived lam_s")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- frozen stack ----
    ck = torch.load(HIST1, map_location="cpu",
                    weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"], ck["aux_dim"])
    hist.load_state_dict(ck["model"])
    hist.eval()
    for p_ in hist.parameters():
        p_.requires_grad_(False)
    xm_mean, xm_std = ck["xm_mean"], ck["xm_std"]
    xa_mean, xa_std = ck["xa_mean"], ck["xa_std"]

    ws1 = torch.load(WS1, map_location="cpu",
                     weights_only=True)
    assert ws1["hist1_ckpt_sha256"] == sha(HIST1)
    assert ws1["basis_sha256"] == sha(BASIS)
    Q = ws1["Q"].cuda()                      # [256, 4096]

    bk = torch.load(BASIS, map_location="cpu",
                    weights_only=True)
    assert bk["rank"] == 256
    E_r = bk["E_r"]                          # [V, 256]
    E_g = E_r.cuda()

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

    # stash normalization for shared eval helpers
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

    @torch.inference_mode()
    def ws1_logits(mfn, afn):
        """Unmodified WS-1 reference path."""
        zr = z_rms_of(mfn, afn)
        return ((zr @ Q.t()) @ E_g.t())[0]

    # ---- step-0 equivalence (fail closed, S60.4) ----
    a_vec = nn.Parameter(torch.zeros(V, device="cuda"))
    s = torch.exp(ALPHA * torch.tanh(a_vec))    # = 1 exactly

    @torch.inference_mode()
    def lrdl1a_logits(mfn, afn, s_vec):
        zr = z_rms_of(mfn, afn)
        return (((zr @ Q.t()) @ E_g.t())[0]
                * s_vec)

    # bitwise: s=1.0 multiply is exact
    c0 = val[0]; S0 = c0["S"]
    mf0 = make_dd_feat(embed,
                       c0["h40"][S0 - 1].float().flatten(),
                       c0["tokens"][S0], c0["tokens"][S0 - 1],
                       S0)
    mfn0 = (mf0 - xm_mean) / xm_std
    afn0 = ((c0["h40"][S0 - 2].float().flatten() - xa_mean)
            / xa_std)
    with torch.inference_mode():
        l_ref = ws1_logits(mfn0, afn0)
        l_new = lrdl1a_logits(mfn0, afn0, s.detach())
    bitwise0 = torch.equal(l_ref, l_new)

    # behavioral: full DEV S1 with s=1 must reproduce WS-1
    ws1_ev = json.loads(WS1_EVAL.read_text())
    ref_pp = ws1_ev["per_prompt"]
    hits0 = 0; max_md = 0.0; all_hits = True
    with torch.inference_mode():
        for c in val:
            S = c["S"]; toks = c["tokens"]
            mf = make_dd_feat(embed,
                              c["h40"][S - 1].float().flatten(),
                              toks[S], toks[S - 1], S)
            mfn = (mf - xm_mean) / xm_std
            afn = ((c["h40"][S - 2].float().flatten()
                    - xa_mean) / xa_std)
            lg = lrdl1a_logits(mfn, afn, s.detach())
            true = toks[S + 1]
            hit = int(lg.argmax()) == true
            hits0 += hit
            if hit != ref_pp[c["category"]]["hit"]:
                all_hits = False
            tl = float(lg[true])
            lgm = lg.clone(); lgm[true] = float("-inf")
            m = tl - float(lg[lgm.argmax()])
            max_md = max(max_md, abs(
                m - ref_pp[c["category"]]["margin"]))
    step0_pass = (bitwise0 and hits0 == 10 and all_hits
                  and max_md < 1e-4)
    print(f"step-0: bitwise={bitwise0} S1={hits0}/21 "
          f"max|Δm|={max_md:.2e} "
          f"{'PASS' if step0_pass else 'FAIL'}", flush=True)
    if not step0_pass:
        raise RuntimeError("step-0 equivalence to WS-1 failed")

    # ---- constant base logits on training rows ----
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

    print(f"caching base logits [{n}, {V}]...",
          flush=True)
    base = torch.empty(n, V, device="cuda")
    with torch.inference_mode():
        for i in range(0, n, 256):
            sl = slice(i, min(i + 256, n))
            z = hist(xmn[sl], xan[sl]).cuda()
            zr = z * torch.rsqrt(
                z.square().mean(-1, keepdim=True) + 1e-6)
            base[sl] = (zr @ Q.t()) @ E_g.t()
    print(f"base cached ({time.time()-t0:.0f}s)", flush=True)

    LAM_KL = 0.5
    CHUNK = 512

    def make_trainable():
        av = nn.Parameter(torch.zeros(V, device="cuda"))
        return av

    def loss_fn(av, sl, lam_s):
        s_v = torch.exp(ALPHA * torch.tanh(av))
        lg = base[sl] * s_v.unsqueeze(0)
        ce = F.cross_entropy(lg.float(), y_g[sl],
                             reduction="none")
        l16 = lg.gather(1, top_g[sl])
        logp = F.log_softmax(l16, dim=-1)
        kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log()
                          - logp)).sum(-1)
        row = ce + LAM_KL * kl
        scale = (ALPHA * torch.tanh(av)).square().mean()
        return row.mean() * (sl.stop - sl.start) / n, \
            ce, kl, scale

    if a.mode == "smoke" or not (OUTDIR
                                 / "smoke.json").exists():
        # ---- S60.5 training-only scale smoke ----
        print("\n=== SCALE SMOKE (training rows only) ===",
              flush=True)
        av = make_trainable()
        opt = torch.optim.AdamW([av], lr=1e-3,
                                weight_decay=1e-4)
        # 50 CE+KL-only steps to move s off 1.0
        for step in range(50):
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                row = loss_fn(av, sl, 0.0)
                part = row[0]
                part.backward()
                tot += part.item()
            torch.nn.utils.clip_grad_norm_([av], 1.0)
            opt.step()
        with torch.no_grad():
            log_s = ALPHA * torch.tanh(av)
            abs_ls = log_s.abs()
            stats = {
                "smoke_steps": 50,
                "mean_ce": None, "mean_kl": None,
                "max_log_s": round(float(abs_ls.max()), 5),
                "mean_log_s": round(float(abs_ls.mean()), 6),
                "p95_log_s": round(float(abs_ls.quantile(
                    0.95)), 5),
                "active_rows_gt_0.01": int(
                    (abs_ls > 0.01).sum().item()),
                "finite_s": bool(torch.isfinite(
                    torch.exp(log_s)).all().item()),
            }
        # gradient norms at the perturbed state
        def grad_norm(term_fn):
            av.grad = None
            tot = 0.0
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                part = term_fn(av, sl)
                part.backward()
            g = av.grad
            return float(g.square().sum().item()) ** 0.5

        def ce_term(av_, sl):
            row, ce, kl, sc = loss_fn(av_, sl, 0.0)
            return ce.mean() * (sl.stop - sl.start) / n

        def scale_term(av_, sl):
            row, ce, kl, sc = loss_fn(av_, sl, 0.0)
            return sc

        gn_ce = grad_norm(ce_term)
        gn_scale = grad_norm(scale_term)
        # magnitudes at perturbed state
        with torch.no_grad():
            mce = 0.0; mkl = 0.0; msc = 0.0
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                s_v = torch.exp(ALPHA * torch.tanh(av))
                lg = base[sl] * s_v.unsqueeze(0)
                ce = F.cross_entropy(lg.float(), y_g[sl],
                                     reduction="none")
                l16 = lg.gather(1, top_g[sl])
                logp = F.log_softmax(l16, dim=-1)
                kl = (pt_g[sl] * (pt_g[sl].clamp_min(
                    1e-9).log() - logp)).sum(-1)
                mce += ce.sum().item(); mkl += kl.sum().item()
            msc = float((ALPHA * torch.tanh(av)
                         ).square().mean().item())
        stats.update({
            "mean_ce": round(mce / n, 4),
            "mean_kl": round(mkl / n, 4),
            "scale_penalty_at_smoke": msc,
            "grad_norm_ce": round(gn_ce, 4),
            "grad_norm_scale_lam1": round(gn_scale, 6),
        })
        lam_derived = None
        if gn_scale > 0:
            lam_derived = round(
                0.1 * gn_ce / gn_scale, 6)
        stats["lam_s_derived_ratio0.1"] = lam_derived
        (OUTDIR / "smoke.json").write_text(
            json.dumps(stats, indent=1))
        print(json.dumps(stats, indent=1), flush=True)
        del av, opt
        if a.mode == "smoke":
            return
    else:
        stats = json.loads(
            (OUTDIR / "smoke.json").read_text())
        print("smoke exists:", json.dumps(stats), flush=True)

    lam_s = a.lam_s if a.lam_s is not None else \
        stats["lam_s_derived_ratio0.1"]
    print(f"\nlam_s = {lam_s}", flush=True)

    # ---- train (fresh a = 0, frozen recipe) ----
    av = make_trainable()
    opt = torch.optim.AdamW([av], lr=1e-3,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=600, eta_min=1e-3 * 0.02)
    telem = []
    for step in range(1, 601):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            row, ce, kl, sc = loss_fn(av, sl, lam_s)
            part = row + lam_s * sc * (sl.stop
                                       - sl.start) / n
            part.backward()
            total += part.item()
        torch.nn.utils.clip_grad_norm_([av], 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 600:
            with torch.no_grad():
                s_v = torch.exp(ALPHA * torch.tanh(av))
                abs_ls = (ALPHA * torch.tanh(av)).abs()
                accs = []
                for i in range(0, n, 2048):
                    sl = slice(i, min(i + 2048, n))
                    lg = base[sl] * s_v.unsqueeze(0)
                    accs.append((lg.argmax(-1) == y_g[sl])
                                .float().mean().item())
                telem.append({
                    "step": step, "loss": round(total, 4),
                    "train_argmax": round(
                        sum(accs) / len(accs), 4),
                    "max_log_s": round(
                        float(abs_ls.max()), 5),
                    "mean_log_s": round(
                        float(abs_ls.mean()), 6),
                    "p95_log_s": round(float(
                        abs_ls.quantile(0.95)), 5),
                    "active_rows": int(
                        (abs_ls > 0.01).sum().item())})
            print(f"step {step}: L {total:.4f} argmax "
                  f"{telem[-1]['train_argmax']} "
                  f"max|ls|={telem[-1]['max_log_s']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- evaluate: standalone + splice ----
    s_final = torch.exp(ALPHA * torch.tanh(
        av.detach())).cpu()

    @torch.inference_mode()
    def l1a_logits(mfn, afn):
        return lrdl1a_logits(mfn, afn, s_final.cuda())

    pp, agg = eval_telemetry(l1a_logits, val, embed, t0,
                             "LRDL-1A")
    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(
        Path("artifacts/d3_v1_A2/student.pt"))
    bridge, b_xm, b_xs = load_bridge(
        Path("artifacts/d3_bridge_r3b/bridge_ps/bridge.pt"))
    if agg["S1"] >= 10 / 21:
        dd1_hits, cs, cond = splice_eval(
            val, embed, l1a_logits, a2, t_mean, t_std,
            a2_xm, a2_xs, bridge, b_xm, b_xs, w_head_g,
            w_norm_g, t0)
        print(f"[LRDL-1A] splice: mean={cs['mean']} "
              f"S1={dd1_hits}/{len(val)} S4={cs['S4']} "
              f"deep={cs['deepest']}", flush=True)
    else:
        dd1_hits, cs, cond = 0, {}, {}
        print("[LRDL-1A] S1 < 10/21 — splice skipped",
              flush=True)

    result = {
        "classification": "D3_LRDL1A",
        "step0_equivalence": {
            "bitwise": bitwise0, "S1": f"{hits0}/21",
            "max_margin_delta": max_md,
            "pass": step0_pass},
        "lam_s": lam_s, "alpha": ALPHA,
        "smoke": stats, "train_telemetry": telem,
        "aggregate": agg, "per_prompt": pp,
        "dd1_splice_S1_count":
            f"{dd1_hits}/{len(val)}" if cs else None,
        "canonical_stats": cs,
        "conditional_survival": cond,
        "gates": {
            "promote_S1_ge_11": agg["S1"] >= 11 / 21,
            "plateau_S1_eq_10": agg["S1"] == 10 / 21,
            "regress_S1_le_9": agg["S1"] <= 9 / 21},
        "folded_E_sha_note":
            "E_final = diag(s) @ E_r; zero runtime overhead",
        "hist1_ckpt_sha256": sha(HIST1),
        "basis_sha256": sha(BASIS),
        "ws1_sha256": sha(WS1),
    }
    torch.save({"a": av.detach().cpu(), "s": s_final,
                "rank": 256, "alpha": ALPHA,
                "lam_s": lam_s,
                "hist1_ckpt_sha256": sha(HIST1),
                "basis_sha256": sha(BASIS),
                "ws1_sha256": sha(WS1)},
               OUTDIR / "lrdl1a.pt")
    (OUTDIR / "lrdl1a_eval.json").write_text(
        json.dumps(result, indent=1))
    print(f"\ngates: {json.dumps(result['gates'])}")
    print(f"wrote {OUTDIR}/lrdl1a.pt, lrdl1a_eval.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
