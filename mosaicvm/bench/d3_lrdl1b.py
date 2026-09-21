"""LRDL-1B: rank-4 residual vocabulary adapter on frozen WS-1
(roadmap S24-S28, authorized after LRDL-1A plateau).

E' = E_r + ΔE,  ΔE = A @ B,  A ∈ R^{V×4}, B ∈ R^{4×256}
LoRA init: A = 0, B ~ N(0, 0.02²) → ΔE = 0 exactly at step 0
(fail-closed equivalence to WS-1). Norm cap by projection:

    ||ΔE||_F / ||E_r||_F ≤ 0.10   (persisted each 50 steps)

Objective: CE_full + 0.5 KL + lam_Δ ||ΔE||²_F/||E||²_F,
lam_Δ from a training-only smoke (0.1 gradient-ratio rule).

Frozen: HIST-1 trunk, Q256, E_r, corpus, split, A2, ps,
evaluator, d3_stats. Trainable: A + B (~518k params).

Constant-base trick: q_cache = z_rms @ Q^T is constant, so
    logits = q_cache @ (E_r + AB)^T
           = base + (q_cache @ B^T) @ A^T
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
OUTDIR = Path("artifacts/d3_lrdl1b")
K_RES = 4
NORM_CAP = 0.10
V = 129280


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("all", "smoke", "train"),
                   default="all")
    p.add_argument("--lam-d", type=float, default=None)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- frozen stack (identical to LRDL-1A) ----
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
    Q = ws1["Q"].cuda()

    bk = torch.load(BASIS, map_location="cpu",
                    weights_only=True)
    assert bk["rank"] == 256
    E_r = bk["E_r"]
    E_g = E_r.cuda()
    E_fro = float(E_g.norm().item())

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

    # ---- step-0 equivalence (A=0 → ΔE=0) ----
    A = nn.Parameter(torch.zeros(V, K_RES, device="cuda"))
    Bp = nn.Parameter(
        torch.randn(K_RES, 256, device="cuda") * 0.02)

    @torch.inference_mode()
    def l1b_logits(mfn, afn, A_v, B_v):
        zr = z_rms_of(mfn, afn)
        lg = (zr @ Q.t()) @ E_g.t() \
            + ((zr @ Q.t()) @ B_v.t()) @ A_v.t()
        return lg[0]

    ws1_ev = json.loads(WS1_EVAL.read_text())
    ref_pp = ws1_ev["per_prompt"]
    c0 = val[0]; S0 = c0["S"]
    mf0 = make_dd_feat(embed,
                       c0["h40"][S0 - 1].float().flatten(),
                       c0["tokens"][S0], c0["tokens"][S0 - 1],
                       S0)
    mfn0 = (mf0 - xm_mean) / xm_std
    afn0 = ((c0["h40"][S0 - 2].float().flatten() - xa_mean)
            / xa_std)
    with torch.inference_mode():
        zr0 = z_rms_of(mfn0, afn0)
        l_ref = ((zr0 @ Q.t()) @ E_g.t())[0]
        l_new = l1b_logits(mfn0, afn0, A.detach(),
                           Bp.detach())
    bitwise0 = torch.equal(l_ref, l_new)

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
            lg = l1b_logits(mfn, afn, A.detach(),
                            Bp.detach())
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

    # ---- training data + constant caches ----
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

    print(f"caching q_cache/base [{n} rows]...", flush=True)
    q_cache = torch.empty(n, 256, device="cuda")
    base = torch.empty(n, V, device="cuda")
    with torch.inference_mode():
        for i in range(0, n, 256):
            sl = slice(i, min(i + 256, n))
            z = hist(xmn[sl], xan[sl]).cuda()
            zr = z * torch.rsqrt(
                z.square().mean(-1, keepdim=True) + 1e-6)
            q_cache[sl] = zr @ Q.t()
            base[sl] = q_cache[sl] @ E_g.t()
    print(f"caches ready ({time.time()-t0:.0f}s)", flush=True)

    LAM_KL = 0.5
    CHUNK = 512

    def loss_terms(A_v, B_v, sl):
        qB = q_cache[sl] @ B_v.t()          # [B, 4]
        lg = base[sl] + qB @ A_v.t()        # [B, V]
        ce = F.cross_entropy(lg.float(), y_g[sl],
                             reduction="none")
        l16 = lg.gather(1, top_g[sl])
        logp = F.log_softmax(l16, dim=-1)
        kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log()
                          - logp)).sum(-1)
        dE = A_v @ B_v
        pen = dE.square().sum() / (E_fro ** 2)
        return ce + LAM_KL * kl, ce, kl, pen

    def ratio_now(A_v, B_v):
        with torch.no_grad():
            dE = A_v @ B_v
            return float(dE.norm().item()) / E_fro

    if a.mode == "smoke" or not (OUTDIR
                                 / "smoke.json").exists():
        print("\n=== RESIDUAL SMOKE (training rows) ===",
              flush=True)
        As = nn.Parameter(torch.zeros(V, K_RES,
                                      device="cuda"))
        Bs = nn.Parameter(torch.randn(
            K_RES, 256, device="cuda") * 0.02)
        opt = torch.optim.AdamW([As, Bs], lr=1e-3,
                                weight_decay=1e-4)
        for step in range(50):
            opt.zero_grad(set_to_none=True)
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                row, ce, kl, pen = loss_terms(As, Bs, sl)
                part = row.mean() * (sl.stop - sl.start) / n
                part.backward()
            torch.nn.utils.clip_grad_norm_([As, Bs], 1.0)
            opt.step()
            r = ratio_now(As, Bs)
            if r > NORM_CAP:   # projection
                with torch.no_grad():
                    scale = (NORM_CAP * E_fro
                             / (r * E_fro))
                    As.mul_(scale)
        stats = {"smoke_steps": 50,
                 "ratio_after_smoke": round(r, 5)}
        with torch.no_grad():
            mce = 0.0; mkl = 0.0
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                row, ce, kl, pen = loss_terms(As, Bs, sl)
                mce += ce.sum().item()
                mkl += kl.sum().item()
        stats["mean_ce"] = round(mce / n, 4)
        stats["mean_kl"] = round(mkl / n, 4)

        def grad_norm(term):
            As.grad = None; Bs.grad = None
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                row, ce, kl, pen = loss_terms(As, Bs, sl)
                term(ce, kl, pen).backward()
            g2 = 0.0
            for g in (As.grad, Bs.grad):
                if g is not None:
                    g2 += g.square().sum().item()
            return g2 ** 0.5

        gn_ce = grad_norm(lambda ce, kl, pen: ce.mean()
                          * 1.0)
        gn_pen = grad_norm(lambda ce, kl, pen: pen)
        stats["grad_norm_ce"] = round(gn_ce, 4)
        stats["grad_norm_penalty_lam1"] = round(gn_pen, 6)
        lam_d = round(0.1 * gn_ce / gn_pen, 6) \
            if gn_pen > 0 else None
        stats["lam_d_derived_ratio0.1"] = lam_d
        (OUTDIR / "smoke.json").write_text(
            json.dumps(stats, indent=1))
        print(json.dumps(stats, indent=1), flush=True)
        del As, Bs, opt
        if a.mode == "smoke":
            return
    else:
        stats = json.loads(
            (OUTDIR / "smoke.json").read_text())
        print("smoke exists:", json.dumps(stats), flush=True)

    lam_d = a.lam_d if a.lam_d is not None else \
        stats["lam_d_derived_ratio0.1"]
    print(f"\nlam_d = {lam_d}", flush=True)

    # ---- train: fresh zero-init A, seeded B ----
    A = nn.Parameter(torch.zeros(V, K_RES, device="cuda"))
    Bp = nn.Parameter(torch.randn(
        K_RES, 256, device="cuda") * 0.02)
    opt = torch.optim.AdamW([A, Bp], lr=1e-3,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=600, eta_min=1e-3 * 0.02)
    telem = []
    for step in range(1, 601):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            row, ce, kl, pen = loss_terms(A, Bp, sl)
            w = (sl.stop - sl.start) / n
            part = (row.mean() + lam_d * pen) * w
            part.backward()
            total += part.item()
        torch.nn.utils.clip_grad_norm_([A, Bp], 1.0)
        opt.step()
        r = ratio_now(A, Bp)
        if r > NORM_CAP:
            with torch.no_grad():
                A.mul_(NORM_CAP / r)
            r = NORM_CAP
        sched.step()
        if step % 50 == 0 or step == 600:
            with torch.no_grad():
                accs = []
                for i in range(0, n, 2048):
                    sl = slice(i, min(i + 2048, n))
                    qB = q_cache[sl] @ Bp.t()
                    lg = base[sl] + qB @ A.t()
                    accs.append((lg.argmax(-1) == y_g[sl])
                                .float().mean().item())
            telem.append({"step": step,
                          "loss": round(total, 4),
                          "train_argmax": round(
                              sum(accs) / len(accs), 4),
                          "dE_ratio": round(r, 5)})
            print(f"step {step}: L {total:.4f} argmax "
                  f"{telem[-1]['train_argmax']} "
                  f"||dE||/||E||={r:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- evaluate ----
    A_d = A.detach(); B_d = Bp.detach()

    @torch.inference_mode()
    def l1b_final(mfn, afn):
        return l1b_logits(mfn, afn, A_d, B_d)

    pp, agg = eval_telemetry(l1b_final, val, embed, t0,
                             "LRDL-1B")
    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(
        Path("artifacts/d3_v1_A2/student.pt"))
    bridge, b_xm, b_xs = load_bridge(
        Path("artifacts/d3_bridge_r3b/bridge_ps/bridge.pt"))
    if agg["S1"] >= 10 / 21:
        dd1_hits, cs, cond = splice_eval(
            val, embed, l1b_final, a2, t_mean, t_std,
            a2_xm, a2_xs, bridge, b_xm, b_xs, w_head_g,
            w_norm_g, t0)
        print(f"[LRDL-1B] splice: mean={cs['mean']} "
              f"S1={dd1_hits}/{len(val)} S4={cs['S4']} "
              f"deep={cs['deepest']}", flush=True)
    else:
        dd1_hits, cs, cond = 0, {}, {}
        print("[LRDL-1B] S1 < 10/21 — splice skipped",
              flush=True)

    result = {
        "classification": "D3_LRDL1B",
        "step0_equivalence": {
            "bitwise": bitwise0, "S1": f"{hits0}/21",
            "max_margin_delta": max_md,
            "pass": step0_pass},
        "k_residual": K_RES, "norm_cap": NORM_CAP,
        "final_dE_ratio": round(ratio_now(A_d, B_d), 5),
        "lam_d": lam_d, "smoke": stats,
        "train_telemetry": telem,
        "aggregate": agg, "per_prompt": pp,
        "dd1_splice_S1_count":
            f"{dd1_hits}/{len(val)}" if cs else None,
        "canonical_stats": cs,
        "conditional_survival": cond,
        "gates": {
            "promote_S1_ge_11": agg["S1"] >= 11 / 21,
            "plateau_S1_eq_10": agg["S1"] == 10 / 21,
            "regress_S1_le_9": agg["S1"] <= 9 / 21},
        "hist1_ckpt_sha256": sha(HIST1),
        "basis_sha256": sha(BASIS),
        "ws1_sha256": sha(WS1),
    }
    torch.save({"A": A_d.cpu(), "B": B_d.cpu(),
                "rank": 256, "k": K_RES,
                "norm_cap": NORM_CAP, "lam_d": lam_d,
                "hist1_ckpt_sha256": sha(HIST1),
                "basis_sha256": sha(BASIS),
                "ws1_sha256": sha(WS1)},
               OUTDIR / "lrdl1b.pt")
    (OUTDIR / "lrdl1b_eval.json").write_text(
        json.dumps(result, indent=1))
    print(f"\ngates: {json.dumps(result['gates'])}")
    print(f"wrote {OUTDIR}/lrdl1b.pt, lrdl1b_eval.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
