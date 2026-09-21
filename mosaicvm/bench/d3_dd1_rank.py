"""DD-1 HIST-1 + hard-negative full-vocab ranking (roadmap
S12-S21). Changes ONLY the training objective vs the frozen
HIST-1 trainer (d3_dd1_hist.py --hist-mode prev):

  frozen: inputs (h40 cur + prev aux), DDHistNet H1024/D3,
  aux 256, corpus, partitions, AdamW lr 1e-3 wd 1e-4, cosine,
  600 steps, CHUNK 512, clip 1.0, mu/lam_m/lam_kl, seed.
  added: L_rank = max(0, m + l_c* - l_y) with c* = the
  model's CURRENT hardest wrong token (dynamic full-vocab
  mining via target-masked argmax), one frozen margin m.

Full-vocab CE remains mandatory (never candidate-only).
Existing loss terms (stored-teacher softplus margin, KL)
remain frozen; only lam_r varies across arms.

--smoke: training-only calibration pass (no training):
per-term magnitudes, CE-only / rank-only gradient norms,
active-hinge fraction. Arm λ_r derivation comes from this,
never from DEV behavior.
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
                                        build_hist_examples,
                                        D_STATE)
from mosaicvm.bench.d3_bridge import load_head, AUDIT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (required without --smoke)")
    p.add_argument("--lam-r", type=float, default=None,
                   help="ranking weight (required without "
                        "--smoke)")
    p.add_argument("--rank-margin", type=float, default=1.0,
                   help="frozen hinge margin m (default 1.0)")
    p.add_argument("--rank-smooth", action="store_true",
                   help="softplus instead of hinge")
    p.add_argument("--smoke", action="store_true",
                   help="calibration pass only, no training")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    if not a.smoke:
        assert a.out is not None and a.lam_r is not None
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)

    t0 = time.time()
    xm, xa, labels, embed, is_boundary = build_hist_examples(
        "h40", "prev")
    n = xm.shape[0]
    n_b = sum(is_boundary)
    print(f"{n} examples ({n_b} boundary) "
          f"({time.time()-t0:.0f}s)", flush=True)

    xm_mean, xm_std = xm.mean(0), xm.std(0).clamp_min(1e-4)
    xmn = (xm - xm_mean) / xm_std
    xa_mean, xa_std = xa.mean(0), xa.std(0).clamp_min(1e-4)
    xan = (xa - xa_mean) / xa_std

    w_head, w_norm = load_head()
    y_idx = torch.tensor([m["y"] for m in labels])
    max_c = max(len(m["comps"]) for m in labels)
    comp_idx = torch.tensor(
        [m["comps"] + [m["y"]] * (max_c - len(m["comps"]))
         for m in labels])
    comp_mask = torch.tensor(
        [[1.0] * len(m["comps"])
         + [0.0] * (max_c - len(m["comps"])) for m in labels])
    top_idx_t = torch.tensor([m["top_idx"] for m in labels])
    p_t = torch.stack([
        F.softmax(torch.tensor(m["top_val"]), dim=0)
        for m in labels])
    y_g = y_idx.cuda(); comp_g = comp_idx.cuda()
    cmask_g = comp_mask.cuda(); top_g = top_idx_t.cuda()
    pt_g = p_t.cuda()

    model = DDHistNet(1024, 3, 256)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"DDHistNet: {n_params:,} params "
          f"({time.time()-t0:.0f}s)", flush=True)

    w_head_g = w_head.cuda().requires_grad_(False)
    w_norm_g = w_norm.cuda().requires_grad_(False)
    CHUNK = 512
    MU, LAM_M, LAM_KL = 1.0, 1.0, 0.5  # frozen HIST-1 values

    def chunk_terms(zc, sl):
        """Per-row loss terms as a dict of [B] tensors."""
        zc_g = zc.cuda()
        zn = zc_g * torch.rsqrt(
            zc_g.square().mean(-1, keepdim=True) + 1e-6)
        zn = zn * w_norm_g
        lg = zn @ w_head_g.t()
        ce = F.cross_entropy(lg.float(), y_g[sl],
                             reduction="none")
        l_y = lg.gather(1, y_g[sl].unsqueeze(1)).squeeze(1)
        # dynamic hardest-wrong mining: mask target, argmax
        lg_masked = lg.scatter(
            1, y_g[sl].unsqueeze(1), float("-inf"))
        l_cstar = lg_masked.max(dim=1).values
        viol = a.rank_margin + l_cstar - l_y
        rank = F.softplus(viol) if a.rank_smooth \
            else F.relu(viol)
        # frozen stored-teacher margin term
        m_st = l_y.unsqueeze(1) - lg.gather(1, comp_g[sl])
        m_rows = (F.softplus(MU - m_st)
                  * cmask_g[sl]).sum(-1) / cmask_g[sl].sum(
                      -1).clamp_min(1.0)
        l16 = lg.gather(1, top_g[sl])
        logp = F.log_softmax(l16, dim=-1)
        kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log()
                          - logp)).sum(-1)
        return {"ce": ce, "rank": rank, "m_stored": m_rows,
                "kl": kl, "viol": viol.detach()}

    if a.smoke:
        # ---- training-only calibration pass (roadmap S17) ----
        model.eval()
        sums = {"ce": 0.0, "rank": 0.0, "m_stored": 0.0,
                "kl": 0.0}
        hinge_active = 0
        viol_pos = 0
        with torch.no_grad():
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                t = chunk_terms(model(xmn[sl], xan[sl]), sl)
                b = sl.stop - sl.start
                for k in sums:
                    sums[k] += t[k].sum().item()
                hinge_active += (t["viol"] > 0).sum().item()
                viol_pos += (t["viol"] > 0).sum().item()
        print(f"\n=== SMOKE (fresh init, n={n}) ===")
        for k, v in sums.items():
            print(f"  mean {k}: {v/n:.4f}")
        print(f"  active hinge fraction: "
              f"{hinge_active/n:.4f}")

        # gradient norms: separate full passes, CE-only vs
        # rank-only (training-weighted chunk scaling, as in
        # training)
        def grad_norm(term):
            model.zero_grad(set_to_none=True)
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                zc = model(xmn[sl], xan[sl])
                t = chunk_terms(zc, sl)
                part = t[term].mean() \
                    * (zc.shape[0] / n)
                part.backward()
            tot = 0.0
            for p_ in model.parameters():
                if p_.grad is not None:
                    tot += p_.grad.square().sum().item()
            return tot ** 0.5

        gn_ce = grad_norm("ce")
        gn_rank = grad_norm("rank")
        gn_m = grad_norm("m_stored")
        gn_kl = grad_norm("kl")
        print(f"  grad norm CE:      {gn_ce:.4f}")
        print(f"  grad norm rank:    {gn_rank:.4f}")
        print(f"  grad norm m_stored:{gn_m:.4f}")
        print(f"  grad norm KL:      {gn_kl:.4f}")
        if gn_rank > 0:
            for name, frac in (("LOW", 0.25), ("MED", 0.5),
                               ("HIGH", 1.0)):
                lam = frac * gn_ce / gn_rank
                print(f"  {name}: lam_r = {lam:.4f} "
                      f"(rank grad -> {frac:.2f}x CE grad)")
        smoke = {
            "n": n, "mean_ce": sums["ce"] / n,
            "mean_rank": sums["rank"] / n,
            "mean_m_stored": sums["m_stored"] / n,
            "mean_kl": sums["kl"] / n,
            "active_hinge_fraction": hinge_active / n,
            "grad_norm_ce": gn_ce, "grad_norm_rank": gn_rank,
            "grad_norm_m_stored": gn_m,
            "grad_norm_kl": gn_kl,
            "rank_margin": a.rank_margin,
            "rank_smooth": a.rank_smooth,
        }
        if gn_rank > 0:
            smoke["lam_r_arms"] = {
                "LOW": round(0.25 * gn_ce / gn_rank, 4),
                "MED": round(0.5 * gn_ce / gn_rank, 4),
                "HIGH": round(1.0 * gn_ce / gn_rank, 4),
            }
        Path("artifacts/d3_dd1_rank_smoke.json"
             ).write_text(json.dumps(smoke, indent=1))
        print(f"wrote artifacts/d3_dd1_rank_smoke.json "
              f"({time.time()-t0:.0f}s)")
        return

    # ---- training (identical recipe + rank term) ----
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.steps, eta_min=1e-3 * 0.02)

    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        h_act = 0; h_n = 0
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            zc = model(xmn[sl], xan[sl])
            t = chunk_terms(zc, sl)
            row = (t["ce"] + LAM_M * t["m_stored"]
                   + LAM_KL * t["kl"] + a.lam_r * t["rank"])
            part = row.mean() * (zc.shape[0] / n)
            part.backward()
            total += part.item()
            h_act += (t["viol"] > 0).sum().item()
            h_n += sl.stop - sl.start
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == a.steps:
            with torch.no_grad():
                accs = []
                for i in range(0, n, 2048):
                    sl = slice(i, min(i + 2048, n))
                    zg = model(xmn[sl], xan[sl]).cuda()
                    zn = zg * torch.rsqrt(
                        zg.square().mean(-1, keepdim=True)
                        + 1e-6) * w_norm_g
                    lg = zn @ w_head_g.t()
                    accs.append((lg.argmax(-1) == y_g[sl])
                                .float().mean().item())
            print(f"step {step}: L {total:.4f} "
                  f"train-argmax {sum(accs)/len(accs):.3f} "
                  f"hinge {h_act/h_n:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # boundary evaluation (train split)
    model.eval()
    with torch.no_grad():
        b_idx = [i for i, b in enumerate(is_boundary) if b]
        bxm = xmn[b_idx]; bxa = xan[b_idx]
        correct = 0; topk = {1: 0, 4: 0, 16: 0}
        for i in range(0, len(b_idx), 64):
            sl = slice(i, min(i + 64, len(b_idx)))
            zg = model(bxm[sl], bxa[sl]).cuda()
            zn = zg * torch.rsqrt(
                zg.square().mean(-1, keepdim=True) + 1e-6
            ) * w_norm_g
            lg = zn @ w_head_g.t()
            ys = y_g[[b_idx[j] for j in range(sl.start,
                                              sl.stop)]]
            pred = lg.argmax(-1)
            correct += (pred == ys).sum().item()
            for k in topk:
                topk[k] += (lg.topk(k, dim=-1).indices
                            == ys.unsqueeze(1)).any(1
                            ).sum().item()
        nb = len(b_idx)
        print(f"\nBOUNDARY (train, n={nb}): "
              f"S1={correct/nb:.4f} Top4={topk[4]/nb:.4f} "
              f"Top16={topk[16]/nb:.4f}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "hidden": 1024,
        "depth": 3, "aux_dim": 256, "hist_mode": "prev",
        "xm_mean": xm_mean, "xm_std": xm_std,
        "xa_mean": xa_mean, "xa_std": xa_std,
        "state_field": "h40",
        "args": {k: str(v) for k, v in vars(a).items()},
        "lam_r": a.lam_r, "rank_margin": a.rank_margin,
        "rank_smooth": a.rank_smooth,
        "seed": 104729, "n_params": n_params,
        "teacher_audit_sha256": hashlib.sha256(
            AUDIT.read_bytes()).hexdigest(),
    }, a.out / "dd1_hist.pt")
    print(f"wrote {a.out / 'dd1_hist.pt'} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
