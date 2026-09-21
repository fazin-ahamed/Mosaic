"""SCALE210 trainer (roadmap S19-S22).

Stage A (--stage hist): train the EXACT HIST-1 architecture
(DDHistNet H1024/D3/aux256, prev-aux, full-vocab CE + stored
margin + KL, AdamW 1e-3/wd 1e-4, cosine 600, CHUNK 512, clip
1.0, seed 104729) on the UNION corpus: 84 original train
prompts (d3_data_105 + d3_r2_teacher) + 126 scale prompts
(d3_scale_data + d3_scale_teacher) = 210 prompts / 6720 rows.

Stage B (--stage ws): freeze the HIST-210 trunk; train only
Q: z_rms -> q256 (init V_r^T) with CE + 0.5 KL on the same
6720 rows (exact WS-1 recipe, frozen E256 basis).

Only dataset size changes — architecture, objective,
optimizer, schedule, seed are frozen (S21).
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
                                      TEACHER, AUDIT)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON

SCALE_CAP = Path("artifacts/d3_scale_data")
SCALE_TEACHER = Path("artifacts/d3_scale_teacher")
SCALE_MANIFEST = Path("artifacts/d3_scale/prompts_scale.json")
BASIS = Path("artifacts/d3_lrdl0/basis_r256.pt")
OUTDIR = Path("artifacts/d3_scale210")


def build_union_examples(state_field, hist_mode):
    """84 original + 126 scale rows, identical construction to
    build_hist_examples (which this mirrors for the original
    84 with its hash gate)."""
    ver = json.loads(AUDIT.read_text())
    assert ver["verdict"] == "PASS"
    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    cap_by_cat = {c["category"]: c for c in caps}
    embed = load_embed()

    main_feats, aux_feats, labels = [], [], []
    sources = []          # provenance per row
    n_leak = 0

    # ---- original 84 (teacher: d3_r2_teacher) ----
    for cat in sorted(ver["record_hashes_full"]):
        rec = torch.load(TEACHER / f"{cat}.pt",
                         map_location="cpu",
                         weights_only=True)
        cap = cap_by_cat[cat]
        S, K = rec["S"], rec["K"]
        toks = rec["tokens"]
        states = cap[state_field][S - 2:S + K - 1].float()
        for j in range(K):
            t = S + j
            row = rec["decisions"][j]
            y = row["true_token"]
            cur_state = states[j + 1].flatten()
            prev_state = states[j].flatten()
            mf = make_dd_feat(embed, cur_state, toks[t],
                              toks[t - 1], t)
            af = torch.zeros(D_STATE) if hist_mode == "zero" \
                else prev_state
            main_feats.append(mf); aux_feats.append(af)
            labels.append({"y": y, "cat": cat, "j": j,
                           "top_idx": row["gold"]["top_idx"],
                           "top_val": row["gold"]["top_val"],
                           "comps": [i for i in
                                     row["gold"]["top_idx"]
                                     if i != y][:8]})
            sources.append("orig84")
            if j == 0 and hist_mode != "zero":
                h40_S = cap[state_field][S].float().flatten()
                if torch.allclose(af, h40_S, atol=1e-6):
                    raise RuntimeError(f"LEAK {cat}")
                n_leak += 1

    # ---- scale 126 (teacher: d3_scale_teacher) ----
    sver = json.loads((SCALE_TEACHER / "audit.json")
                      .read_text())
    assert sver["overlap_checks"]["all_empty_required"]
    spec = json.loads(SCALE_MANIFEST.read_text())
    scale_prompts = {p["category"]: p["prompt"]
                     for p in spec["prompts"]}
    scale_caps = load_captures(str(SCALE_CAP), spec, None,
                               HORIZON)
    scam_by_cat = {c["category"]: c for c in scale_caps}
    for cat in sorted(sver["records"]):
        rec = torch.load(SCALE_TEACHER / f"{cat}.pt",
                         map_location="cpu",
                         weights_only=True)
        cap = scam_by_cat[cat]
        S, K = rec["S"], rec["K"]
        toks = rec["tokens"]
        states = cap[state_field][S - 2:S + K - 1].float()
        for j in range(K):
            t = S + j
            row = rec["decisions"][j]
            y = row["true_token"]
            cur_state = states[j + 1].flatten()
            prev_state = states[j].flatten()
            mf = make_dd_feat(embed, cur_state, toks[t],
                              toks[t - 1], t)
            af = torch.zeros(D_STATE) if hist_mode == "zero" \
                else prev_state
            main_feats.append(mf); aux_feats.append(af)
            labels.append({"y": y, "cat": cat, "j": j,
                           "top_idx": row["gold"]["top_idx"],
                           "top_val": row["gold"]["top_val"],
                           "comps": [i for i in
                                     row["gold"]["top_idx"]
                                     if i != y][:8]})
            sources.append("scale126")
            if j == 0:
                h40_S = cap[state_field][S].float().flatten()
                if torch.allclose(af, h40_S, atol=1e-6):
                    raise RuntimeError(f"LEAK {cat}")
                n_leak += 1

    xm = torch.stack(main_feats); xa = torch.stack(aux_feats)
    from collections import Counter
    counts = Counter(sources)
    print(f"union: {xm.shape[0]} rows "
          f"({dict(counts)}), leak checks {n_leak}",
          flush=True)
    return xm, xa, labels, embed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("hist", "ws"),
                   required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    xm, xa, labels, embed = build_union_examples(
        "h40", "prev")
    n = xm.shape[0]
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
         + [0.0] * (max_c - len(m["comps"]))
         for m in labels])
    top_idx_t = torch.tensor(
        [m["top_idx"] for m in labels])
    p_t = torch.stack([
        F.softmax(torch.tensor(m["top_val"]), dim=0)
        for m in labels])
    y_g = y_idx.cuda(); comp_g = comp_idx.cuda()
    cmask_g = comp_mask.cuda(); top_g = top_idx_t.cuda()
    pt_g = p_t.cuda()

    if a.stage == "hist":
        model = DDHistNet(1024, 3, 256)
        n_params = sum(p_.numel()
                       for p_ in model.parameters())
        print(f"DDHistNet(210): {n_params:,} params, "
              f"{n} rows ({time.time()-t0:.0f}s)", flush=True)
        w_head_g = w_head.cuda().requires_grad_(False)
        w_norm_g = w_norm.cuda().requires_grad_(False)
        MU, LAM_M, LAM_KL = 1.0, 1.0, 0.5
        CHUNK = 512

        def chunk_losses(zc, sl):
            zc_g = zc.cuda()
            zn = zc_g * torch.rsqrt(
                zc_g.square().mean(-1, keepdim=True) + 1e-6)
            zn = zn * w_norm_g
            lg = zn @ w_head_g.t()
            ce = F.cross_entropy(lg.float(), y_g[sl],
                                 reduction="none")
            l_y = lg.gather(
                1, y_g[sl].unsqueeze(1)).squeeze(1)
            m = l_y.unsqueeze(1) - lg.gather(1, comp_g[sl])
            m_rows = (F.softplus(MU - m)
                      * cmask_g[sl]).sum(-1) / cmask_g[sl].sum(
                          -1).clamp_min(1.0)
            l16 = lg.gather(1, top_g[sl])
            logp = F.log_softmax(l16, dim=-1)
            kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log()
                              - logp)).sum(-1)
            return ce + LAM_M * m_rows + LAM_KL * kl

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=600, eta_min=1e-3 * 0.02)
        for step in range(1, 601):
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                zc = model(xmn[sl], xan[sl])
                part = chunk_losses(zc, sl).mean() \
                    * (zc.shape[0] / n)
                part.backward()
                total += part.item()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            opt.step(); sched.step()
            if step % 50 == 0 or step == 600:
                with torch.no_grad():
                    accs = []
                    for i in range(0, n, 2048):
                        sl = slice(i, min(i + 2048, n))
                        zg = model(xmn[sl],
                                   xan[sl]).cuda()
                        zn = zg * torch.rsqrt(
                            zg.square().mean(
                                -1, keepdim=True) + 1e-6
                        ) * w_norm_g
                        lg = zn @ w_head_g.t()
                        accs.append(
                            (lg.argmax(-1) == y_g[sl])
                            .float().mean().item())
                print(f"step {step}: L {total:.4f} "
                      f"argmax {sum(accs)/len(accs):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        model.eval()
        torch.save({
            "model": model.state_dict(), "hidden": 1024,
            "depth": 3, "aux_dim": 256,
            "hist_mode": "prev",
            "xm_mean": xm_mean, "xm_std": xm_std,
            "xa_mean": xa_mean, "xa_std": xa_std,
            "state_field": "h40",
            "args": {"stage": "hist", "corpus": "210"},
            "seed": 104729, "n_params": n_params,
            "n_rows": n,
            "teacher_audit_sha256": hashlib.sha256(
                AUDIT.read_bytes()).hexdigest(),
            "scale_teacher_audit_sha256": hashlib.sha256(
                (SCALE_TEACHER / "audit.json")
                .read_bytes()).hexdigest(),
        }, OUTDIR / "hist210.pt")
        print(f"wrote {OUTDIR}/hist210.pt "
              f"({time.time()-t0:.0f}s)", flush=True)
        return

    # ---- stage ws ----
    ck = torch.load(OUTDIR / "hist210.pt",
                    map_location="cpu", weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"],
                     ck["aux_dim"])
    hist.load_state_dict(ck["model"])
    hist.eval()
    for p_ in hist.parameters():
        p_.requires_grad_(False)
    # use the HIST-210's own normalization stats
    xmn = (xm - ck["xm_mean"]) / ck["xm_std"]
    xan = (xa - ck["xa_mean"]) / ck["xa_std"]

    bk = torch.load(BASIS, map_location="cpu",
                    weights_only=True)
    E_r = bk["E_r"]; V_r = bk["V_r"]
    E_g = E_r.cuda()
    w_norm_g = w_norm.cuda()

    # cache z_rms for all rows
    print(f"caching z_rms for {n} rows...", flush=True)
    zr_cache = torch.empty(n, 4096, device="cuda")
    with torch.inference_mode():
        for i in range(0, n, 256):
            sl = slice(i, min(i + 256, n))
            z = hist(xmn[sl], xan[sl]).cuda()
            zr_cache[sl] = z * torch.rsqrt(
                z.square().mean(-1, keepdim=True) + 1e-6)
    print(f"cached ({time.time()-t0:.0f}s)", flush=True)

    Q = nn.Parameter(V_r.t().clone().cuda())
    opt = torch.optim.AdamW([Q], lr=1e-3,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=600, eta_min=1e-3 * 0.02)
    LAM_KL = 0.5
    CHUNK = 512
    for step in range(1, 601):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            q = zr_cache[sl] @ Q.t()
            lg = q @ E_g.t()
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
        opt.step(); sched.step()
        if step % 100 == 0 or step == 600:
            with torch.no_grad():
                accs = []
                for i in range(0, n, 2048):
                    sl = slice(i, min(i + 2048, n))
                    lg = (zr_cache[sl] @ Q.t()) @ E_g.t()
                    accs.append((lg.argmax(-1) == y_g[sl])
                                .float().mean().item())
            print(f"step {step}: L {total:.4f} "
                  f"argmax {sum(accs)/len(accs):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"Q": Q.detach().cpu(), "rank": 256,
                "hist210": str(OUTDIR / "hist210.pt"),
                "hist210_sha256": hashlib.sha256(
                    (OUTDIR / "hist210.pt").read_bytes()
                ).hexdigest(),
                "basis_sha256": hashlib.sha256(
                    BASIS.read_bytes()).hexdigest(),
                "n_rows": n, "seed": 104729},
               OUTDIR / "ws210.pt")
    print(f"wrote {OUTDIR}/ws210.pt "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
