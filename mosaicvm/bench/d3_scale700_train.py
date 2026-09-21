"""SCALE700 trainer (roadmap S13-S17).

Stage A (--stage hist): HIST-1 architecture (DDHistNet
H1024/D3/aux256, prev-aux, full-vocab CE + stored margin +
KL, AdamW 1e-3/wd 1e-4, cosine 600, CHUNK 512, clip 1.0,
seed 104729) on the 700-prompt union: 84 original + 126
scale210 + 490 scale700 = 700 prompts / 22,400 transitions.

Stage B (--stage ws): frozen HIST-700 trunk; train only
Q: z_rms -> q256 (init V_r^T), CE + 0.5 KL (WS-1 recipe).

Dataset size is the only experimental lever. Each stage is a
separate process invocation; each builds the union corpus
itself (~3 min) and frees raw tensors after normalization.
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

from mosaicvm.bench.d3_dd1_hist import DDHistNet
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import (load_head, load_embed,
                                      TEACHER, AUDIT)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON

S210_CAP = Path("artifacts/d3_scale_data")
S210_TEACHER = Path("artifacts/d3_scale_teacher")
S210_MANIFEST = Path(
    "artifacts/d3_scale/prompts_scale.json")
S700_CAP = Path("artifacts/d3_scale700_data")
S700_TEACHER = Path("artifacts/d3_scale700_teacher")
S700_MANIFEST = Path(
    "artifacts/d3_scale700/prompts_scale700.json")
BASIS = Path("artifacts/d3_lrdl0/basis_r256.pt")
OUTDIR = Path("artifacts/d3_scale700_train")


def emit_rows(rec, cap, embed, xm, xa, labels):
    S, K = rec["S"], rec["K"]
    toks = rec["tokens"]
    states = cap["h40"][S - 2:S + K - 1].float()
    for j in range(K):
        t = S + j
        row = rec["decisions"][j]
        y = row["true_token"]
        cur = states[j + 1].flatten()
        prev = states[j].flatten()
        mf = make_dd_feat(embed, cur, toks[t],
                          toks[t - 1], t)
        xm.append(mf)
        xa.append(prev)
        labels.append(
            {"y": y, "cat": rec["category"], "j": j,
             "top_idx": row["gold"]["top_idx"],
             "top_val": row["gold"]["top_val"],
             "comps": [i for i in
                       row["gold"]["top_idx"]
                       if i != y][:8]})
    # boundary leak check (prev state must not equal the
    # first gold future state)
    h40_S = cap["h40"][S].float().flatten()
    prev0 = cap["h40"][S - 2].float().flatten()
    if torch.allclose(prev0, h40_S, atol=1e-6):
        raise RuntimeError(f"LEAK {rec['category']}")


def build_union():
    embed = load_embed()
    xm, xa, labels = [], [], []
    counts = {}

    def load_teachers(teacher_dir, audit_name, cap_dir,
                      manifest, tag):
        audit = json.loads(
            (teacher_dir / "audit.json").read_text())
        assert audit["overlap_checks"]["all_empty_required"]
        spec = json.loads(manifest.read_text())
        caps = load_captures(str(cap_dir), spec, None,
                             HORIZON)
        by = {c["category"]: c for c in caps}
        base = len(xm)
        for cat in sorted(audit["records"]):
            rec = torch.load(teacher_dir / f"{cat}.pt",
                             map_location="cpu",
                             weights_only=True)
            emit_rows(rec, by[cat], embed, xm, xa, labels)
        counts[tag] = len(xm) - base

    # 84 original
    ver = json.loads(AUDIT.read_text())
    assert ver["verdict"] == "PASS"
    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    by = {c["category"]: c for c in caps}
    base = len(xm)
    for cat in sorted(ver["record_hashes_full"]):
        rec = torch.load(TEACHER / f"{cat}.pt",
                         map_location="cpu",
                         weights_only=True)
        emit_rows(rec, by[cat], embed, xm, xa, labels)
    counts["orig84"] = len(xm) - base

    load_teachers(S210_TEACHER, "s210", S210_CAP,
                  S210_MANIFEST, "scale126")
    load_teachers(S700_TEACHER, "s700", S700_CAP,
                  S700_MANIFEST, "scale490")

    xm_t = torch.stack(xm)
    xa_t = torch.stack(xa)
    del xm, xa
    print(f"union: {xm_t.shape[0]} rows ({counts})",
          flush=True)
    return xm_t, xa_t, labels, counts


def label_tensors(labels):
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
    return y_idx, comp_idx, comp_mask, top_idx_t, p_t


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

    xm, xa, labels, counts = build_union()
    n = xm.shape[0]
    assert n == 700 * 32, (n, 700 * 32)
    y_idx, comp_idx, comp_mask, top_idx_t, p_t = \
        label_tensors(labels)
    y_g = y_idx.cuda(); comp_g = comp_idx.cuda()
    cmask_g = comp_mask.cuda(); top_g = top_idx_t.cuda()
    pt_g = p_t.cuda()
    del labels

    w_head, w_norm = load_head()

    if a.stage == "hist":
        xm_mean = xm.mean(0)
        xm_std = xm.std(0).clamp_min(1e-4)
        xmn = (xm - xm_mean) / xm_std
        xa_mean = xa.mean(0)
        xa_std = xa.std(0).clamp_min(1e-4)
        xan = (xa - xa_mean) / xa_std
        del xm, xa
        print(f"normalized; training HIST-700 on {n} rows "
              f"({time.time()-t0:.0f}s)", flush=True)

        model = DDHistNet(1024, 3, 256)
        n_params = sum(p_.numel()
                       for p_ in model.parameters())
        w_head_g = w_head.cuda().requires_grad_(False)
        w_norm_g = w_norm.cuda().requires_grad_(False)
        MU, LAM_M, LAM_KL = 1.0, 1.0, 0.5
        CHUNK = 512

        def chunk_losses(zc, sl):
            zc_g = zc.cuda()
            zn = zc_g * torch.rsqrt(
                zc_g.square().mean(-1, keepdim=True)
                + 1e-6) * w_norm_g
            lg = zn @ w_head_g.t()
            ce = F.cross_entropy(lg.float(), y_g[sl],
                                 reduction="none")
            l_y = lg.gather(
                1, y_g[sl].unsqueeze(1)).squeeze(1)
            m = l_y.unsqueeze(1) - lg.gather(1,
                                             comp_g[sl])
            m_rows = (F.softplus(MU - m)
                      * cmask_g[sl]).sum(-1) \
                / cmask_g[sl].sum(-1).clamp_min(1.0)
            l16 = lg.gather(1, top_g[sl])
            logp = F.log_softmax(l16, dim=-1)
            kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9)
                              .log() - logp)).sum(-1)
            return ce + LAM_M * m_rows + LAM_KL * kl

        opt = torch.optim.AdamW(model.parameters(),
                                lr=1e-3,
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
                    for i in range(0, n, 4096):
                        sl = slice(i, min(i + 4096, n))
                        zg = model(xmn[sl],
                                   xan[sl]).cuda()
                        zn = zg * torch.rsqrt(
                            zg.square().mean(
                                -1, keepdim=True)
                            + 1e-6) * w_norm_g
                        lg = zn @ w_head_g.t()
                        accs.append(
                            (lg.argmax(-1) == y_g[sl])
                            .float().mean().item())
                print(f"step {step}: L {total:.4f} argmax "
                      f"{sum(accs)/len(accs):.3f} "
                      f"({time.time()-t0:.0f}s)",
                      flush=True)
        model.eval()
        torch.save({
            "model": model.state_dict(), "hidden": 1024,
            "depth": 3, "aux_dim": 256,
            "hist_mode": "prev",
            "xm_mean": xm_mean, "xm_std": xm_std,
            "xa_mean": xa_mean, "xa_std": xa_std,
            "state_field": "h40",
            "args": {"stage": "hist",
                     "corpus": "700"},
            "seed": 104729, "n_params": n_params,
            "n_rows": n, "row_counts": counts,
            "teacher_audit_sha256": hashlib.sha256(
                AUDIT.read_bytes()).hexdigest(),
            "scale210_teacher_audit_sha256":
                hashlib.sha256((S210_TEACHER
                                / "audit.json")
                               .read_bytes()).hexdigest(),
            "scale700_teacher_audit_sha256":
                hashlib.sha256((S700_TEACHER
                                / "audit.json")
                               .read_bytes()).hexdigest(),
        }, OUTDIR / "hist700.pt")
        print(f"wrote {OUTDIR}/hist700.pt "
              f"({time.time()-t0:.0f}s)", flush=True)
        return

    # ---- stage ws ----
    ck = torch.load(OUTDIR / "hist700.pt",
                    map_location="cpu",
                    weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"],
                     ck["aux_dim"])
    hist.load_state_dict(ck["model"])
    hist.eval()
    for p_ in hist.parameters():
        p_.requires_grad_(False)
    xmn = (xm - ck["xm_mean"]) / ck["xm_std"]
    xan = (xa - ck["xa_mean"]) / ck["xa_std"]
    del xm, xa
    print(f"normalized by hist700 stats; caching z_rms "
          f"({time.time()-t0:.0f}s)", flush=True)

    bk = torch.load(BASIS, map_location="cpu",
                    weights_only=True)
    E_g = bk["E_r"].cuda()
    V_r = bk["V_r"]
    w_norm_g = w_norm.cuda()

    zr_cache = torch.empty(n, 4096, device="cuda")
    with torch.inference_mode():
        for i in range(0, n, 256):
            sl = slice(i, min(i + 256, n))
            z = hist(xmn[sl], xan[sl]).cuda()
            zr_cache[sl] = z * torch.rsqrt(
                z.square().mean(-1, keepdim=True) + 1e-6)
    del xmn, xan
    print(f"z_rms cached ({time.time()-t0:.0f}s)",
          flush=True)

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
            kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9)
                              .log() - logp)).sum(-1)
            part = (ce + LAM_KL * kl).mean() \
                * (q.shape[0] / n)
            part.backward()
            total += part.item()
        torch.nn.utils.clip_grad_norm_([Q], 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == 600:
            with torch.no_grad():
                accs = []
                for i in range(0, n, 4096):
                    sl = slice(i, min(i + 4096, n))
                    lg = (zr_cache[sl] @ Q.t()) @ E_g.t()
                    accs.append((lg.argmax(-1) == y_g[sl])
                                .float().mean().item())
            print(f"step {step}: L {total:.4f} argmax "
                  f"{sum(accs)/len(accs):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"Q": Q.detach().cpu(), "rank": 256,
                "hist700": str(OUTDIR / "hist700.pt"),
                "hist700_sha256": hashlib.sha256(
                    (OUTDIR / "hist700.pt").read_bytes()
                ).hexdigest(),
                "basis_sha256": hashlib.sha256(
                    BASIS.read_bytes()).hexdigest(),
                "n_rows": n, "seed": 104729},
               OUTDIR / "ws700.pt")
    print(f"wrote {OUTDIR}/ws700.pt "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
