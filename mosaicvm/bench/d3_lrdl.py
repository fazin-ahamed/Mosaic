"""LRDL-0: low-rank direct-logit latent, frozen source-head
basis control (roadmap S15-S28).

Architecture: root features (identical HIST-1 inputs) →
DDHistNet trunk (H1024/D3/aux256) → q ∈ R^256 → logits =
E_r @ q, where E_r = U_r Σ_r is a FROZEN rank-256 basis from
randomized SVD of the effective source head
W_eff = W_head · diag(w_norm).

Frozen: inputs, trunk architecture, corpus, partitions,
AdamW lr 1e-3 wd 1e-4, cosine 600 steps, CHUNK 512, clip 1.0,
seed 104729. Objective (S26, simple first): full-vocab CE +
λ_KL=0.5 · KL against teacher Top16. NO stored-margin term,
NO hard-negative hinge.

Modes:
  --verify-only   build + verify basis, write telemetry, exit
  --out DIR       train LRDL-0 and save checkpoint
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
from mosaicvm.bench.d3_bridge import load_head, load_embed, \
    AUDIT

BASIS_PATH = Path("artifacts/d3_lrdl0/basis_r256.pt")
TELEM_PATH = Path("artifacts/d3_lrdl0/basis_verify.json")
RANK = 256  # default; --rank overrides (basis path per rank)


class LRDLNet(nn.Module):
    """DDHistNet trunk with a 256-dim decision-native head."""

    def __init__(self, hidden=1024, depth=3, aux_dim=256,
                 rank=RANK):
        super().__init__()
        self.base = DDHistNet(hidden, depth, aux_dim)
        # replace the 4096-dim pseudo-state head with the
        # low-rank latent head
        self.base.head = nn.Linear(hidden, rank)

    def forward(self, x_main, x_aux):
        return self.base(x_main, x_aux)


def build_basis(rank=RANK):
    """Rank-r randomized SVD of W_eff = W_head·diag(w_norm).

    Returns E_r = U_r Σ_r [V, r] (frozen logit basis) and
    V_r [4096, r] (analysis only, for ceiling telemetry).
    """
    w_head, w_norm = load_head()
    # effective head: lg = W_head @ (w_norm ⊙ z_rms)
    w_eff = w_head * w_norm.unsqueeze(0)      # [V, 4096]
    w_eff_g = w_eff.cuda()
    del w_head
    torch.cuda.empty_cache()
    U, S, V = torch.svd_lowrank(w_eff_g, q=rank + 64,
                                niter=6)
    U_r = U[:, :rank].contiguous()            # [V, r]
    S_r = S[:rank].contiguous()
    V_r = V[:, :rank].contiguous()            # [4096, r]
    E_r = (U_r * S_r.unsqueeze(0)).contiguous()
    return E_r.cpu(), V_r.cpu(), S_r.cpu(), w_norm


def verify_basis(E_r, V_r, S_r, w_norm, rank=RANK):
    """S7 verification: dimensions, reconstruction telemetry,
    full-vocab argmax path, functional ceiling on the frozen
    HIST-1 model's actual z vectors."""
    w_head, _ = load_head()
    w_eff = w_head * w_norm.unsqueeze(0)
    del w_head
    # relative Frobenius error via singular values
    fro = w_eff.square().sum().item()
    err = max(0.0, fro - S_r.square().sum().item())
    rel_err = (err / fro) ** 0.5
    energy = S_r.square().sum().item() / fro

    tel = {
        "rank": rank,
        "E_shape": list(E_r.shape),
        "V_shape": list(V_r.shape),
        "rel_frobenius_error": round(rel_err, 6),
        "energy_fraction_top256": round(energy, 6),
        "s1": round(float(S_r[0]), 4),
        f"s{rank}": round(float(S_r[-1]), 4),
        "s_spectrum_decay": [round(float(S_r[i]), 3)
                             for i in range(0, rank,
                                            max(1, rank // 6))],
    }

    # functional ceiling: frozen HIST-1 model's z on all 2688
    # training rows — full-head logits vs low-rank logits
    ck = torch.load(
        "artifacts/d3_dd1_hist/dd1_hist_prev/dd1_hist.pt",
        map_location="cpu", weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"],
                     ck["aux_dim"])
    hist.load_state_dict(ck["model"])
    hist.eval()

    xm, xa, labels, embed, is_boundary = build_hist_examples(
        "h40", "prev")
    xmn = (xm - ck["xm_mean"]) / ck["xm_std"]
    xan = (xa - ck["xa_mean"]) / ck["xa_std"]
    y_idx = torch.tensor([m["y"] for m in labels])

    E_g = E_r.cuda()
    V_g = V_r.cuda()
    w_head2, w_norm2 = load_head()
    w_head_g = w_head2.cuda()
    w_norm_g = w_norm2.cuda()

    top1_agree = 0
    top16_overlap_sum = 0.0
    tgt_rank_lr_worse = 0
    n = xmn.shape[0]
    with torch.inference_mode():
        for i in range(0, n, 512):
            sl = slice(i, min(i + 512, n))
            z = hist(xmn[sl], xan[sl]).cuda()
            # RMS-only z: w_norm lives inside W_eff basis
            zr = z * torch.rsqrt(
                z.square().mean(-1, keepdim=True) + 1e-6)
            # full canonical path (both equivalent forms)
            lg_full = (zr * w_norm_g) @ w_head_g.t()
            lg_full_b = zr @ (w_head_g
                              * w_norm_g.unsqueeze(0)).t()
            assert torch.allclose(lg_full, lg_full_b,
                                  atol=1e-3)
            # low-rank path on the SAME RMS-only z
            q_z = zr @ V_g                   # [B, 256]
            lg_lr = q_z @ E_g.t()
            top1_agree += (lg_full.argmax(-1)
                           == lg_lr.argmax(-1)).sum().item()
            t16f = lg_full.topk(16, -1).indices
            t16l = lg_lr.topk(16, -1).indices
            for b in range(t16f.shape[0]):
                top16_overlap_sum += len(
                    set(t16f[b].tolist())
                    & set(t16l[b].tolist()))
            # target rank change
            ysl = y_idx[sl].cuda()
            rf = (lg_full > lg_full.gather(
                1, ysl.unsqueeze(1))).sum(1) + 1
            rl = (lg_lr > lg_lr.gather(
                1, ysl.unsqueeze(1))).sum(1) + 1
            tgt_rank_lr_worse += (rl > rf).sum().item()

    tel["functional_ceiling_on_hist1_z"] = {
        "n_rows": n,
        "top1_agreement": round(top1_agree / n, 4),
        "mean_top16_overlap": round(
            top16_overlap_sum / n / 16, 4),
        "target_rank_worse_fraction": round(
            tgt_rank_lr_worse / n, 4),
    }
    return tel


def probe_free_q(E_r):
    """Expressive ceiling of the frozen basis: optimize a FREE
    per-row q (no trunk) for each of the 2688 training rows
    with full-vocab CE. Reports the fraction of rows whose
    true token can be made argmax within span(E_r). This is
    the basis ceiling WITHOUT trunk-generalization confounds.
    Distinct true tokens get a shared q when identical (y, z)
    cannot be separated anyway; we optimize per row."""
    xm, xa, labels, embed, is_boundary = build_hist_examples(
        "h40", "prev")
    y_idx = torch.tensor([m["y"] for m in labels])
    E_g = E_r.cuda()
    q = torch.zeros(xm.shape[0], RANK, device="cuda",
                    requires_grad=True)
    opt = torch.optim.Adam([q], lr=0.05)
    for step in range(300):
        opt.zero_grad(set_to_none=True)
        lg = q @ E_g.t()
        loss = F.cross_entropy(lg, y_idx.cuda())
        loss.backward()
        opt.step()
    with torch.no_grad():
        lg = q @ E_g.t()
        top1 = (lg.argmax(-1) == y_idx.cuda()
                ).float().mean().item()
        # boundary rows only
        b = torch.tensor(is_boundary)
        top1_b = (lg.argmax(-1) == y_idx.cuda())[b
            ].float().mean().item()
    return {"n": int(xm.shape[0]), "steps": 300,
            "free_q_top1_all": round(top1, 4),
            "free_q_top1_boundary": round(top1_b, 4),
            "final_ce": round(float(loss), 4)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--probe", action="store_true",
                   help="run the free-q expressive probe")
    p.add_argument("--rank", type=int, default=RANK,
                   choices=(128, 256, 512))
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    basis_path = Path(
        f"artifacts/d3_lrdl0/basis_r{a.rank}.pt")
    telem_path = Path(
        f"artifacts/d3_lrdl0/basis_verify_r{a.rank}.json")

    # ---- basis (build once, reuse) ----
    rank = a.rank
    if basis_path.exists():
        bk = torch.load(basis_path, map_location="cpu",
                        weights_only=True)
        assert bk["rank"] == rank, (
            f"basis file rank {bk['rank']} != --rank {rank}")
        assert (bk["E_r"].shape[1] == rank
                and bk["V_r"].shape[1] == rank
                and bk["S_r"].shape[0] == rank), (
            f"basis dims {tuple(bk['E_r'].shape)} do not "
            f"match rank {rank}")
        E_r, V_r, S_r, w_norm = (bk["E_r"], bk["V_r"],
                                  bk["S_r"], bk["w_norm"])
        print(f"basis loaded rank={rank} "
              f"({time.time()-t0:.0f}s)", flush=True)
    else:
        E_r, V_r, S_r, w_norm = build_basis(rank)
        basis_path.parent.mkdir(parents=True,
                                exist_ok=True)
        torch.save({"E_r": E_r, "V_r": V_r, "S_r": S_r,
                    "w_norm": w_norm, "rank": rank},
                   basis_path)
        print(f"basis built+saved ({time.time()-t0:.0f}s)",
              flush=True)

    if not telem_path.exists():
        tel = verify_basis(E_r, V_r, S_r, w_norm, rank)
        telem_path.parent.mkdir(parents=True,
                                exist_ok=True)
        telem_path.write_text(json.dumps(tel, indent=1))
        print("basis verification:", json.dumps(
            {k: v for k, v in tel.items()
             if k != "s_spectrum_decay"}, indent=1),
            flush=True)
    else:
        print("basis telemetry exists", flush=True)

    if a.probe:
        res = probe_free_q(E_r)
        print("free-q probe:", json.dumps(res), flush=True)
        Path("artifacts/d3_lrdl0/free_q_probe.json"
             ).write_text(json.dumps(res, indent=1))
        return
    if a.verify_only:
        return
    assert a.out is not None

    # ---- training data (identical HIST-1 inputs) ----
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

    y_idx = torch.tensor([m["y"] for m in labels])
    top_idx_t = torch.tensor([m["top_idx"] for m in labels])
    p_t = torch.stack([
        F.softmax(torch.tensor(m["top_val"]), dim=0)
        for m in labels])
    y_g = y_idx.cuda(); top_g = top_idx_t.cuda()
    pt_g = p_t.cuda()

    model = LRDLNet(1024, 3, 256, a.rank)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"LRDLNet: {n_params:,} params "
          f"({time.time()-t0:.0f}s)", flush=True)

    E_g = E_r.cuda().requires_grad_(False)
    LAM_KL = 0.5  # frozen recipe value
    CHUNK = 512

    def chunk_losses(zc, sl):
        """q → logits via FROZEN basis; CE_full + KL only."""
        lg = zc.cuda() @ E_g.t()              # [B, V]
        ce = F.cross_entropy(lg.float(), y_g[sl],
                             reduction="none")
        l16 = lg.gather(1, top_g[sl])
        logp = F.log_softmax(l16, dim=-1)
        kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log()
                          - logp)).sum(-1)
        return ce + LAM_KL * kl

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.steps, eta_min=1e-3 * 0.02)

    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            zc = model(xmn[sl], xan[sl])
            part = chunk_losses(zc, sl).mean() \
                * (zc.shape[0] / n)
            part.backward()
            total += part.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == a.steps:
            with torch.no_grad():
                accs = []
                for i in range(0, n, 2048):
                    sl = slice(i, min(i + 2048, n))
                    q = model(xmn[sl], xan[sl]).cuda()
                    lg = q @ E_g.t()
                    accs.append((lg.argmax(-1) == y_g[sl])
                                .float().mean().item())
            print(f"step {step}: L {total:.4f} "
                  f"train-argmax {sum(accs)/len(accs):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # boundary evaluation (train split)
    model.eval()
    with torch.no_grad():
        b_idx = [i for i, b in enumerate(is_boundary) if b]
        bxm = xmn[b_idx]; bxa = xan[b_idx]
        correct = 0; topk = {1: 0, 4: 0, 16: 0}
        for i in range(0, len(b_idx), 64):
            sl = slice(i, min(i + 64, len(b_idx)))
            q = model(bxm[sl], bxa[sl]).cuda()
            lg = q @ E_g.t()
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
        "rank": a.rank,
        "xm_mean": xm_mean, "xm_std": xm_std,
        "xa_mean": xa_mean, "xa_std": xa_std,
        "state_field": "h40",
        "basis_sha256": hashlib.sha256(
            basis_path.read_bytes()).hexdigest(),
        "args": {k: str(v) for k, v in vars(a).items()},
        "lam_kl": LAM_KL,
        "seed": 104729, "n_params": n_params,
        "teacher_audit_sha256": hashlib.sha256(
            AUDIT.read_bytes()).hexdigest(),
    }, a.out / "lrdl0.pt")
    print(f"wrote {a.out / 'lrdl0.pt'} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
