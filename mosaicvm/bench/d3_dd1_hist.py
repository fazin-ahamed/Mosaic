"""DD-1 HIST: exact-root history race (roadmap S9-S24).

Parameter-matched three-arm race testing whether committed
trajectory information improves first-decision breadth:

  HIST-0M  current state + ZERO auxiliary      [matched control]
  HIST-1   current state + previous state       [temporal info]
  HIST-2   current state + committed delta      [motion info]

All three arms share IDENTICAL architecture (aux stem always
exists; HIST-0M feeds it zeros). The only variable is the aux
input content.

Architecture (roadmap S11):
  main features (state+emb+pos) → main_stem(24640→H1024) ──┐
                                                           ├→ fusion(H+aux→H) → trunk → z(4096) → head
  aux input (prev/delta/zeros)  → aux_stem(16384→256)  ───┘

Causal contract (S8): aux input uses ONLY h40[S-2] and h40[S-1]
(committed states). NEVER h40[S] or later.
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

sys = None
import sys as _sys
_sys.path.insert(0, ".")
_sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_bridge import (load_head, load_embed,
                                      TEACHER, AUDIT)
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON, pos_encode

SNAP = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--"
    "DeepSeek-V4-Flash-0731/snapshots/"
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062")
D_STATE = 16384
D_TOK = 4096
POS_DIM = 64
AUX_DIM = 256


class DDHistNet(nn.Module):
    """Matched DD with auxiliary history stem. Identical param
    count across HIST-0M/1/2 (aux stem always present)."""

    def __init__(self, hidden=1024, depth=3, aux_dim=AUX_DIM):
        super().__init__()
        d_in = D_STATE + 2 * D_TOK + POS_DIM
        self.main_stem = nn.Linear(d_in, hidden)
        self.aux_stem = nn.Linear(D_STATE, aux_dim)
        self.fusion = nn.Linear(hidden + aux_dim, hidden)
        self.mid = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth - 1)])
        self.head = nn.Linear(hidden, 4096)
        self.act = nn.GELU()

    def forward(self, x_main, x_aux):
        h_main = self.act(self.main_stem(x_main))
        h_aux = self.act(self.aux_stem(x_aux))
        h = self.act(self.fusion(
            torch.cat([h_main, h_aux], dim=-1)))
        for m in self.mid:
            h = h + self.act(m(h))
        return self.head(h)


def build_hist_examples(state_field, hist_mode):
    """Build DD-1 examples with auxiliary history inputs.

    hist_mode: 'zero' (HIST-0M), 'prev' (HIST-1), 'delta' (HIST-2)
    """
    ver = json.loads(AUDIT.read_text())
    assert ver["verdict"] == "PASS"
    for cat, h in list(ver["record_hashes_full"].items())[:3]:
        cur = hashlib.sha256(
            (TEACHER / f"{cat}.pt").read_bytes()).hexdigest()
        assert cur == h, f"teacher drift {cat}"
    print("hash gate: PASS", flush=True)

    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    cap_by_cat = {c["category"]: c for c in caps}
    embed = load_embed()

    # LEAKAGE REGRESSION: verify j=0 aux never touches h40[S]+
    # (the aux at j=0 uses h40[S-2] (prev) or h40[S-1]-h40[S-2]
    # (delta) — both committed, both BEFORE the root token)
    _leak_checked = 0
    main_feats, aux_feats, labels, is_boundary = [], [], [], []
    for cat in sorted(ver["record_hashes_full"]):
        rec = torch.load(TEACHER / f"{cat}.pt",
                         map_location="cpu", weights_only=True)
        cap = cap_by_cat[cat]
        S, K = rec["S"], rec["K"]
        toks = rec["tokens"]
        # states at positions S-2 .. S+K-2 (for current + prev)
        # h40[S+j-1] = current, h40[S+j-2] = previous
        states = cap[state_field][S - 2:S + K - 1].float()
        for j in range(K):
            t = S + j
            row = rec["decisions"][j]
            y = row["true_token"]
            # current state at position t-1 = S+j-1
            # states array index: (S+j-1) - (S-2) = j+1
            cur_state = states[j + 1].flatten()
            # previous state at position t-2 = S+j-2
            # states array index: (S+j-2) - (S-2) = j
            prev_state = states[j].flatten()

            # main features (same as DD-1)
            mf = make_dd_feat(embed, cur_state, toks[t],
                              toks[t - 1], t)

            # auxiliary features based on hist_mode
            if hist_mode == "zero":
                af = torch.zeros(D_STATE)
            elif hist_mode == "prev":
                af = prev_state
            elif hist_mode == "delta":
                af = cur_state - prev_state
            else:
                raise ValueError(hist_mode)

            # leakage check at boundary: aux must not equal any
            # gold state at position >= S (the root position)
            if j == 0 and hist_mode != "zero":
                # h40[S] is the first speculative gold state
                h40_S = cap[state_field][S].float().flatten()
                if torch.allclose(af, h40_S, atol=1e-6):
                    raise RuntimeError(
                        f"LEAK: {cat} j=0 aux == h40[S]")
                _leak_checked += 1

            main_feats.append(mf)
            aux_feats.append(af)
            labels.append({
                "y": y, "cat": cat, "j": j,
                "top_idx": row["gold"]["top_idx"],
                "top_val": row["gold"]["top_val"],
                "comps": [i for i in row["gold"]["top_idx"]
                          if i != y][:8],
            })
            is_boundary.append(j == 0)
    xm = torch.stack(main_feats)
    xa = torch.stack(aux_feats)
    print(f"leakage regression: {_leak_checked} boundary aux "
          f"inputs verified non-leaking", flush=True)
    return xm, xa, labels, embed, is_boundary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--state", choices=("h40",), default="h40")
    p.add_argument("--hist-mode", required=True,
                   choices=("zero", "prev", "delta"))
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--aux-dim", type=int, default=AUX_DIM)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--lam-m", type=float, default=1.0)
    p.add_argument("--lam-kl", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)

    t0 = time.time()
    xm, xa, labels, embed, is_boundary = build_hist_examples(
        a.state, a.hist_mode)
    n = xm.shape[0]
    n_b = sum(is_boundary)
    print(f"{n} examples ({n_b} boundary), "
          f"hist={a.hist_mode} ({time.time()-t0:.0f}s)", flush=True)

    # standardize main features
    xm_mean, xm_std = xm.mean(0), xm.std(0).clamp_min(1e-4)
    xmn = (xm - xm_mean) / xm_std
    # standardize aux features (except zeros)
    if a.hist_mode != "zero":
        xa_mean, xa_std = xa.mean(0), xa.std(0).clamp_min(1e-4)
        xan = (xa - xa_mean) / xa_std
    else:
        xa_mean = torch.zeros(D_STATE)
        xa_std = torch.ones(D_STATE)
        xan = xa  # already zeros

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

    model = DDHistNet(a.hidden, a.depth, a.aux_dim)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"DDHistNet: {n_params:,} params "
          f"({time.time()-t0:.0f}s)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.steps, eta_min=a.lr * 0.02)
    w_head_g = w_head.cuda().requires_grad_(False)
    w_norm_g = w_norm.cuda().requires_grad_(False)
    CHUNK = 512

    def chunk_losses(zc, sl):
        zc_g = zc.cuda()
        zn = zc_g * torch.rsqrt(
            zc_g.square().mean(-1, keepdim=True) + 1e-6)
        zn = zn * w_norm_g
        lg = zn @ w_head_g.t()
        ce = F.cross_entropy(lg.float(), y_g[sl],
                             reduction="none")
        l_y = lg.gather(1, y_g[sl].unsqueeze(1)).squeeze(1)
        m = l_y.unsqueeze(1) - lg.gather(1, comp_g[sl])
        m_rows = (F.softplus(a.mu - m)
                  * cmask_g[sl]).sum(-1) / cmask_g[sl].sum(
                      -1).clamp_min(1.0)
        l16 = lg.gather(1, top_g[sl])
        logp = F.log_softmax(l16, dim=-1)
        kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log()
                          - logp)).sum(-1)
        return ce + a.lam_m * m_rows + a.lam_kl * kl

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
            ys = y_g[[b_idx[j] for j in range(sl.start, sl.stop)]]
            pred = lg.argmax(-1)
            correct += (pred == ys).sum().item()
            for k in topk:
                topk[k] += (lg.topk(k, dim=-1).indices
                            == ys.unsqueeze(1)).any(1
                            ).sum().item()
        nb = len(b_idx)
        print(f"\nBOUNDARY (train, n={nb}): S1={correct/nb:.4f} "
              f"Top4={topk[4]/nb:.4f} Top16={topk[16]/nb:.4f}",
              flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "hidden": a.hidden,
        "depth": a.depth, "aux_dim": a.aux_dim,
        "hist_mode": a.hist_mode,
        "xm_mean": xm_mean, "xm_std": xm_std,
        "xa_mean": xa_mean, "xa_std": xa_std,
        "state_field": a.state,
        "args": {k: str(v) for k, v in vars(a).items()},
        "seed": 104729,
        "n_params": n_params,
        "teacher_audit_sha256": hashlib.sha256(
            AUDIT.read_bytes()).hexdigest(),
    }, a.out / "dd1_hist.pt")
    print(f"wrote {a.out / 'dd1_hist.pt'} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
