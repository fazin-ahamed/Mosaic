"""DD-1: First-Decision Specialist (roadmap S27-S36).

Predicts ONLY the first speculative decision from the EXACT
committed root state. No recurrence, no future-state regression,
no h40 prediction. The model maps:

  (exact h42[S-1], root token, prev token, position)
  → z ∈ R^4096 → frozen FP32 head → logits

Training uses ALL positions (each position t is a valid
"state at t-1 → predict token at t+1" example) for data
efficiency; evaluation reports the boundary-position S1 as the
primary metric plus transition-level Top-k.

Two input variants raced (S34):
  A: h42 (exact final state — the state that produced the root)
  B: h40 (exact layer-40 state — pre-tail)

Full-vocab CE + margin + KL (the restricted-objective negative
control is permanent). Chunked GPU scoring as in d3_bridge.
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

from mosaicvm.bench.d3_bridge import (BridgeNet, load_head,
                                      load_embed, TEACHER, AUDIT)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON, pos_encode

SNAP = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--"
    "DeepSeek-V4-Flash-0731/snapshots/"
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062")


def make_dd_feat(embed, state_flat, driver_tok, cur_tok, pos):
    """DD-1 feature: exact state + token embeddings + position.
    state_flat: [16384] (h42 or h40, flattened [4,4096])."""
    return torch.cat([
        state_flat,
        embed[driver_tok].float(),
        embed[cur_tok].float(),
        pos_encode(torch.tensor([pos]))[0],
    ])


def build_dd_examples(state_field):
    """All DD-1 examples from the teacher + captures.

    For each position j (0..K-1) in each prompt:
      input state: c[state_field][S+j-1] (the state BEFORE pos S+j)
      driver token: toks[S+j]
      cur token: toks[S+j-1]
      target: toks[S+j+1] (= teacher decision j's true token)
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

    feats, labels, is_boundary = [], [], []
    for cat in sorted(ver["record_hashes_full"]):
        rec = torch.load(TEACHER / f"{cat}.pt",
                         map_location="cpu", weights_only=True)
        cap = cap_by_cat[cat]
        S, K = rec["S"], rec["K"]
        toks = rec["tokens"]
        # state at positions S-1 .. S+K-2 (one BEFORE each spec pos)
        states = cap[state_field][S - 1:S + K - 1].float()
        for j in range(K):
            t = S + j
            row = rec["decisions"][j]
            y = row["true_token"]      # = toks[t+1]
            sf = states[j].flatten()   # state at position t-1
            f = make_dd_feat(embed, sf, toks[t], toks[t - 1], t)
            feats.append(f)
            labels.append({
                "y": y, "cat": cat, "j": j,
                "top_idx": row["gold"]["top_idx"],
                "top_val": row["gold"]["top_val"],
                "comps": [i for i in row["gold"]["top_idx"]
                          if i != y][:8],
                "is_boundary": j == 0,
            })
            is_boundary.append(j == 0)
    x = torch.stack(feats)
    return x, labels, embed, is_boundary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--state", choices=("h42", "h40"),
                   default="h42",
                   help="which exact state to use as input")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--lam-m", type=float, default=1.0)
    p.add_argument("--lam-kl", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--boundary-mass", type=float, default=None,
                   help="group-mass boundary weighting: fraction "
                        "of the TOTAL optimization objective on "
                        "boundary (j=0) rows. None = natural "
                        "84/2688. Weights apply to the COMPLETE "
                        "per-row objective (CE+margin+KL).")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)

    t0 = time.time()
    x, labels, embed, is_boundary = build_dd_examples(a.state)
    n = x.shape[0]
    n_boundary = sum(is_boundary)
    print(f"{n} examples ({n_boundary} boundary / {n} total), "
          f"state={a.state} ({time.time()-t0:.0f}s)", flush=True)
    x_mean, x_std = x.mean(0), x.std(0).clamp_min(1e-4)
    xn = (x - x_mean) / x_std

    w_head, w_norm = load_head()
    print(f"head loaded ({time.time()-t0:.0f}s)", flush=True)

    # labels
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

    y_g = y_idx.cuda()
    comp_g = comp_idx.cuda()
    cmask_g = comp_mask.cuda()
    top_g = top_idx_t.cuda()
    pt_g = p_t.cuda()

    # ---- boundary group-mass weighting ----
    n_b = sum(is_boundary)
    n_o = n - n_b
    if a.boundary_mass is not None:
        gb = a.boundary_mass
        if not 0 < gb < 1:
            raise ValueError("boundary_mass in (0,1)")
        row_w = torch.zeros(n)
        b_idx = torch.tensor([i for i, b in
                              enumerate(is_boundary) if b])
        o_idx = torch.tensor([i for i, b in
                              enumerate(is_boundary) if not b])
        row_w[b_idx] = gb / n_b
        row_w[o_idx] = (1 - gb) / n_o
        row_w = row_w * n / row_w.sum()  # mean weight 1
        realized = row_w[b_idx].sum() / row_w.sum()
        print(f"boundary-mass {gb}: n_b={n_b} n_o={n_o} "
              f"mean_b_w={row_w[b_idx].mean():.4f} "
              f"mean_o_w={row_w[o_idx].mean():.4f} "
              f"realized={realized:.4f}", flush=True)
        if abs(realized.item() - gb) > 0.01:
            raise RuntimeError(
                f"realized boundary mass {realized:.4f} != "
                f"requested {gb}")
    else:
        row_w = torch.ones(n)
        print(f"boundary-mass: natural ({n_b}/{n} = "
              f"{n_b/n:.4f})", flush=True)
    row_w_g = row_w.cuda()

    model = BridgeNet(a.hidden, a.depth)  # same arch as bridge
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
            zc = model(xn[sl])
            rl = chunk_losses(zc, sl)      # per-row combined
            w = row_w_g[sl]                 # group-mass weights
            part = (rl * w).sum() / w.sum() \
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
                    zg = model(xn[sl]).cuda()
                    zn = zg * torch.rsqrt(
                        zg.square().mean(-1, keepdim=True)
                        + 1e-6) * w_norm_g
                    lg = zn @ w_head_g.t()
                    accs.append((lg.argmax(-1) == y_g[sl])
                                .float().mean().item())
            print(f"step {step}: L {total:.4f} "
                  f"train-argmax {sum(accs)/len(accs):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- boundary evaluation (S1's decision) ----
    model.eval()
    with torch.no_grad():
        b_idx = [i for i, b in enumerate(is_boundary) if b]
        bx = xn[b_idx]
        correct = 0
        topk = {1: 0, 4: 0, 16: 0}
        for i in range(0, len(b_idx), 64):
            sl = slice(i, min(i + 64, len(b_idx)))
            zg = model(bx[sl]).cuda()
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
        s1_train = correct / nb
        print(f"\nBOUNDARY (train split, n={nb}): "
              f"S1={s1_train:.4f} "
              f"Top4={topk[4]/nb:.4f} "
              f"Top16={topk[16]/nb:.4f}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "hidden": a.hidden,
        "depth": a.depth, "x_mean": x_mean, "x_std": x_std,
        "state_field": a.state,
        "boundary_mass": a.boundary_mass,
        "args": {k: str(v) for k, v in vars(a).items()},
        "seed": 104729,
        "train_boundary_S1": round(s1_train, 4),
        "teacher_audit_sha256": hashlib.sha256(
            AUDIT.read_bytes()).hexdigest(),
    }, a.out / "dd1.pt")
    print(f"wrote {a.out / 'dd1.pt'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
