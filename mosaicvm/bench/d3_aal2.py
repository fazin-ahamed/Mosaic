"""AAL-2 — verification-exposed on-policy drift repair of ps1001.

The ps decision maps an A2-PREDICTED continuation state -> token. At
inference A2's state has drifted AND the driver token is ps's own draft.
Offline teacher training only ever saw gold/os/chain states built with
TRUE-token drivers, so ps never trained on the states its own wrong drafts
induce. That is the exposure gap the CAP-DEV70 UNRANKABLE tail reflects.

Exact-semantics label rule: the target is deterministic greedy over the
committed context, so the accepted token at every within-block position is
the GOLD token toks[t+1] regardless of the draft prefix. Therefore the
correct on-policy label is gold, and NO new source capture is needed.

Loop (DAgger / scheduled sampling, on-policy):
  1. current ps + A2 roll K positions from the exact committed anchor,
     feeding ps's own drafts back as drivers -> drift-induced states.
  2. record (bridge feature, gold token, depth j, alive-before-j).
  3. train ps CE(gold) weighted by prefix survival (only reachable
     positions) + small anti-forget floor + light restricted KL anchor.
  4. refresh states with the updated ps every REFRESH steps.

Reuses HIST1001-free path (ps is state->token), A2 rollout, gold labels,
embed/head. No capture, no CAP set, no root retrain. Starts from ps1001.
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

from mosaicvm.bench.d3_bridge import (BridgeNet, load_head, load_embed,
                                      make_bridge_feat)
from mosaicvm.bench.d3_v1 import make_feat, HORIZON
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_ws import load_a2

TRAIN = Path("artifacts/d3_scale1000_train")
PS1001 = TRAIN / "ps1001.pt"
A2 = "artifacts/d3_v1_A2/student.pt"
CKPT = TRAIN / "aal2_ckpt"
K_ROLL = 8            # exposure horizon (d2-4 + margin)
GAMMA = 0.7           # early-position emphasis
FLOOR = 0.1           # anti-forget weight for unreachable positions
LAM_KL = 0.1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refreshes", type=int, default=4)
    p.add_argument("--steps-per", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=str(TRAIN / "ps1001_aal2.pt"))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    CKPT.mkdir(parents=True, exist_ok=True)

    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(Path(A2))   # CPU (state math)
    embed_g = load_embed()                                # CPU
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda().requires_grad_(False)
    w_norm_g = w_norm.cuda().requires_grad_(False)

    ck = torch.load(PS1001, map_location="cpu", weights_only=True)
    model = BridgeNet(1024, 3); model.load_state_dict(ck["model"])
    model = model.cuda()
    x_mean = ck["x_mean"].cuda(); x_std = ck["x_std"].cuda()

    # teacher top16 for gold-state restricted KL anchor; keyed by (cat,j)
    tea = {}
    spec = json.loads(Path(
        "artifacts/d3_scale1000/prompts_scale1000.json").read_text())
    for cat in spec["splits"]["train"]:
        rec = torch.load(Path("artifacts/d3_scale1000_teacher") / f"{cat}.pt",
                         map_location="cpu", weights_only=True)
        for j, row in enumerate(rec["decisions"]):
            tea[(cat, j)] = (row["gold"]["top_idx"],
                             F.softmax(torch.tensor(row["gold"]["top_val"]), 0))

    caps = load_captures("artifacts/d3_scale1000_data",
                         json.loads(Path(
                             "artifacts/d3_scale1000/prompts_scale1000.json"
                         ).read_text()), None, HORIZON, require_both=False)
    train = [c for c in caps if c["split"] == "train"]
    print(f"{len(train)} train prompts ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def ps_argmax(bf):
        z = model((bf.cuda() - x_mean) / x_std).squeeze(0)
        zn = z * torch.rsqrt(z.square().mean() + 1e-6) * w_norm_g
        return int((w_head_g @ zn).argmax())

    @torch.no_grad()
    def rollout():
        """Collect drift-induced (feature, gold, cat, j, alive) rows. CPU."""
        feats, golds, cats, js, alive = [], [], [], [], []
        for c in train:
            S = c["S"]; K = min(K_ROLL, c["h40"].shape[0] - S - 1)
            if K < 1:
                continue
            toks = c["tokens"]; h = c["h40"].float()
            state = h[S - 1].flatten(); prev = h[S - 2].flatten()
            cur_tok = toks[S - 1]; driver = toks[S]
            live = True
            for j in range(K):
                t = S + j
                fa = make_feat(embed_g, state, state - prev,
                               driver, cur_tok, t)
                pred = a2(((fa - a2_xm) / a2_xs).unsqueeze(0))[0] \
                    * t_std + t_mean
                new_state = state + pred
                bf = make_bridge_feat(embed_g, new_state, driver, cur_tok, t)
                feats.append(bf); golds.append(toks[t + 1])
                cats.append(c["category"]); js.append(j)
                alive.append(1 if live else 0)
                draft = ps_argmax(bf)
                if draft != toks[t + 1]:
                    live = False
                cur_tok = driver; driver = draft
                prev = state; state = new_state
        X = torch.stack(feats)
        y = torch.tensor(golds)
        jj = torch.tensor(js); al = torch.tensor(alive, dtype=torch.float32)
        return X, y, jj, al, cats

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total_steps = a.refreshes * a.steps_per
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=a.lr * 0.05)
    CHUNK = 2048
    gstep = 0
    for rf in range(a.refreshes):
        model.eval()
        X, y, jj, al, cats = rollout()
        n = X.shape[0]
        # survival weight: reachable positions weighted gamma^j; unreachable
        # get FLOOR (anti-forget). first-rejection (alive & would-die) is the
        # highest-value repair signal and is naturally alive==1.
        w = torch.where(al > 0, (GAMMA ** jj.float()),
                        torch.full_like(al, FLOOR))
        # teacher top16 anchor tensors
        top = torch.zeros(n, 16, dtype=torch.long)
        tp = torch.zeros(n, 16)
        has = torch.zeros(n)
        for i in range(n):
            key = (cats[i], int(jj[i]))
            if key in tea:
                ti, pv = tea[key]
                top[i] = torch.tensor(ti); tp[i] = pv; has[i] = 1.0
        Xg = X.cuda(); yg = y.cuda(); wg = w.cuda()
        topg = top.cuda(); tpg = tp.cuda(); hasg = has.cuda()
        alive_frac = al.mean().item()
        print(f"refresh {rf}: {n} rows, alive_frac {alive_frac:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        model.train()
        for s in range(a.steps_per):
            gstep += 1
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for i in range(0, n, CHUNK):
                sl = slice(i, min(i + CHUNK, n))
                z = model((Xg[sl] - x_mean) / x_std)
                zn = z * torch.rsqrt(z.square().mean(-1, keepdim=True)
                                     + 1e-6) * w_norm_g
                lg = zn @ w_head_g.t()
                ce = F.cross_entropy(lg.float(), yg[sl], reduction="none")
                l16 = lg.gather(1, topg[sl])
                logp = F.log_softmax(l16, dim=-1)
                kl = (tpg[sl] * (tpg[sl].clamp_min(1e-9).log() - logp)
                      ).sum(-1) * hasg[sl]
                row = ce + LAM_KL * kl
                wgt = wg[sl]
                part = (row * wgt).sum() / wg.sum()
                part.backward()
                tot += part.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if gstep % 25 == 0:
                print(f"  step {gstep}: L {tot:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        torch.save({"step": gstep, "model": model.state_dict(),
                    "opt": opt.state_dict()}, CKPT / "aal2_ckpt.pt")

    model.eval()
    torch.save({"model": model.state_dict(), "hidden": 1024, "depth": 3,
                "x_mean": ck["x_mean"], "x_std": ck["x_std"],
                "init_from": str(PS1001), "seed": 104729,
                "args": {k: str(v) for k, v in vars(a).items()},
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest()},
               Path(a.out))
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
