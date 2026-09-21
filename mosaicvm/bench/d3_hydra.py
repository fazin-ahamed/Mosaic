"""Targeted Hydra-D2/D3/D4 continuation heads (direct token prediction).

Replaces the ps recurrence at speculative depths 2,3,4 with sequentially
dependent heads conditioned on the EXACT committed root state + the actual
preceding speculative token embeddings — no synthetic future-state drift.

  q_d = Head_d( committed_root_feat, emb(y_1..y_{d-1}) )   d in {2,3,4}
  token = argmax( head_geometry(q_d) )   (full source head at eval)

Direct token prediction (EAGLE-3 lesson); shared trunk + depth adapters;
frozen source token embeddings + frozen head geometry. Teacher-forced
preceding tokens in v1. Trains on the 1001 union with a prompt-level 80/20
split to detect memorization (ps1001 memorized train).

Depth indexing: speculative position j (0-based). Hydra owns j=1,2,3
(= d2,d3,d4). j=0 stays the frozen HIST/dd1 first decision.
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

from mosaicvm.bench.d3_bridge import load_head, load_embed
from mosaicvm.bench.d3_v1 import pos_encode, POS_DIM
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON

TRAIN = Path("artifacts/d3_scale1000_train")
CKPT = TRAIN / "hydra_ckpt"
D_STATE = 4 * 4096
D_TOK = 4096
DEPTHS = (1, 2, 3)          # speculative j owned by Hydra = d2,d3,d4
MAXPREV = 3                 # max preceding tokens conditioned on


class HydraHeads(nn.Module):
    """Shared trunk over (committed root feat + prev-token sum + depth),
    depth-specific adapter -> 4096 decision vector."""

    def __init__(self, hidden=1024, depth=3):
        super().__init__()
        d_in = D_STATE + D_TOK + POS_DIM + 8   # root + prevsum + pos + depth-onehot
        self.inp = nn.Linear(d_in, hidden)
        self.mid = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth - 1)])
        self.adapters = nn.ModuleDict(
            {str(d): nn.Linear(hidden, hidden) for d in DEPTHS})
        self.head = nn.Linear(hidden, 4096)
        self.act = nn.GELU()

    def forward(self, root_feat, prev_sum, pos, depth_id):
        d1h = F.one_hot(depth_id, num_classes=8).float()
        x = torch.cat([root_feat, prev_sum, pos, d1h], -1)
        h = self.act(self.inp(x))
        for m in self.mid:
            h = h + self.act(m(h))
        # depth adapter (per-row depth): gather via loop-free trick
        out = torch.zeros_like(h)
        for d in DEPTHS:
            mask = (depth_id == d)
            if mask.any():
                out[mask] = self.act(self.adapters[str(d)](h[mask]))
        return self.head(h + out)


def build_rows(caps, teacher_dir, embed):
    """Rows: (root_feat[16384], prev_sum[4096], pos, depth_id, gold).
    prev tokens are the TRUE tokens (teacher forcing v1)."""
    roots, prevs, poss, depths, golds, catrow = [], [], [], [], [], []
    for c in caps:
        cat = c["category"]
        rec = torch.load(Path(teacher_dir) / f"{cat}.pt",
                         map_location="cpu", weights_only=True)
        S, K = rec["S"], rec["K"]
        toks = rec["tokens"]
        root = c["h40"][S - 1].float().flatten()      # committed root state
        for j in DEPTHS:
            if j >= K:
                continue
            t = S + j
            gold = rec["decisions"][j]["true_token"]
            # preceding speculative tokens = true tokens toks[S+1..S+j]
            prev_ids = toks[S + 1: S + 1 + j][-MAXPREV:]
            ps = sum(embed[pi].float() for pi in prev_ids) \
                if prev_ids else torch.zeros(D_TOK)
            roots.append(root); prevs.append(ps)
            poss.append(pos_encode(torch.tensor([t]))[0])
            depths.append(j); golds.append(gold); catrow.append(cat)
    return (torch.stack(roots), torch.stack(prevs), torch.stack(poss),
            torch.tensor(depths), torch.tensor(golds), catrow)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=str(TRAIN / "hydra.pt"))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    CKPT.mkdir(parents=True, exist_ok=True)

    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda().requires_grad_(False)
    w_norm_g = w_norm.cuda().requires_grad_(False)

    # 1001 union: orig84 + scale126/490/301 (teacher dirs with gold decisions)
    tiers = [("artifacts/d3_data_105", "artifacts/stage1_prompts.json",
              "artifacts/d3_r2_teacher"),
             ("artifacts/d3_scale_data", "artifacts/d3_scale/prompts_scale.json",
              "artifacts/d3_scale_teacher"),
             ("artifacts/d3_scale700_data",
              "artifacts/d3_scale700/prompts_scale700.json",
              "artifacts/d3_scale700_teacher"),
             ("artifacts/d3_scale1000_data",
              "artifacts/d3_scale1000/prompts_scale1000.json",
              "artifacts/d3_scale1000_teacher")]
    caps = []
    for data, man, td in tiers:
        spec = json.loads(Path(man).read_text())
        cc = load_captures(data, spec, None, HORIZON, require_both=False)
        for c in cc:
            if c["split"] == "train":
                c["_teacher"] = td
                caps.append(c)
    print(f"union prompts: {len(caps)} ({time.time()-t0:.0f}s)", flush=True)

    # prompt-level 80/20 split, domain-stratified
    import collections, random
    by_dom = collections.defaultdict(list)
    for c in caps:
        by_dom[c["category"].rsplit("_", 1)[0]].append(c)
    rng = random.Random(104729)
    train_caps, val_caps = [], []
    for dom, cs in by_dom.items():
        rng.shuffle(cs)
        k = int(round(len(cs) * 0.8))
        train_caps += cs[:k]; val_caps += cs[k:]
    print(f"train {len(train_caps)} / holdout {len(val_caps)} prompts "
          f"({time.time()-t0:.0f}s)", flush=True)

    def rows_for(cs):
        # group by teacher dir
        out = [[], [], [], [], [], []]
        for td in set(c["_teacher"] for c in cs):
            sub = [c for c in cs if c["_teacher"] == td]
            r = build_rows(sub, td, embed)
            for i in range(6):
                if i < 5:
                    out[i].append(r[i])
                else:
                    out[i].extend(r[i])
        return (torch.cat(out[0]), torch.cat(out[1]), torch.cat(out[2]),
                torch.cat(out[3]), torch.cat(out[4]), out[5])

    Rtr = rows_for(train_caps)
    Rvl = rows_for(val_caps) if a.stage == "pilot" else None
    ntr = Rtr[0].shape[0]
    print(f"train rows {ntr} ({time.time()-t0:.0f}s)", flush=True)

    # normalize root feat only (large); prev/pos are bounded
    rmean = Rtr[0].mean(0); rstd = Rtr[0].std(0).clamp_min(1e-4)

    model = HydraHeads(1024, 3).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.steps, eta_min=a.lr * 0.05)
    rmean_g = rmean.cuda(); rstd_g = rstd.cuda()
    root_g = Rtr[0].cuda(); prev_g = Rtr[1].cuda(); pos_g = Rtr[2].cuda()
    dep_g = Rtr[3].cuda(); gold_g = Rtr[4].cuda()
    CHUNK = 1024

    def logits_of(rf, pv, po, dp):
        z = model((rf - rmean_g) / rstd_g, pv, po, dp)
        zn = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6) \
            * w_norm_g
        return zn @ w_head_g.t()

    @torch.no_grad()
    def eval_split(R, tag):
        rf = R[0].cuda(); pv = R[1].cuda(); po = R[2].cuda()
        dp = R[3].cuda(); gd = R[4].cuda()
        accs = {}
        for d in DEPTHS:
            m = (dp == d)
            if not m.any():
                continue
            lg = logits_of(rf[m], pv[m], po[m], dp[m])
            top1 = (lg.argmax(-1) == gd[m]).float().mean().item()
            # top4 membership
            t4 = lg.topk(4, -1).indices
            top4 = (t4 == gd[m].unsqueeze(1)).any(1).float().mean().item()
            accs[f"d{d+1}"] = {"top1": round(top1, 3), "top4": round(top4, 3)}
        print(f"  [{tag}] {accs} ({time.time()-t0:.0f}s)", flush=True)
        return accs

    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(0, ntr, CHUNK):
            sl = slice(i, min(i + CHUNK, ntr))
            lg = logits_of(root_g[sl], prev_g[sl], pos_g[sl], dep_g[sl])
            ce = F.cross_entropy(lg.float(), gold_g[sl])
            part = ce * ((sl.stop - sl.start) / ntr)
            part.backward(); tot += part.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 25 == 0 or step == 1:
            print(f"step {step}: L {tot:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
            if Rvl is not None:
                model.eval()
                eval_split(Rtr, "train")
                eval_split(Rvl, "holdout")
                model.train()
            torch.save({"step": step, "model": model.state_dict()},
                       CKPT / "hydra_ckpt.pt")

    model.eval()
    torch.save({"model": model.state_dict(), "hidden": 1024, "depth": 3,
                "rmean": rmean, "rstd": rstd, "depths": list(DEPTHS),
                "maxprev": MAXPREV, "stage": a.stage, "seed": 104729,
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest()},
               Path(a.out))
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
