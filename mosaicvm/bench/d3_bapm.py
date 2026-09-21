"""BAPM — Branch-Aware Prefix Margin fine-tune of ps1001.

Polishes the frozen continuation model for the EXACT deployed policy
(root greedy, d2 greedy, d3 top-4 branch, d4+ greedy) instead of
full-vocab CE/KL. Starts from ps1001, reuses the cached normalized
union (ps_cache/xn.f32 + meta.pt) — no new capture, no new teacher.

Per (prompt, variant) group of K ordered positions j=0..K-1:
  candidate set C_j = {y} U teacher_top16   (<=17 ids, cached in meta)
  sparse logits = z_norm @ w_head[C_j]^T    (only ~17 output rows)
  greedy depths (j != 2): margin m_j = z_y - max_{c!=y} z_c   (top1)
  branch depth (j == 2): m_j = z_y - r4      (r4 = 4th-highest wrong)
  survival rho_j = prod_{i<j} stopgrad(sigmoid(m_i/tau))
  L_BAPM = sum_j rho_j * softplus(-m_j/tau)
  + lam_anchor * KL(teacher_p16 || student_p16)  (restricted)

j==2 is the 3rd speculative position = the d3 branch depth in the probe.

Staged: 100-step pilot; extend only if dEA >= 0.10 on burned dev sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_bridge import BridgeNet, load_head
from mosaicvm.bench.d3_v1 import POS_DIM

TRAIN = Path("artifacts/d3_scale1000_train")
CACHE = TRAIN / "ps_cache"
PS1001 = TRAIN / "ps1001.pt"
CKPT = TRAIN / "bapm_ckpt"
DIM = 4 * 4096 + 2 * 4096 + POS_DIM
BRANCH_J = 2          # 3rd speculative position = d3 branch depth
BRANCH_K = 4          # d3 top-4
MEM_FLOOR_GB = 3.0


def open_mmap(path, rows, cols):
    t = torch.from_file(str(path), shared=True, size=rows * cols,
                        dtype=torch.float32)
    return t.view(rows, cols)


def mem_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return float("inf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lam-anchor", type=float, default=0.2)
    p.add_argument("--group-batch", type=int, default=128)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=str(TRAIN / "ps1001_bapm.pt"))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    CKPT.mkdir(parents=True, exist_ok=True)

    man = json.loads((CACHE / "manifest.json").read_text())
    n = man["n"]
    xn = open_mmap(CACHE / "xn.f32", n, DIM)
    meta = torch.load(CACHE / "meta.pt", weights_only=False)
    assert len(meta) == n
    st = torch.load(CACHE / "norm_stats.pt", weights_only=True)

    # group rows by (cat, variant), ordered by j
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["cat"], m["variant"])].append(i)
    for k in groups:
        groups[k].sort(key=lambda i: meta[i]["j"])
    group_list = list(groups.values())
    print(f"{n} rows, {len(group_list)} groups ({time.time()-t0:.0f}s)",
          flush=True)

    # candidate ids per row: {y} U top16 (dedup, y first)
    cand_ids = []
    cand_teacher_p = []   # teacher prob over its top16 aligned to cand slots
    for m in meta:
        y = m["y"]
        top = m["top_idx"]           # 16 ids, teacher order
        pt = m["p_t"]                # softmax over top_val (16)
        ids = [y] + [t for t in top if t != y]
        # teacher prob per candidate id (0 for y if y not in top... y IS top of gold)
        pmap = {int(t): float(pt[k]) for k, t in enumerate(top)}
        probs = [pmap.get(int(c), 0.0) for c in ids]
        cand_ids.append(ids)
        cand_teacher_p.append(probs)
    maxC = max(len(c) for c in cand_ids)
    cand = torch.zeros(n, maxC, dtype=torch.long)
    cmask = torch.zeros(n, maxC)
    tprob = torch.zeros(n, maxC)
    for i, ids in enumerate(cand_ids):
        cand[i, :len(ids)] = torch.tensor(ids)
        cmask[i, :len(ids)] = 1.0
        tprob[i, :len(cand_teacher_p[i])] = torch.tensor(cand_teacher_p[i])
    cand_g = cand.cuda(); cmask_g = cmask.cuda(); tprob_g = tprob.cuda()
    jvec = torch.tensor([m["j"] for m in meta])

    w_head, w_norm = load_head()
    w_head_g = w_head.cuda().requires_grad_(False)     # [V,4096]
    w_norm_g = w_norm.cuda().requires_grad_(False)

    model = BridgeNet(1024, 3)  # H1024 D3
    ck = torch.load(PS1001, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.steps, eta_min=a.lr * 0.05)

    start = 1
    cp = CKPT / "bapm_ckpt.pt"
    if cp.exists():
        c = torch.load(cp, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"]); start = c["step"] + 1
        print(f"resumed step {c['step']}", flush=True)

    def group_loss(gidx_batch):
        """One batch of groups -> scalar loss. Survival within group."""
        rows = []
        for g in gidx_batch:
            rows.extend(g)
        rows_t = torch.tensor(rows)
        xb = xn[rows_t].cuda()                         # [R, DIM]
        z = model(xb)                                  # [R, 4096]
        zn = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6) \
            * w_norm_g
        # sparse candidate logits: gather head rows per row
        c_ids = cand_g[rows_t.cuda()]                  # [R, maxC]
        Wc = w_head_g[c_ids]                           # [R, maxC, 4096]
        lg = torch.einsum("rd,rcd->rc", zn, Wc)        # [R, maxC]
        lg = lg.masked_fill(cmask_g[rows_t.cuda()] == 0, -1e9)
        # slot 0 is y; competitors slots 1..
        z_y = lg[:, 0]
        comp = lg[:, 1:]
        jr = jvec[rows_t].cuda()
        # top1 margin (default)
        r1 = comp.max(dim=1).values
        # top4 margin at branch depth: 4th-highest wrong
        k4 = min(BRANCH_K, comp.shape[1])
        r4 = comp.topk(k4, dim=1).values[:, -1]
        rsel = torch.where(jr == BRANCH_J, r4, r1)
        m = z_y - rsel                                 # [R]
        # anchor KL over candidate slots (restricted)
        logp = F.log_softmax(lg, dim=-1)
        tp = tprob_g[rows_t.cuda()]
        tp = tp / tp.sum(-1, keepdim=True).clamp_min(1e-9)
        anchor = (tp * (tp.clamp_min(1e-9).log() - logp)).sum(-1)
        # per-row base loss
        ell = F.softplus(-m / a.tau)
        # survival weighting within each group (cumulative prod of prior a_i)
        surv = torch.ones_like(m)
        off = 0
        a_i = torch.sigmoid(m.detach() / a.tau)
        for g in gidx_batch:
            L = len(g)
            sl = slice(off, off + L)
            # prefix product of a_i within this group's ordered rows
            av = a_i[sl]
            pref = torch.cumprod(
                torch.cat([torch.ones(1, device=av.device), av[:-1]]), 0)
            surv[sl] = pref
            off += L
        loss = (surv * ell).mean() + a.lam_anchor * anchor.mean()
        return loss

    gorder = list(range(len(group_list)))
    for step in range(start, a.steps + 1):
        av = mem_avail_gb()
        if av < MEM_FLOOR_GB:
            torch.save({"step": step - 1, "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict()}, cp)
            print(f"WATCHDOG avail {av:.1f}GB ckpt {step-1} exit", flush=True)
            return
        torch.manual_seed(104729 + step)
        perm = torch.randperm(len(group_list)).tolist()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        nb = 0
        for i in range(0, len(perm), a.group_batch):
            gb = [group_list[j] for j in perm[i:i + a.group_batch]]
            part = group_loss(gb) * (len(gb) / len(group_list))
            part.backward()
            total += part.item()
            nb += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 25 == 0 or step == a.steps or step == 1:
            print(f"step {step}: L {total:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        if step % 50 == 0:
            torch.save({"step": step, "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict()}, cp)

    model.eval()
    torch.save({"model": model.state_dict(), "hidden": 1024, "depth": 3,
                "x_mean": st["mean"], "x_std": st["std"],
                "init_from": str(PS1001), "seed": 104729,
                "args": {k: str(v) for k, v in vars(a).items()},
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest()},
               Path(a.out))
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
