"""DepthScale-1001: ps continuation trained on the 1001-prompt union.

Frozen recipe (identical to artifacts/d3_bridge_r3b/bridge_ps):
  BridgeNet H1024/D3, 800 steps, AdamW lr 1e-3 wd 1e-4, cosine (eta 0.02),
  mu 1.0, lam_m 1.0, lam_kl 0.5, gold_frac 0.25, pos_weight strong,
  CHUNK 512, clip 1.0, seed 104729, full-vocab CE + margin + KL.

ONLY the training corpus changes: orig84 (r2 teacher) + scale126 + scale490
+ scale301, all with h40 gold/os/chain variants -> 1001 prompts, 3 variants
per position. FP32 out-of-core mmap backend (same as d3_scale1000_train):
raw+normalized features live on disk; anon RSS stays bounded. cgroup +
checkpoint + watchdog for reboot safety.

Output: artifacts/d3_scale1000_train/ps1001.pt (drop-in for splice/eval as
the ps continuation checkpoint).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_bridge import (BridgeNet, load_head, load_embed,
                                      make_bridge_feat, TEACHER, AUDIT)
from mosaicvm.bench.d3_v1 import HORIZON, POS_DIM

OUTDIR = Path("artifacts/d3_scale1000_train")
CACHE = OUTDIR / "ps_cache"
CKPT = OUTDIR / "ps_ckpt"
D_STATE = 4 * 4096
D_TOK = 4096
DIM = D_STATE + 2 * D_TOK + POS_DIM   # 24640
MEM_FLOOR_GB = 3.0
CKPT_EVERY = 50

# (gold/aug teacher dir, manifest, uses_r2_layout)
TIERS = [
    ("artifacts/d3_r2_teacher", "artifacts/stage1_prompts.json", True),
    ("artifacts/d3_scale_teacher_aug", "artifacts/d3_scale/prompts_scale.json", False),
    ("artifacts/d3_scale700_teacher_aug",
     "artifacts/d3_scale700/prompts_scale700.json", False),
    ("artifacts/d3_scale1000_teacher_aug",
     "artifacts/d3_scale1000/prompts_scale1000.json", False),
]
VARIANTS = ["h40_gold_future", "h40_pred_os", "h40_pred_chain"]
VAR_TAG = {"h40_gold_future": "future", "h40_pred_os": "os",
           "h40_pred_chain": "chain"}


def open_mmap(path, rows, cols):
    t = torch.from_file(str(path), shared=True, size=rows * cols,
                        dtype=torch.float32)
    return t.view(rows, cols)


def tier_cats(tdir, manifest, is_r2):
    if is_r2:
        ver = json.loads(AUDIT.read_text())
        assert ver["verdict"] == "PASS"
        return sorted(ver["record_hashes_full"])
    spec = json.loads(Path(manifest).read_text())
    return sorted(spec["splits"]["train"])


def count_rows():
    n = 0
    per = {}
    for tdir, manifest, is_r2 in TIERS:
        tdir = Path(tdir)
        cats = tier_cats(tdir, manifest, is_r2)
        c = 0
        for cat in cats:
            rec = torch.load(tdir / f"{cat}.pt", map_location="cpu",
                             weights_only=True)
            c += rec["K"] * len(VARIANTS)
        per[tdir.name] = c
        n += c
    return n, per


def build(embed):
    CACHE.mkdir(parents=True, exist_ok=True)
    man = CACHE / "manifest.json"
    prov = hashlib.sha256()
    for tdir, manifest, is_r2 in TIERS:
        prov.update(Path(manifest).read_bytes())
    prov = prov.hexdigest()
    n, per = count_rows()
    if (man.exists() and (CACHE / "x.f32").exists()
            and (CACHE / "meta.pt").exists()):
        m = json.loads(man.read_text())
        if m.get("prov") == prov and m.get("n") == n:
            x = open_mmap(CACHE / "x.f32", n, DIM)
            meta = torch.load(CACHE / "meta.pt", weights_only=False)
            print(f"reuse cached ps union: {n} rows {per}", flush=True)
            return x, meta

    x = open_mmap(CACHE / "x.f32", n, DIM)
    meta = []
    off = 0
    for tdir, manifest, is_r2 in TIERS:
        tdir = Path(tdir)
        cats = tier_cats(tdir, manifest, is_r2)
        for cat in cats:
            rec = torch.load(tdir / f"{cat}.pt", map_location="cpu",
                             weights_only=True)
            S, K = rec["S"], rec["K"]
            toks = rec["tokens"]
            for var in VARIANTS:
                states = rec[var]  # [K,4,4096]
                for j in range(K):
                    t = S + j
                    row = rec["decisions"][j]
                    y = row["true_token"]
                    state = states[j].flatten().float()
                    f = make_bridge_feat(embed, state, toks[t],
                                         toks[t - 1], t)
                    x[off] = f
                    comps = [i for i in row["gold"]["top_idx"]
                             if i != y][:8]
                    p_t = F.softmax(torch.tensor(row["gold"]["top_val"]),
                                    dim=0)
                    meta.append({"y": y, "comps": comps,
                                 "top_idx": row["gold"]["top_idx"],
                                 "p_t": p_t, "variant": VAR_TAG[var],
                                 "j": j, "cat": cat})
                    off += 1
        print(f"  built {tdir.name} (total {off})", flush=True)
    assert off == n, (off, n)
    torch.save(meta, CACHE / "meta.pt")
    man.write_text(json.dumps({"prov": prov, "n": n, "per": per}, indent=1))
    print(f"ps union built: {n} rows {per}", flush=True)
    return x, meta


def mem_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return float("inf")


def save_ckpt(path, step, model, opt, sched):
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "model": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "py_rng": torch.random.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None)}, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--lam-m", type=float, default=1.0)
    p.add_argument("--lam-kl", type=float, default=0.5)
    p.add_argument("--gold-frac", type=float, default=0.25)
    p.add_argument("--pos-weight", default="strong")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    embed = load_embed()
    x, meta = build(embed)
    n = x.shape[0]

    # mixture weights (gold:pred over combined objective) + pos-weight
    gold_idx = [i for i, m in enumerate(meta) if m["variant"] == "future"]
    pred_idx = [i for i, m in enumerate(meta) if m["variant"] != "future"]
    assert gold_idx and pred_idx
    row_w = torch.zeros(n)
    row_w[gold_idx] = a.gold_frac / len(gold_idx)
    row_w[pred_idx] = (1 - a.gold_frac) / len(pred_idx)
    row_w = row_w * n / row_w.sum()
    if a.pos_weight != "none":
        js = torch.tensor([m["j"] for m in meta], dtype=torch.float32)
        pw = (1.0 / (1.0 + js) ** 2 if a.pos_weight == "strong"
              else 1.0 / (1.0 + js))
        pw = pw / pw.mean()
        row_w = row_w * pw
    row_w_g = row_w.cuda()
    print(f"{n} rows; gold {len(gold_idx)} pred {len(pred_idx)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # normalization (fp32 mmap)
    nman = CACHE / "norm.json"
    if nman.exists() and (CACHE / "xn.f32").exists():
        st = torch.load(CACHE / "norm_stats.pt", weights_only=True)
        x_mean, x_std = st["mean"], st["std"]
        xn = open_mmap(CACHE / "xn.f32", n, DIM)
        print("reuse cached normalized ps union", flush=True)
    else:
        x32 = x
        x_mean = x32.mean(0)
        x_std = x32.std(0).clamp_min(1e-4)
        xn = open_mmap(CACHE / "xn.f32", n, DIM)
        CH = 1024
        for i in range(0, n, CH):
            sl = slice(i, min(i + CH, n))
            xn[sl] = (x[sl] - x_mean) / x_std
        torch.save({"mean": x_mean, "std": x_std},
                   CACHE / "norm_stats.pt")
        nman.write_text(json.dumps({"n": n}))
        print(f"normalized ({time.time()-t0:.0f}s)", flush=True)

    w_head, w_norm = load_head()
    w_head_g = w_head.cuda().requires_grad_(False)
    w_norm_g = w_norm.cuda().requires_grad_(False)
    y_g = torch.tensor([m["y"] for m in meta]).cuda()
    max_c = max(len(m["comps"]) for m in meta)
    comp_g = torch.tensor(
        [m["comps"] + [m["y"]] * (max_c - len(m["comps"])) for m in meta]
    ).cuda()
    cmask_g = torch.tensor(
        [[1.0] * len(m["comps"]) + [0.0] * (max_c - len(m["comps"]))
         for m in meta]).cuda()
    top_g = torch.tensor([m["top_idx"] for m in meta]).cuda()
    pt_g = torch.stack([m["p_t"] for m in meta]).cuda()

    model = BridgeNet(1024, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.steps, eta_min=a.lr * 0.02)

    start = 1
    cp = CKPT / "ps_ckpt.pt"
    if cp.exists():
        c = torch.load(cp, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        torch.random.set_rng_state(c["py_rng"])
        if c.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(c["cuda_rng"])
        start = c["step"] + 1
        print(f"resumed step {c['step']}", flush=True)

    def chunk_losses(zc, sl):
        zc_g = zc.cuda()
        zn = zc_g * torch.rsqrt(zc_g.square().mean(-1, keepdim=True)
                                + 1e-6) * w_norm_g
        lg = zn @ w_head_g.t()
        ce = F.cross_entropy(lg.float(), y_g[sl], reduction="none")
        l_y = lg.gather(1, y_g[sl].unsqueeze(1)).squeeze(1)
        m = l_y.unsqueeze(1) - lg.gather(1, comp_g[sl])
        m_rows = (F.softplus(a.mu - m) * cmask_g[sl]).sum(-1) \
            / cmask_g[sl].sum(-1).clamp_min(1.0)
        l16 = lg.gather(1, top_g[sl])
        logp = F.log_softmax(l16, dim=-1)
        kl = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log() - logp)).sum(-1)
        row = ce + a.lam_m * m_rows + a.lam_kl * kl
        w = row_w_g[sl]
        return (row * w).sum() / w.sum()

    CHUNK = 512
    for step in range(start, a.steps + 1):
        av = mem_avail_gb()
        if av < MEM_FLOOR_GB:
            save_ckpt(cp, step - 1, model, opt, sched)
            print(f"WATCHDOG avail {av:.1f}GB; ckpt {step-1}; exit",
                  flush=True)
            return
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            zc = model(xn[sl])
            part = chunk_losses(zc, sl) * ((sl.stop - sl.start) / n)
            part.backward()
            total += part.item()
            del zc
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % CKPT_EVERY == 0:
            save_ckpt(cp, step, model, opt, sched)
        if step % 25 == 0 or step == a.steps:
            with torch.no_grad():
                accs = []
                for i in range(0, n, 2048):
                    sl = slice(i, min(i + 2048, n))
                    zg = model(xn[sl]).cuda()
                    zn = zg * torch.rsqrt(
                        zg.square().mean(-1, keepdim=True) + 1e-6) * w_norm_g
                    lg = zn @ w_head_g.t()
                    accs.append((lg.argmax(-1) == y_g[sl])
                                .float().mean().item())
            print(f"step {step}: L {total:.4f} argmax "
                  f"{sum(accs)/len(accs):.3f} ({time.time()-t0:.0f}s)",
                  flush=True)

    model.eval()
    torch.save({"model": model.state_dict(), "hidden": 1024, "depth": 3,
                "x_mean": x_mean, "x_std": x_std,
                "args": {k: str(v) for k, v in vars(a).items()},
                "seed": 104729, "n_rows": n, "corpus": "1001",
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest()},
               OUTDIR / "ps1001.pt")
    print(f"wrote {OUTDIR}/ps1001.pt ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
