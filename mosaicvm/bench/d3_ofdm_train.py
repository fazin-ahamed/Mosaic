"""OFDM repair: fine-tune ps1001 on out-of-fold drift states.

50% OOF hard states (ps_old-induced, unseen-prompt, ps1001 true_rank>=4)
+ 50% ps1001 anchor rows (gold states from ps_cache, anti-forget).
Same BridgeNet arch, same base loss (full-vocab CE + restricted KL);
ONLY the state distribution changes. Starts from ps1001.

Staged: 100 steps, checkpoint at 25/50/75/100.
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

from mosaicvm.bench.d3_bridge import BridgeNet, load_head
from mosaicvm.bench.d3_v1 import POS_DIM

TRAIN = Path("artifacts/d3_scale1000_train")
CACHE = TRAIN / "ps_cache"
PS1001 = TRAIN / "ps1001.pt"
BANK = TRAIN / "ofdm_bank.pt"
CKPT = TRAIN / "ofdm_ckpt"
DIM = 4 * 4096 + 2 * 4096 + POS_DIM
LAM_KL = 0.5


def open_mmap(path, rows, cols):
    t = torch.from_file(str(path), shared=True, size=rows * cols,
                        dtype=torch.float32)
    return t.view(rows, cols)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--anchor-frac", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=str(TRAIN / "ps1001_ofdm.pt"))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    CKPT.mkdir(parents=True, exist_ok=True)

    w_head, w_norm = load_head()
    w_head_g = w_head.cuda().requires_grad_(False)
    w_norm_g = w_norm.cuda().requires_grad_(False)
    ck = torch.load(PS1001, map_location="cpu", weights_only=True)
    model = BridgeNet(1024, 3); model.load_state_dict(ck["model"])
    model = model.cuda()
    x_mean = ck["x_mean"].cuda(); x_std = ck["x_std"].cuda()

    # --- OOF hard states (features already in bridge space, unnormalized) ---
    bank = torch.load(BANK, map_location="cpu", weights_only=False)["bank"]
    Xh = torch.stack([b["feature"] for b in bank])         # [H, DIM]
    yh = torch.tensor([b["gold"] for b in bank])
    H = Xh.shape[0]
    # OOF states have no teacher top16 -> CE only (khas=0)
    print(f"OOF hard states: {H} ({time.time()-t0:.0f}s)", flush=True)

    # --- anchor rows: sample gold states from ps_cache ---
    man = json.loads((CACHE / "manifest.json").read_text())
    n = man["n"]
    xn = open_mmap(CACHE / "xn.f32", n, DIM)   # normalized
    meta = torch.load(CACHE / "meta.pt", weights_only=False)
    # gold variant rows only, sample A = round(H*anchor/(1-anchor))
    gold_rows = [i for i, m in enumerate(meta) if m["variant"] == "future"]
    A = min(len(gold_rows),
            round(H * a.anchor_frac / (1 - a.anchor_frac)))
    torch.manual_seed(999)
    sel = torch.tensor(gold_rows)[torch.randperm(len(gold_rows))[:A]]
    Xa = xn[sel]                                # normalized already
    ya = torch.tensor([meta[i]["y"] for i in sel.tolist()])
    top_a = torch.tensor([meta[i]["top_idx"] for i in sel.tolist()])
    tp_a = torch.stack([meta[i]["p_t"] for i in sel.tolist()])
    print(f"anchor rows: {A} ({time.time()-t0:.0f}s)", flush=True)

    # move to GPU; hard states normalized on the fly (raw bridge feat)
    Xh_g = Xh.cuda(); yh_g = yh.cuda()
    Xa_g = Xa.cuda(); ya_g = ya.cuda()
    top_a_g = top_a.cuda(); tp_a_g = tp_a.cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.steps, eta_min=a.lr * 0.05)
    CHUNK = 1024

    def zlogits(xb, normalized):
        xin = xb if normalized else (xb - x_mean) / x_std
        z = model(xin)
        zn = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6) \
            * w_norm_g
        return zn @ w_head_g.t()

    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        # hard states (CE only), raw features -> normalize inside
        for i in range(0, H, CHUNK):
            sl = slice(i, min(i + CHUNK, H))
            lg = zlogits(Xh_g[sl], normalized=False)
            ce = F.cross_entropy(lg.float(), yh_g[sl])
            part = ce * ((sl.stop - sl.start) / (H + A))
            part.backward(); total += part.item()
        # anchor states (CE + restricted KL), pre-normalized
        for i in range(0, A, CHUNK):
            sl = slice(i, min(i + CHUNK, A))
            lg = zlogits(Xa_g[sl], normalized=True)
            ce = F.cross_entropy(lg.float(), ya_g[sl], reduction="none")
            l16 = lg.gather(1, top_a_g[sl])
            logp = F.log_softmax(l16, dim=-1)
            kl = (tp_a_g[sl] * (tp_a_g[sl].clamp_min(1e-9).log() - logp)
                  ).sum(-1)
            part = (ce + LAM_KL * kl).mean() * ((sl.stop - sl.start) / (H + A))
            part.backward(); total += part.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 25 == 0 or step == 1:
            print(f"step {step}: L {total:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
            torch.save({"step": step, "model": model.state_dict()},
                       CKPT / "ofdm_ckpt.pt")

    model.eval()
    torch.save({"model": model.state_dict(), "hidden": 1024, "depth": 3,
                "x_mean": ck["x_mean"], "x_std": ck["x_std"],
                "init_from": str(PS1001), "seed": 104729,
                "n_hard": H, "n_anchor": A,
                "args": {k: str(v) for k, v in vars(a).items()},
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest()},
               Path(a.out))
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
