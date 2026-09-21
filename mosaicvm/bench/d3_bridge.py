"""R2-5 (roadmap S32-S35): Error-Correcting Decision Bridge.

Frozen A2 recurrence + learned bridge G(h40_pred, tokens, pos) -> z
(a decision-equivalent state), scored by the FROZEN production
norm+FP32 head. The bridge is trained on gold + A2 one-step + A2
chained predicted h40 states, ALL labeled with the TRUE future
target decision from the validated teacher dataset — it learns to
correct A2's characteristic state errors into correct drafts.

Loss (S35): margin against the teacher's top competitors +
KL to the teacher's top-16 distribution:

    L = lam_m * sum_j softplus[mu - (l_y - l_j)]
        + lam_kl * KL(p_T_top16 || softmax(l_top16))

Head rows are GATHERED per example (true token + competitors +
top-16), so no full 129k logit materialization is needed for
training. All supervision comes from the stored teacher labels;
the exact tail is never run inside an optimizer step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TEACHER = Path("artifacts/d3_r2_teacher")
AUDIT = Path("artifacts/d3_r2_teacher_audit.json")
SNAP = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--"
    "DeepSeek-V4-Flash-0731/snapshots/"
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062")
D_STATE = 16384
D_TOK = 4096
POS_DIM = 64


def pos_encode(pos):
    half = POS_DIM // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32)
                      * (-math.log(10000.0) / half))
    ang = pos.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([ang.sin(), ang.cos()], dim=1)


def make_bridge_feat(embed, state, driver_tok, cur_tok, pos):
    """Canonical bridge feature (single source of truth; the eval
    imports this). state: [16384] fp32; pos: int position."""
    return torch.cat([
        state,
        embed[driver_tok].float(),
        embed[cur_tok].float(),
        pos_encode(torch.tensor([pos]))[0],
    ])


class BridgeNet(nn.Module):
    def __init__(self, hidden=1024, depth=3):
        super().__init__()
        d_in = D_STATE + 2 * D_TOK + POS_DIM
        self.inp = nn.Linear(d_in, hidden)
        self.mid = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth - 1)])
        self.head = nn.Linear(hidden, 4096)   # decision state z
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.inp(x))
        for m in self.mid:
            h = h + self.act(m(h))     # true skip
        return self.head(h)


def load_head():
    """Frozen production norm weight + FP32 head (CPU)."""
    from safetensors import safe_open
    index = json.loads((SNAP / "model.safetensors.index.json")
                       .read_text())["weight_map"]
    with safe_open(str(SNAP / index["head.weight"]),
                   framework="pt") as f:
        w = f.get_tensor("head.weight").float()      # [V, 4096]
    with safe_open(str(SNAP / index["norm.weight"]),
                   framework="pt") as f:
        nrm = f.get_tensor("norm.weight").float()
    return w, nrm


def load_embed():
    from safetensors import safe_open
    index = json.loads((SNAP / "model.safetensors.index.json")
                       .read_text())["weight_map"]
    with safe_open(str(SNAP / index["embed.weight"]),
                   framework="pt") as f:
        return f.get_tensor("embed.weight")          # bf16 [V,4096]


def build_examples(audit):
    """All bridge training examples from the validated teacher set.

    Per position t: three input variants (gold / os / chain h40)
    with the SAME true-decision label. Returns tensors + label
    structures. Fails closed if the audit is not PASS."""
    ver = json.loads(AUDIT.read_text())
    if ver["verdict"] != "PASS":
        raise RuntimeError("teacher audit is not PASS — refusing")
    import hashlib as _hl
    cats = sorted(ver.get("record_hashes_full")
                  or ver["record_hashes"])
    # fail closed: recompute EVERY record hash now; any drift from
    # the audited state refuses to train
    for cat in cats:
        cur = _hl.sha256((TEACHER / f"{cat}.pt").read_bytes()
                         ).hexdigest()
        stored = ver["record_hashes_full"][cat] \
            if "record_hashes_full" in ver \
            else ver["record_hashes"][cat]
        if cur != stored:
            raise RuntimeError(
                f"teacher record {cat} changed since audit "
                f"— refusing to train")
    print(f"hash check: {len(cats)}/{len(cats)} records match "
          f"the audit", flush=True)
    embed = load_embed()

    feats, meta = [], []
    for cat in cats:
        rec = torch.load(TEACHER / f"{cat}.pt", map_location="cpu",
                         weights_only=True)
        S, K = rec["S"], rec["K"]
        toks = rec["tokens"]
        for j in range(K):
            t = S + j
            row = rec["decisions"][j]
            y = row["true_token"]
            top_idx = row["gold"]["top_idx"]
            top_val = torch.tensor(row["gold"]["top_val"])
            # competitors: teacher top-k excluding the true token
            comps = [i for i in top_idx if i != y][:8]
            # teacher probs over its own top-16 (restricted)
            p_t = F.softmax(top_val, dim=0)
            for var in ("h40_gold_future", "h40_pred_os",
                        "h40_pred_chain"):
                state = rec[var][j].flatten().float()
                f = make_bridge_feat(embed, state, toks[t],
                                     toks[t - 1], t)
                feats.append(f)
                meta.append({"y": y, "comps": comps,
                             "top_idx": top_idx, "p_t": p_t,
                             "variant": var.split("_")[-1],
                             "cat": cat, "j": j})
    x = torch.stack(feats)
    return x, meta, embed


def mixture_weights(meta, gold_frac):
    """Per-row loss weights implementing the gold:predicted
    mixture over the COMBINED row objective (roadmap S26). Raises
    on an empty required group. Prints realized-mixture stats."""
    n = len(meta)
    gold_idx = [i for i, m in enumerate(meta)
                if m["variant"] == "future"]
    pred_idx = [i for i, m in enumerate(meta)
                if m["variant"] != "future"]
    if gold_frac is None:
        print(f"mixture: uniform ({len(gold_idx)} gold / "
              f"{len(pred_idx)} pred rows)", flush=True)
        return torch.ones(n)
    if not gold_idx or not pred_idx:
        raise RuntimeError(
            f"mixture requires both groups; gold={len(gold_idx)} "
            f"pred={len(pred_idx)}")
    w = torch.zeros(n)
    w[gold_idx] = gold_frac / len(gold_idx)
    w[pred_idx] = (1 - gold_frac) / len(pred_idx)
    w = w * n / w.sum()              # mean weight 1
    realized = w[gold_idx].sum() / w.sum()
    print(f"mixture gold_frac={gold_frac}: n_gold={len(gold_idx)} "
          f"n_pred={len(pred_idx)} "
          f"mean_gold_w={w[gold_idx].mean():.4f} "
          f"mean_pred_w={w[pred_idx].mean():.4f} "
          f"realized_gold_mass={realized:.4f}", flush=True)
    if abs(realized.item() - gold_frac) > 0.01:
        raise RuntimeError(
            f"realized gold mass {realized:.4f} != requested "
            f"{gold_frac}")
    return w


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/d3_bridge_r2"))
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--lam-m", type=float, default=1.0)
    p.add_argument("--lam-kl", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--gold-frac", type=float, default=None,
                   help="R3-A: fraction of training mass on GOLD "
                        "states (default None = equal 1/3 across "
                        "gold/os/chain). Applied as per-row loss "
                        "weights; predicted variants share the "
                        "remainder equally.")
    p.add_argument("--pos-weight", choices=("none", "mild",
                                            "strong"),
                   default="none",
                   help="R3-B: position weighting of the decision "
                        "loss. w_j = 1/(1+j) (mild) or 1/(1+j)^2 "
                        "(strong) on the position index within the "
                        "speculative window; multiplies row_w.")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)

    t0 = time.time()
    x, meta, embed = build_examples(None)
    row_w = mixture_weights(meta, a.gold_frac)
    if a.pos_weight != "none":
        # R3-B: early speculative positions weighted higher in
        # the DECISION loss (full-vocab CE), not state MSE
        # (distinct from the closed B3). j = position index in
        # the K window, stored in meta from build_examples.
        pos_js = torch.tensor([m["j"] for m in meta],
                              dtype=torch.float32)
        pw = (1.0 / (1.0 + pos_js)
              if a.pos_weight == "mild"
              else 1.0 / (1.0 + pos_js) ** 2)
        pw = pw / pw.mean()        # mean 1
        row_w = row_w * pw
        print(f"pos-weight {a.pos_weight}: applied over j in "
              f"[0, {int(pos_js.max())}], mean-normalized",
              flush=True)
    row_w_g = row_w.cuda()
    n = x.shape[0]
    print(f"{n} examples ({n // 3} positions x 3 variants) "
          f"({time.time()-t0:.0f}s)", flush=True)
    # per-dim input standardization (train only)
    x_mean, x_std = x.mean(0), x.std(0).clamp_min(1e-4)
    xn = (x - x_mean) / x_std

    w_head, w_norm = load_head()
    print(f"head loaded {tuple(w_head.shape)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # label gathers (once)
    y_idx = torch.tensor([m["y"] for m in meta])
    W_y = w_head[y_idx]                                # [n, 4096]
    max_c = max(len(m["comps"]) for m in meta)
    comp_idx = torch.tensor(
        [m["comps"] + [m["y"]] * (max_c - len(m["comps"]))
         for m in meta])                               # [n, max_c]
    comp_mask = torch.tensor(
        [[1.0] * len(m["comps"])
         + [0.0] * (max_c - len(m["comps"])) for m in meta])
    top_idx = torch.tensor([m["top_idx"] for m in meta])  # [n, 16]
    p_t = torch.stack([m["p_t"] for m in meta])        # [n, 16]
    print(f"labels gathered ({time.time()-t0:.0f}s)", flush=True)
    # W_y (the restricted gather) is no longer needed: the GPU
    # full-vocab logits replace all restricted gathers.
    del W_y

    model = BridgeNet(a.hidden, a.depth)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.steps, eta_min=a.lr * 0.02)

    # GPU-resident frozen head + norm: FULL-VOCAB scoring (the
    # restricted-objective failure — train top16-acc 100% but
    # full-argmax 2-7/32 on train — showed ~129k untrained head
    # rows overtaking the argmax; CE over ALL rows fixes it).
    w_head_g = w_head.cuda().requires_grad_(False)
    w_norm_g = w_norm.cuda().requires_grad_(False)
    y_idx_g = y_idx.cuda()
    comp_idx_g = comp_idx.cuda()
    comp_mask_g = comp_mask.cuda()
    top_idx_g = top_idx.cuda()
    p_t_g = p_t.cuda()

    def chunk_losses(zc, sl):
        """One chunk's per-row COMBINED loss (CE + margin + KL),
        then group-weighted (roadmap S26: the mixture applies to
        the TOTAL objective, not CE alone)."""
        zc_g = zc.cuda()
        zn = zc_g * torch.rsqrt(
            zc_g.square().mean(-1, keepdim=True) + 1e-6)
        zn = zn * w_norm_g
        lg = zn @ w_head_g.t()                       # [c, V]
        ce_rows = F.cross_entropy(lg.float(), y_idx_g[sl],
                                  reduction="none")
        l_y = lg.gather(1, y_idx_g[sl].unsqueeze(1)).squeeze(1)
        m = l_y.unsqueeze(1) - lg.gather(1, comp_idx_g[sl])
        m_rows = (F.softplus(a.mu - m)
                  * comp_mask_g[sl]).sum(-1) \
            / comp_mask_g[sl].sum(-1).clamp_min(1.0)
        l16 = lg.gather(1, top_idx_g[sl])
        logp_s = F.log_softmax(l16, dim=-1)
        kl_rows = (p_t_g[sl] * (p_t_g[sl].clamp_min(1e-9).log()
                                - logp_s)).sum(-1)
        row_loss = (ce_rows + a.lam_m * m_rows
                    + a.lam_kl * kl_rows)
        w = row_w_g[sl]
        return (row_loss * w).sum() / w.sum()

    CHUNK = 512
    n_ex = xn.shape[0]

    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        # per-chunk forward+backward through the model input
        # slice: each chunk builds its OWN graph (no retain_graph
        # needed) and only one [c, V] logit block is ever live
        for i in range(0, n_ex, CHUNK):
            sl = slice(i, i + CHUNK)
            c = sl.stop - sl.start
            zc = model(xn[sl])                # [c, 4096]
            part = chunk_losses(zc, sl) * (c / n_ex)
            part.backward()
            total += part.item()
            del zc
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == a.steps:
            with torch.no_grad():
                accs = []
                for i in range(0, n_ex, 2048):
                    sl = slice(i, i + 2048)
                    zg = model(xn[sl]).cuda()
                    zn = zg * torch.rsqrt(
                        zg.square().mean(-1, keepdim=True) + 1e-6
                    ) * w_norm_g
                    lg = zn @ w_head_g.t()
                    accs.append((lg.argmax(-1) == y_idx_g[sl])
                                .float().mean().item())
                acc = sum(accs) / len(accs)
            print(f"step {step}: L {total:.4f} "
                  f"FULL-argmax-acc {acc:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "hidden": a.hidden,
        "depth": a.depth, "x_mean": x_mean, "x_std": x_std,
        "args": {k: str(v) for k, v in vars(a).items()},
        "seed": 104729,
        "teacher_audit_sha256_16": hashlib.sha256(
            AUDIT.read_bytes()).hexdigest()[:16],
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
    }, a.out / "bridge.pt")
    print(f"wrote {a.out / 'bridge.pt'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
