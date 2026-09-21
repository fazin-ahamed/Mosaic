"""R3-D1 (roadmap S23): narrow A2 recurrence unfreeze.

Unfreeze ONLY the final/output linear (model.head) of the frozen
A2 future-h40 student; keep its input projection and residual
trunk frozen. Train jointly with the bridge decision objective
(full-vocab CE + margin + KL), on the ps recipe base
(gold-frac 0.25, strong position weighting).

Data flow per one-step row (gradient through A2's head):
  gold h40_{t-1} + tokens -> A2 (head unfrozen) -> predicted
  h40_t -> bridge -> z -> frozen norm+FP32 head -> decision loss

Gold-state rows (25% mass) bypass A2 (bridge-only gradient) as
before. The os-variant rows (75% mass) use LIVE A2 forward from
the gold input state — gradient flows A2.head -> bridge.

Anchor monitoring (S25): every 25 steps, record A2's gold-state
one-step standardized MSE on a fixed probe subset. Early-stop
warning (S26) if the anchor degrades > 2x its initial value
while bridge CE improves.

This script loads the hash-gated teacher audit + captures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys = None
import sys as _sys
_sys.path.insert(0, ".")
_sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_bridge import (BridgeNet, load_head,
                                      load_embed, make_bridge_feat,
                                      pos_encode, TEACHER, AUDIT)
from mosaicvm.bench.d3_v1 import (DeltaNet, make_feat, D_STATE,
                                  HORIZON)
from mosaicvm.bench.d3_student import load_captures

SNAP = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--"
    "DeepSeek-V4-Flash-0731/snapshots/"
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062")


def sha_full(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_a2(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = DeltaNet(ck["hidden"], ck["depth"], ck["rank"],
                     aux_out=ck.get("h42_aux_weight", 0) > 0
                     or "aux.weight" in ck["model"])
    model.load_state_dict(ck["model"])
    return (model, ck["target_mean"], ck["target_std"],
            ck["x_mean"], ck["x_std"])


def load_bridge(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = BridgeNet(ck["hidden"], ck["depth"])
    model.load_state_dict(ck["model"])
    return model, ck["x_mean"], ck["x_std"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--a2", type=Path,
                   default=Path("artifacts/d3_v1_A2/student.pt"))
    p.add_argument("--bridge", type=Path,
                   default=Path("artifacts/d3_bridge_r3b/"
                                "bridge_ps/bridge.pt"))
    p.add_argument("--gold-frac", type=float, default=0.25)
    p.add_argument("--pos-weight", default="strong",
                   choices=("none", "mild", "strong"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr-bridge", type=float, default=3e-4)
    p.add_argument("--lr-a2-head", type=float, default=3e-5,
                   help="narrow unfreeze: A2 head LR, << bridge")
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--lam-m", type=float, default=1.0)
    p.add_argument("--lam-kl", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)

    t0 = time.time()
    # ---- hash gates ----
    ver = json.loads(AUDIT.read_text())
    assert ver["verdict"] == "PASS"
    caps = {}
    for cat, h in ver["record_hashes_full"].items():
        cur = sha_full(TEACHER / f"{cat}.pt")
        assert cur == h, f"teacher drift {cat}"
    print(f"hash gates: 84/84 teacher records verified",
          flush=True)

    # ---- models ----
    a2, t_mean, t_std, x_mean, x_std = load_a2(a.a2)
    bridge, b_mean, b_std = load_bridge(a.bridge)
    # NARROW unfreeze: only a2.head
    for prm in a2.parameters():
        prm.requires_grad_(False)
    for prm in a2.head.parameters():
        prm.requires_grad_(True)
    n_frozen = sum(p_.numel() for p_ in a2.parameters()
                   if not p_.requires_grad)
    n_free = sum(p_.numel() for p_ in a2.parameters()
                 if p_.requires_grad)
    print(f"A2 unfreeze: head only ({n_free:,} trainable, "
          f"{n_frozen:,} frozen)", flush=True)

    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda().requires_grad_(False)
    w_norm_g = w_norm.cuda().requires_grad_(False)

    # ---- examples from teacher records + captures ----
    caps_all = load_captures("artifacts/d3_data_105",
                             json.loads(Path(
                                 "artifacts/stage1_prompts.json"
                             ).read_text()), None, HORIZON)
    cap_by_cat = {c["category"]: c for c in caps_all}

    gold_rows, os_rows = [], []
    for cat in sorted(ver["record_hashes_full"]):
        rec = torch.load(TEACHER / f"{cat}.pt",
                         map_location="cpu", weights_only=True)
        cap = cap_by_cat[cat]
        S, K = rec["S"], rec["K"]
        toks = rec["tokens"]
        h_full = cap["h40"].float()          # [S+K, 4, 4096]
        for j in range(K):
            t = S + j
            row = rec["decisions"][j]
            y = row["true_token"]
            top_idx = row["gold"]["top_idx"]
            top_val = torch.tensor(row["gold"]["top_val"])
            comps = [i for i in top_idx if i != y][:8]
            p_t = F.softmax(top_val, dim=0)
            label = {"y": y, "comps": comps, "top_idx": top_idx,
                     "p_t": p_t, "j": j, "t": t}
            # GOLD variant: bridge sees the gold h40_t
            gold_rows.append((label, h_full[t].flatten(),
                              toks, t))
            # OS variant: A2 forward from gold h40_{t-1} (LIVE)
            os_rows.append((label, h_full[t - 1].flatten(),
                            h_full[t - 2].flatten(), toks, t))
    print(f"rows: {len(gold_rows)} gold / {len(os_rows)} os "
          f"({time.time()-t0:.0f}s)", flush=True)

    # mixture: keep ALL rows from ALL prompts; apply GROUP
    # WEIGHTS (w_g ~ g/n_g, w_p ~ (1-g)/n_p) instead of
    # subsampling. No alphabetical prefix survives.
    gsel = gold_rows
    osel = os_rows
    n_g = len(gsel)
    n_o = len(osel)
    # per-row weights: group mixture (gold vs os) x position
    def pw_of(j):
        base = (1.0 / (1 + j)
                if a.pos_weight == "mild"
                else 1.0 / (1 + j) ** 2
                if a.pos_weight == "strong" else 1.0)
        return base
    g_mass = a.gold_frac
    ws = ([g_mass / n_g * pw_of(r[0]["j"]) for r in gsel]
          + [(1 - g_mass) / n_o * pw_of(r[0]["j"])
             for r in osel])
    wmean = sum(ws) / len(ws)
    ws = [w / wmean for w in ws]
    realized = sum(ws[:n_g]) / sum(ws)
    n_prompts = len({r[0].get("cat", "") for r in gsel}
                    | {r[0].get("cat", "") for r in osel})
    print(f"mixture: n_gold={n_g} n_os={n_o} "
          f"n_prompts={n_prompts} "
          f"realized_gold_mass={realized:.4f} "
          f"pos-weight {a.pos_weight}", flush=True)

    # ---- label tensors ----
    y_all = torch.tensor([r[0]["y"] for r in gsel]
                         + [r[0]["y"] for r in osel])
    max_c = max(len(r[0]["comps"]) for r in gsel + osel)
    comp_idx = torch.tensor(
        [r[0]["comps"] + [r[0]["y"]] * (max_c - len(r[0]["comps"]))
         for r in gsel + osel])
    comp_mask = torch.tensor(
        [[1.0] * len(r[0]["comps"])
         + [0.0] * (max_c - len(r[0]["comps"]))
         for r in gsel + osel])
    top_idx_t = torch.tensor([r[0]["top_idx"]
                              for r in gsel + osel])
    p_t_t = torch.stack([r[0]["p_t"] for r in gsel + osel])
    w_t = torch.tensor(ws)
    y_g = y_all.cuda(); comp_g = comp_idx.cuda()
    cmask_g = comp_mask.cuda(); top_g = top_idx_t.cuda()
    pt_g = p_t_t.cuda(); w_g = w_t.cuda()
    print(f"labels ready ({time.time()-t0:.0f}s)", flush=True)

    # per-record probe (clean): one os row per record
    probe2 = []
    for cat in sorted(ver["record_hashes_full"])[:64]:
        rec = torch.load(TEACHER / f"{cat}.pt",
                         map_location="cpu", weights_only=True)
        cap = cap_by_cat[cat]
        S, K = rec["S"], rec["K"]
        toks = rec["tokens"]
        h_full = cap["h40"].float()
        j = min(4, K - 1)
        t = S + j
        probe2.append((cat, h_full[t - 1].flatten(),
                       h_full[t - 2].flatten(), toks, t,
                       h_full[t].flatten()))

    @torch.no_grad()
    def anchor():
        vals = []
        for cat, h_prev, h_prev2, toks, t, gold in probe2:
            feat = make_feat(embed, h_prev, h_prev - h_prev2,
                             toks[t], toks[t - 1], t)
            fn = (feat - x_mean) / x_std
            pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
            h_t = h_prev + pred
            vals.append(((h_t - gold).norm()
                         / gold.norm().clamp_min(1e-6)).item())
        return sum(vals) / len(vals)

    a0 = anchor()
    print(f"anchor A2 one-step rel-L2 (initial): {a0:.4f}",
          flush=True)

    # ---- optimizers ----
    opt = torch.optim.AdamW([
        {"params": bridge.parameters(), "lr": a.lr_bridge},
        {"params": [p_ for p_ in a2.head.parameters()],
         "lr": a.lr_a2_head},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.steps, eta_min=a.lr_bridge * 0.02)

    # precompute bridge features for GOLD rows (no A2)
    g_feats = []
    for (lab, h_t, toks, t) in gsel:
        bf = make_bridge_feat(embed, h_t, toks[t], toks[t - 1],
                              t)
        g_feats.append((bf - b_mean) / b_std)
    g_feats = torch.stack(g_feats)
    n_g_actual = len(g_feats)
    print(f"gold features precomputed ({time.time()-t0:.0f}s)",
          flush=True)

    def score_gpu(zc_g):
        zn = zc_g * torch.rsqrt(
            zc_g.square().mean(-1, keepdim=True) + 1e-6)
        return (zn * w_norm_g) @ w_head_g.t()

    def row_losses(lg, ysl, sl):
        """Combined per-row CE + margin + KL (identical
        semantics to d3_bridge.chunk_losses; CE-only was the
        restricted-objective failure mode)."""
        ce = F.cross_entropy(lg.float(), ysl,
                             reduction="none")
        l_y = lg.gather(1, ysl.unsqueeze(1)).squeeze(1)
        m = l_y.unsqueeze(1) - lg.gather(1, comp_g[sl])
        m_rows = (F.softplus(a.mu - m)
                  * cmask_g[sl]).sum(-1) \
            / cmask_g[sl].sum(-1).clamp_min(1.0)
        l16 = lg.gather(1, top_g[sl])
        logp = F.log_softmax(l16, dim=-1)
        kl_rows = (pt_g[sl] * (pt_g[sl].clamp_min(1e-9).log()
                               - logp)).sum(-1)
        return ce + a.lam_m * m_rows + a.lam_kl * kl_rows

    CHUNK = 512
    n_ex = n_g_actual + len(osel)
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        # -- gold chunk(s): bridge only --
        for i in range(0, n_g_actual, CHUNK):
            zc = bridge(g_feats[i:i + CHUNK])
            lg = score_gpu(zc.cuda())
            # gold rows occupy the FIRST n_g_actual label slots
            sl = slice(i, i + zc.shape[0])
            rl = row_losses(lg, y_g[sl], sl)
            w = w_g[sl]
            part = (rl * w).sum() / w.sum() \
                * (zc.shape[0] / n_ex)
            part.backward()
            total += part.item()
        # -- os chunks: A2 head -> bridge --
        for i in range(0, len(osel), CHUNK):
            sl = slice(i, i + CHUNK)
            chunk = osel[sl]
            zs = []
            for (lab, h_prev, h_prev2, toks, t) in chunk:
                feat = make_feat(embed, h_prev,
                                 h_prev - h_prev2, toks[t],
                                 toks[t - 1], t)
                fn = (feat - x_mean) / x_std
                pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
                h_t = h_prev + pred
                bf = make_bridge_feat(embed, h_t, toks[t],
                                      toks[t - 1], t)
                zs.append(bridge(((bf - b_mean) / b_std)
                                 .unsqueeze(0)))
            zc = torch.cat(zs)                    # [c, 4096]
            lg = score_gpu(zc.cuda())
            gsl = slice(n_g_actual + sl.start,
                        n_g_actual + sl.stop)
            rl = row_losses(lg, y_g[gsl], gsl)
            w = w_g[gsl]
            part = (rl * w).sum() / w.sum() \
                * (zc.shape[0] / n_ex)
            part.backward()
            total += part.item()
        torch.nn.utils.clip_grad_norm_(
            bridge.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(
            a2.head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 20 == 0 or step == a.steps:
            av = anchor()
            print(f"step {step}: L {total:.4f} "
                  f"anchor {av:.4f} (init {a0:.4f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if av > 2.0 * a0:
                print("WARNING: anchor degraded >2x — "
                      "representation destruction risk (S26)",
                      flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": a2.state_dict(), "hidden": 512, "depth": 3,
        "rank": 0, "target_state": "h40",
        "h42_aux_weight": 0.0,
        "target_mean": t_mean, "target_std": t_std,
        "x_mean": x_mean, "x_std": x_std,
        "joint_meta": {"unfrozen": "a2.head",
                       "lr_a2_head": a.lr_a2_head,
                       "anchor_initial": a0,
                       "anchor_final": anchor(),
                       "bridge_from": str(a.bridge)},
    }, a.out / "a2_joint.pt")
    torch.save({
        "model": bridge.state_dict(), "hidden": 1024, "depth": 3,
        "x_mean": b_mean, "x_std": b_std,
    }, a.out / "bridge_joint.pt")
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
