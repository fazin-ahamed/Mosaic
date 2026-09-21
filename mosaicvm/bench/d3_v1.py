"""D3-v1: token-conditioned one-step Delta-h40 student.

Roadmap D3-v1 (supersedes the failed rank32 PCA/ridge baseline):
learn the one-step state transition

    Delta h40_{t+1} = F(h40_t, Delta h40_t, emb(token_{t+1}),
                        emb(token_t), pos t+1)

so that h40_{t+1} = h40_t + Delta. Every 33-state capture yields 32
sequential transitions (K1 data), instead of one K32 pair.

Training is teacher-forced on gold states/tokens. Export modes:
  hold      root held constant (floor)
  one_step  gold input state + true token (isolated per-step error)
  tf_chain  chained predicted states + TRUE token conditioning
            (state feedback; token feedback is the evaluator's job)

Free-running closed-loop evaluation (student's own tail-decoded
tokens fed back) is d3_v1_eval.py — the promotion metric.

CPU-only training; the exact source tail remains the evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn

from mosaicvm.bench.d3_student import load_captures

SNAP = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--"
    "DeepSeek-V4-Flash-0731/snapshots/"
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062")
D_STATE = 16384      # 4 x 4096 flattened HC state
D_TOK = 4096
POS_DIM = 64
HORIZON = 32


def pos_encode(pos: torch.Tensor) -> torch.Tensor:
    """Sinusoidal position features for integer positions [B]."""
    half = POS_DIM // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32)
                      * (-math.log(10000.0) / half))
    ang = pos.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([ang.sin(), ang.cos()], dim=1)


def make_feat(embed, state, delta_in, driver_tok, cur_tok, pos):
    """Canonical D3-v1 feature. The pairing contract (roadmap
    S14): predicting the state at position `pos` uses the input
    state h40[pos-1], its delta, the DRIVER token = tokens[pos]
    (the token consumed to produce h40[pos]; in production the
    draft chosen from h42[pos-1]), and the CURRENT token =
    tokens[pos-1]. ALL paths (training, TF export, exposure,
    tf_loop, cl_loop) must build features through this function."""
    return torch.cat([
        state, delta_in,
        embed[driver_tok].float(),
        embed[cur_tok].float(),
        pos_encode(torch.tensor([pos]))[0],
    ])


class DeltaNet(nn.Module):
    """Residual MLP: token-conditioned one-step delta predictor.
    aux_out adds a second head predicting the same-position h42
    (downstream-alignment target, roadmap S31) from the shared
    trunk; the primary head is unchanged."""

    def __init__(self, hidden=1024, depth=3, rank=0, aux_out=False):
        super().__init__()
        d_in = D_STATE + D_STATE + 2 * D_TOK + POS_DIM
        self.rank = rank
        self.aux_out = aux_out
        self.inp = nn.Linear(d_in, hidden)
        self.mid = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth - 1)])
        if rank > 0:
            self.head = nn.Linear(hidden, rank)
            self.U = nn.Parameter(torch.zeros(D_STATE, rank))
            nn.init.orthogonal_(self.U)
        else:
            self.head = nn.Linear(hidden, D_STATE)
        if aux_out:
            self.aux = nn.Linear(hidden, D_STATE)
        self.act = nn.GELU()

    def forward(self, x, with_aux=False):
        h = self.act(self.inp(x))
        for m in self.mid:
            h = h + self.act(m(h))     # true skip: never re-activate
        z = self.head(h)
        if self.rank > 0:
            out = z @ self.U.t()
        else:
            out = z
        if with_aux and self.aux_out:
            return out, self.aux(h)
        return out


def build_transitions(captures, embed, split, state_field="h40",
                      aux_field=None):
    """Teacher-forced transitions from captures of one split.

    state_field: which trajectory the recurrence/delta is built
    from ("h40" or "h42"; positions/alignment are identical).
    aux_field: optional second trajectory recorded per row as an
    auxiliary same-position target (e.g. "h42" for downstream
    alignment of an h40 student).

    Returns dict with precomputed fp32 inputs [N, d_in], targets
    [N, 16384] (aux targets [N, 16384] or None), plus per-prompt
    index bookkeeping for rollouts.
    """
    rows, meta = [], []
    for c in captures:
        h = c[state_field].float()              # [S+K, 4, 4096]
        ha = c[aux_field].float() if aux_field else None
        toks = c["tokens"]
        S = c["S"]
        K = min(HORIZON, h.shape[0] - S)
        idx = []
        for t in range(S - 1, S + K - 1):
            cur, nxt = toks[t], toks[t + 1]
            feat = make_feat(embed, h[t].flatten(),
                             (h[t] - h[t - 1]).flatten(),
                             nxt, cur, t + 1)
            aux_t = ha[t + 1].flatten() if ha is not None else None
            rows.append((feat, (h[t + 1] - h[t]).flatten(),
                         t + 1, aux_t))
            idx.append(len(rows) - 1)
        meta.append({"category": c["category"], "S": S, "K": K,
                     "row_idx": idx, "capture": c})
    x = torch.stack([r[0] for r in rows])   # pos already in make_feat
    y = torch.stack([r[1] for r in rows])
    y_aux = (torch.stack([r[3] for r in rows])
             if rows and rows[0][3] is not None else None)
    return {"x": x, "y": y, "y_aux": y_aux, "meta": meta,
            "split": split}


def load_embed():
    from safetensors import safe_open
    index = json.loads((SNAP / "model.safetensors.index.json")
                       .read_text())["weight_map"]
    shard = index["embed.weight"]
    with safe_open(str(SNAP / shard), framework="pt") as f:
        return f.get_tensor("embed.weight")   # bf16 [V, 4096] cpu


def rel_l2(pred, target):
    return ((pred - target).norm(dim=-1) /
            target.norm(dim=-1).clamp_min(1e-6))


def rollout_chain(model, prompt_meta, embed, std, mean,
                  x_mean=None, x_std=None, max_k=None):
    """Chained prediction with TRUE token conditioning (tf_chain)."""
    c = prompt_meta["capture"]
    S, K = prompt_meta["S"], prompt_meta["K"]
    if max_k:
        K = min(K, max_k)
    toks = c["tokens"]
    h = c["h40"].float()
    state = h[S - 1].flatten()
    prev_gold = h[S - 2].flatten()      # exact prefix delta at root
    out = []
    model.eval()
    with torch.no_grad():
        for j in range(K):
            t = S + j                   # position being predicted
            delta_in = state - prev_gold
            # driver token = tokens[t] (chosen from h42[t-1]; the
            # production feedback token); cur = tokens[t-1]
            feat = make_feat(embed, state, delta_in,
                             toks[t], toks[t - 1], t)
            if x_mean is not None:
                feat = (feat - x_mean) / x_std
            pred = model(feat.unsqueeze(0))[0] * std + mean
            out.append(state + pred)
            prev_gold = state
            state = state + pred
    return torch.stack(out).view(K, 4, 4096).to(torch.bfloat16)


def rollout_onestep(model, prompt_meta, embed, std, mean,
                    x_mean=None, x_std=None):
    """Gold-state one-step predictions (isolated error)."""
    c = prompt_meta["capture"]
    S, K = prompt_meta["S"], prompt_meta["K"]
    toks = c["tokens"]
    h = c["h40"].float()
    out = []
    model.eval()
    with torch.no_grad():
        for j in range(K):
            t = S + j
            feat = make_feat(embed, h[t - 1].flatten(),
                             (h[t - 1] - h[t - 2]).flatten(),
                             toks[t], toks[t - 1], t)
            if x_mean is not None:
                feat = (feat - x_mean) / x_std
            pred = model(feat.unsqueeze(0))[0] * std + mean
            out.append(h[t - 1].flatten() + pred)
    return torch.stack(out).view(K, 4, 4096).to(torch.bfloat16)


def train_model(train_ds, val_ds, hidden, depth, rank, steps,
                batch, lr, log_every=50, aux_weight=0.0):
    torch.manual_seed(104729)
    model = DeltaNet(hidden, depth, rank,
                     aux_out=aux_weight > 0)
    x, y = train_ds["x"], train_ds["y"]
    y_aux = train_ds.get("y_aux")
    # per-dim input + target standardization (train only; the real
    # input has a strong common-mode cone that saturates GELU
    # units identically across samples without this)
    x_mean, x_std = x.mean(0), x.std(0).clamp_min(1e-4)
    x = (x - x_mean) / x_std
    mean = y.mean(0)
    std = y.std(0).clamp_min(1e-4)
    yn = (y - mean) / std
    if y_aux is not None and aux_weight > 0:
        aux_mean = y_aux.mean(0)
        aux_std = y_aux.std(0).clamp_min(1e-4)
        yn_aux = (y_aux - aux_mean) / aux_std
    else:
        aux_mean = aux_std = yn_aux = None
    n = x.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.01)
    t0 = time.perf_counter()
    hist = []
    use_full = batch == 0 or batch >= n
    for step in range(1, steps + 1):
        if use_full:
            xb, ynb = x, yn
            xab = yn_aux
        else:
            gen = torch.Generator().manual_seed(104729 + step)
            idx = torch.randint(0, n, (batch,), generator=gen)
            xb, ynb = x[idx], yn[idx]
            xab = yn_aux[idx] if yn_aux is not None else None
        if xab is not None:
            pred, pred_aux = model(xb, with_aux=True)
            loss = (((pred - ynb) ** 2).mean()
                    + aux_weight
                    * ((pred_aux - xab) ** 2).mean())
        else:
            pred = model(xb)
            loss = ((pred - ynb) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                v = quick_val(model, val_ds, mean, std, x_mean, x_std)
            hist.append({"step": step, "train_mse": loss.item(),
                         "val_step_rel_l2": v})
            print(f"step {step}: loss {loss.item():.4f} "
                  f"val_step_relL2 {v:.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return (model, mean, std, hist, x_mean, x_std,
            aux_mean, aux_std)


@torch.no_grad()
def quick_val(model, val_ds, mean, std, x_mean, x_std, k=512):
    """Mean per-step rel-L2 on a subsample of val transitions."""
    x, y = val_ds["x"], val_ds["y"]
    n = min(k, x.shape[0])
    idx = torch.linspace(0, x.shape[0] - 1, n).long()
    xn = (x[idx] - x_mean) / x_std
    pred = model(xn) * std + mean
    return rel_l2(pred, y[idx]).mean().item()


@torch.no_grad()
def build_exposure(model, train_ds, embed, mean, std, x_mean, x_std,
                   max_drift=0.75, horizon=None, telemetry=None):
    """Chained rollouts on TRAIN prompts; target = correction that
    lands on the gold state (recovery from own drift). Chains that
    drift beyond max_drift re-anchor to gold — training sees only
    plausible drift, not catastrophic off-manifold states.

    horizon limits each chain to its first H positions (horizon
    curriculum). telemetry (list) receives one record per position:
    {prompt, j, drift, reset} — the B0 instrumentation contract."""
    rows = []
    for pm in train_ds["meta"]:
        c = pm["capture"]
        S, K = pm["S"], pm["K"]
        if horizon is not None:
            K = min(K, horizon)
        toks = c["tokens"]
        h = c["h40"].float()
        state = h[S - 1].flatten()
        prev = h[S - 2].flatten()
        for j in range(K):
            t = S + j
            gold_prev = h[t - 1].flatten()
            drift = ((state - gold_prev).norm()
                     / gold_prev.norm().clamp_min(1e-6)).item()
            reset = drift > max_drift
            if telemetry is not None:
                telemetry.append({"prompt": pm["category"], "j": j,
                                  "drift": round(drift, 4),
                                  "reset": bool(reset)})
            if reset:
                state = gold_prev          # re-anchor
                prev = h[t - 2].flatten()
            delta_in = state - prev
            feat = make_feat(embed, state, delta_in,
                             toks[t], toks[t - 1], t)
            xn = (feat - x_mean) / x_std
            pred = model(xn.unsqueeze(0))[0] * std + mean
            gold = h[t].flatten()
            rows.append((xn, gold - state, j, drift))  # recovery
            prev = state
            state = state + pred
    return (torch.stack([r[0] for r in rows]),
            torch.stack([r[1] for r in rows]),
            torch.tensor([r[2] for r in rows]),
            torch.tensor([r[3] for r in rows]))


@torch.no_grad()
def build_collector_exposure(collector, train_ds, embed, mean, std,
                             x_mean, x_std, max_drift=3.0,
                             horizon=None):
    """Uninterrupted rollouts of a FROZEN collector model on train
    prompts. Positions whose collector-drift exceeds max_drift are
    DROPPED at generation time (not re-anchored — the trajectory
    remains the collector's own). Recovery target: gold state -
    collector state. This supplies genuinely uninterrupted
    off-manifold states at bounded drift, distinct from the
    training model's re-anchored chains (B1)."""
    rows = []
    drifts_all = []
    tel = []
    for pm in train_ds["meta"]:
        c = pm["capture"]
        S, K = pm["S"], pm["K"]
        if horizon is not None:
            K = min(K, horizon)
        toks = c["tokens"]
        h = c["h40"].float()
        state = h[S - 1].flatten()
        prev = h[S - 2].flatten()
        for j in range(K):
            t = S + j
            gold_prev = h[t - 1].flatten()
            drift = ((state - gold_prev).norm()
                     / gold_prev.norm().clamp_min(1e-6)).item()
            if drift > max_drift:
                break                     # drop rest of trajectory
            feat = make_feat(embed, state, state - prev,
                             toks[t], toks[t - 1], t)
            xn = (feat - x_mean) / x_std
            pred = collector(xn.unsqueeze(0))[0] * std + mean
            gold = h[t].flatten()
            rows.append((xn, gold - state, j))
            drifts_all.append(drift)
            tel.append({"prompt": pm["category"], "j": j,
                        "drift": round(drift, 4), "reset": False})
            prev = state
            state = state + pred
    return (torch.stack([r[0] for r in rows]),
            torch.stack([r[1] for r in rows]),
            torch.tensor([r[2] for r in rows]),
            torch.tensor(drifts_all), tel)


def train_exposure(model, train_ds, embed, mean, std, x_mean, x_std,
                   rounds, steps_per_round, lr, log_every=50,
                   threshold_schedule=None, horizon_schedule=None,
                   prefix_weight=None, telemetry_out=None,
                   exposure_mode="reset", rail_weights=(1.0, 1.0),
                   input_clamp=None, collector_model=None):
    """Phase 2: mix teacher-forced and own-trajectory examples.

    threshold_schedule(r, rounds) -> drift threshold for round r
    (None = fixed 0.75). horizon_schedule(r, rounds) -> chain
    horizon for round r (None = full K). prefix_weight: None or
    'harmonic' (w_j = 1/(1+j): early positions weighted higher).
    telemetry_out: optional Path for per-round B0 telemetry JSON.
    exposure_mode: 'reset' (re-anchor beyond threshold), 'free'
    (never reset — uninterrupted rollouts; requires a staged
    threshold_schedule per the d3_c2_preflight STAGE/BOUND
    verdict — raw free rollouts explode to drift ~288), 'dagger'
    (free + drift-balanced oversampling of the recovery set),
    'staged_free' (threshold-bounded free exposure: positions
    with drift <= threshold run FREE; beyond threshold re-anchor
    — the progressive-free curriculum). rail_weights:
    (lambda_anchor, lambda_free) with independent per-rail loss
    normalization (dual-rail objective, roadmap S21/S24)."""
    x_tf = (train_ds["x"] - x_mean) / x_std
    y_tf = (train_ds["y"] - mean) / std
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=1e-4)
    t0 = time.perf_counter()
    hist = []
    tel_rounds = []
    for r in range(1, rounds + 1):
        thr = (threshold_schedule(r, rounds) if threshold_schedule
               else 0.75)
        if exposure_mode == "free":
            thr = float("inf")
        # staged_free: threshold bounds the collector rollout
        # acceptance (schedule relaxes it 0.75 -> 3.0 across
        # rounds); dagger keeps the same bounded regime with
        # drift-balanced recovery oversampling
        hz = (horizon_schedule(r, rounds) if horizon_schedule
              else None)
        tel = []
        if exposure_mode == "staged_free":
            # STAGED FREE RAIL (distinct from reset/B1): states come
            # from a FROZEN collector model's uninterrupted rollouts
            # (drift-bounded by generation-time acceptance), not from
            # the training model's own re-anchored chains. The
            # recovery problem is collector-state -> gold; the
            # anchor rail (loss_a on gold x_tf) runs every update.
            xe, ye, js, drifts, tel2 = build_collector_exposure(
                collector_model, train_ds, embed, mean, std,
                x_mean, x_std, max_drift=thr, horizon=hz)
            tel = tel2
        else:
            xe, ye, js, drifts = build_exposure(model, train_ds,
                                                embed, mean, std,
                                                x_mean, x_std,
                                                max_drift=thr,
                                                horizon=hz,
                                                telemetry=tel)
        if exposure_mode == "dagger":
            # drift-balanced oversampling: equalize mild (<0.5),
            # mid (0.5-1.5), large (>1.5) drift buckets
            b = drifts
            idx_mild = (b < 0.5).nonzero().flatten()
            idx_mid = ((b >= 0.5) & (b < 1.5)).nonzero().flatten()
            idx_big = (b >= 1.5).nonzero().flatten()
            target_n = max(len(idx_mild), len(idx_mid),
                           len(idx_big), 1)
            g = torch.Generator().manual_seed(104729 + r)
            picks = []
            for idx_b in (idx_mild, idx_mid, idx_big):
                if len(idx_b) == 0:
                    continue
                rep = torch.randint(
                    0, len(idx_b), (target_n - len(idx_b),),
                    generator=g)
                picks.append(torch.cat([idx_b, idx_b[rep]]))
            if picks:
                sel = torch.cat(picks)
                xe, ye, js = xe[sel], ye[sel], js[sel]
        # B0 telemetry aggregation per round (contract: segments
        # NEVER cross prompt boundaries; every prompt reported
        # including zero-reset ones; reset positions and post-reset
        # drift retained)
        by_prompt = {}
        for t in tel:
            by_prompt.setdefault(t["prompt"], []).append(t)
        resets = [t for t in tel if t["reset"]]
        seg_hist, seg_all = {}, []
        post_reset_drifts, reset_positions = [], []
        per_prompt = {}
        for p, ts in by_prompt.items():
            segs_p, run = [], 0
            for i, t in enumerate(ts):
                if t["reset"]:
                    segs_p.append(run)      # close segment at reset
                    run = 0
                    reset_positions.append({"prompt": p, "j": t["j"],
                                            "drift": t["drift"]})
                    if i + 1 < len(ts):     # first step after reset
                        post_reset_drifts.append(ts[i + 1]["drift"])
                else:
                    run += 1
            segs_p.append(run)              # close final segment
            seg_all.extend(segs_p)
            for s in segs_p:
                seg_hist[s] = seg_hist.get(s, 0) + 1
            per_prompt[p] = {
                "positions": len(ts),
                "resets": sum(1 for t in ts if t["reset"]),
                "uninterrupted_len_max": max(segs_p),
                "drift_max": max(t["drift"] for t in ts),
            }
        tel_rounds.append({
            "round": r, "threshold": thr, "horizon": hz,
            "positions": len(tel), "resets": len(resets),
            "reset_fraction": round(len(resets) / max(len(tel), 1), 4),
            "uninterrupted_len_histogram": seg_hist,
            "uninterrupted_len_mean": round(
                sum(seg_all) / max(len(seg_all), 1), 2),
            "uninterrupted_len_max": max(seg_all),
            "drift_mean": round(sum(t["drift"] for t in tel)
                                / max(len(tel), 1), 4),
            "drift_max": round(max(t["drift"] for t in tel), 4),
            "pre_reset_drift_mean": round(
                sum(t["drift"] for t in resets) / max(len(resets), 1), 4)
            if resets else None,
            "post_reset_drift_mean": round(
                sum(post_reset_drifts) / max(len(post_reset_drifts), 1), 4)
            if post_reset_drifts else None,
            "reset_positions": reset_positions,
            "per_prompt": per_prompt,       # ALL prompts, incl. 0-reset
        })
        # recovery targets are clamped at ±4 standardized (this is
        # the training objective definition, not an "exact" target)
        ye_n = ((ye - mean) / std).clamp(-4, 4)
        # NO unconditional input clamp: it silently changes the
        # reset/B0/C1 objective too. free/dagger arms may pass
        # input_clamp explicitly (see below); the staged collector
        # bounds states instead of clamping features.
        if exposure_mode == "free" and input_clamp:
            xe = xe.clamp(-input_clamp, input_clamp)
        la, lf = rail_weights
        if prefix_weight == "harmonic":
            w_exp = 1.0 / (1.0 + js.float())
        else:
            w_exp = None
        for step in range(1, steps_per_round + 1):
            pred_a = model(x_tf)
            loss_a = ((pred_a - y_tf) ** 2).mean()
            pred_f = model(xe)
            if w_exp is None:
                loss_f = ((pred_f - ye_n) ** 2).mean()
            else:
                loss_f = (w_exp.unsqueeze(-1)
                          * (pred_f - ye_n) ** 2).sum() / (
                    w_exp.sum() * pred_f.shape[-1])
            loss = la * loss_a + lf * loss_f
            if not torch.isfinite(loss):
                print(f"exposure round {r}: non-finite loss, "
                      f"skipping update", flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        hist.append({"round": r, "mixed_mse": loss.item(),
                     "threshold": thr, "horizon": hz,
                     "reset_fraction": tel_rounds[-1]["reset_fraction"]})
        print(f"exposure round {r}: mixed {loss.item():.4f} "
              f"thr {thr} hz {hz} resets "
              f"{tel_rounds[-1]['reset_fraction']:.3f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    if telemetry_out is not None:
        Path(telemetry_out).write_text(json.dumps(
            {"per_round": tel_rounds}, indent=1))
    return hist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path,
                   default=Path("artifacts/stage1_prompts.json"))
    p.add_argument("--categories")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--rank", type=int, default=0,
                   help="0 = full-state head")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=0,
                   help="0 = full batch (required for the 41k-dim "
                        "input; minibatch noise freezes the first "
                        "layer subspace rotation)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--exposure-rounds", type=int, default=0)
    p.add_argument("--exposure-steps", type=int, default=40)
    p.add_argument("--threshold-decay", choices=("none", "relax"),
                   default="none",
                   help="relax: drift threshold 0.75 -> 3.0 "
                        "geometrically across exposure rounds")
    p.add_argument("--horizon-curriculum", action="store_true",
                   help="H = 2,4,8,16,32 in equal round phases")
    p.add_argument("--prefix-weight", choices=("none", "harmonic"),
                   default="none")
    p.add_argument("--telemetry-out", type=Path, default=None)
    p.add_argument("--exposure-mode",
                   choices=("reset", "free", "dagger",
                            "staged_free"),
                   default="reset")
    p.add_argument("--rail-weights", default="1,1",
                   help="lambda_anchor,lambda_free")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="warm-start from a student.pt checkpoint "
                        "(C1 late fine-tune)")
    p.add_argument("--collector-from", type=Path, default=None,
                   help="frozen collector model for staged_free "
                        "exposure (e.g. the B1 checkpoint)")
    p.add_argument("--target-state", choices=("h40", "h42"),
                   default="h40",
                   help="h42: the recurrence/delta is built on the "
                        "h42 trajectory (direct-h42 drafter, "
                        "roadmap S34); export artifacts carry "
                        "target_state metadata")
    p.add_argument("--h42-aux-weight", type=float, default=0.0,
                   help=">0: add a same-position h42 auxiliary "
                        "head+loss (downstream alignment, S31); "
                        "only valid with --target-state h40")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    if a.h42_aux_weight > 0 and a.target_state != "h40":
        raise ValueError("--h42-aux-weight requires --target-state h40")
    captures = load_captures(
        a.data, json.loads(a.manifest.read_text()),
        a.categories.split(",") if a.categories else None, HORIZON)
    train = [c for c in captures if c["split"] == "train"]
    val = [c for c in captures if c["split"] == "val"]
    print(f"train {len(train)} val {len(val)} prompts", flush=True)
    if a.target_state == "h42":
        # Direct-h42 drafter: all state machinery operates on the
        # h42 trajectory (positions/alignment identical to h40).
        # Shallow-copy captures with the trajectory swapped; the
        # original h40 field stays available under "h40_orig".
        def swap(cs):
            out = []
            for c in cs:
                c2 = dict(c)
                c2["h40_orig"] = c["h40"]
                c2["h40"] = c["h42"]
                out.append(c2)
            return out
        train, val = swap(train), swap(val)
        print("target-state h42: recurrence built on h42 "
              "trajectory", flush=True)
    embed = load_embed()
    aux_field = "h42" if a.h42_aux_weight > 0 else None
    # NOTE: under target-state h42 the swap above means aux_field
    # is None (aux only valid for h40 arms).
    tr = build_transitions(train, embed, "train",
                           aux_field=aux_field)
    va = build_transitions(val, embed, "val")
    print(f"transitions: train {tr['x'].shape[0]} "
          f"val {va['x'].shape[0]}", flush=True)

    if a.resume_from is not None:
        ck = torch.load(a.resume_from, map_location="cpu",
                        weights_only=True)
        model = DeltaNet(ck["hidden"], ck["depth"], ck["rank"],
                         aux_out=ck.get("h42_aux_weight", 0) > 0
                         or "aux.weight" in ck["model"])
        model.load_state_dict(ck["model"])
        mean, std = ck["target_mean"], ck["target_std"]
        x_mean, x_std = ck["x_mean"], ck["x_std"]
        aux_mean = ck.get("aux_mean")
        aux_std = ck.get("aux_std")
        hist = [{"resumed_from": str(a.resume_from)}]
        print(f"resumed from {a.resume_from} "
              f"(hidden={ck['hidden']})", flush=True)
    else:
        (model, mean, std, hist, x_mean, x_std, aux_mean, aux_std
         ) = train_model(tr, va, a.hidden, a.depth, a.rank,
                         a.steps, a.batch, a.lr,
                         aux_weight=a.h42_aux_weight)
    if a.exposure_rounds > 0:
        tsched = None
        if a.threshold_decay == "relax":
            def tsched(r, n):
                return 0.75 * (4.0) ** ((r - 1) / max(n - 1, 1))
        hsched = None
        if a.horizon_curriculum:
            def hsched(r, n):
                phases = (2, 4, 8, 16, 32)
                return phases[min(5 * (r - 1) // n, 4)]
        la, lf = (float(x) for x in a.rail_weights.split(","))
        collector = None
        if a.collector_from is not None:
            ck = torch.load(a.collector_from, map_location="cpu",
                            weights_only=True)
            collector = DeltaNet(ck["hidden"], ck["depth"],
                                 ck["rank"],
                                 aux_out=ck.get("h42_aux_weight", 0)
                                 > 0 or "aux.weight" in ck["model"])
            collector.load_state_dict(ck["model"])
            collector.eval()
        hist2 = train_exposure(model, tr, embed, mean, std, x_mean,
                               x_std, a.exposure_rounds,
                               a.exposure_steps, a.lr * 0.3,
                               threshold_schedule=tsched,
                               horizon_schedule=hsched,
                               prefix_weight=a.prefix_weight,
                               telemetry_out=a.telemetry_out,
                               exposure_mode=a.exposure_mode,
                               rail_weights=(la, lf),
                               collector_model=collector)
        hist.extend(hist2)

    a.out.mkdir(parents=True, exist_ok=True)
    h_saved = ck["hidden"] if a.resume_from else a.hidden
    torch.save({"model": model.state_dict(), "hidden": h_saved,
                "depth": a.depth, "rank": a.rank,
                "target_state": a.target_state,
                "h42_aux_weight": a.h42_aux_weight,
                "aux_mean": aux_mean, "aux_std": aux_std,
                "target_mean": mean, "target_std": std,
                "x_mean": x_mean, "x_std": x_std,
                "args": {k: str(v) for k, v in vars(a).items()},
                "seed": 104729,
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(
                    a.manifest.read_bytes()).hexdigest(),
                "data_dir": str(a.data),
                "train_categories": [c["category"] for c in train],
                "val_categories": [c["category"] for c in val],
                "history": hist},
               a.out / "student.pt")

    # diagnostics + prediction export (same schema as d3_student)
    rows = []
    for pm in va["meta"]:
        c = pm["capture"]
        s, k = pm["S"], pm["K"]
        root = c["h40"][s - 1]
        preds = {
            "hold": root.expand(k, -1, -1).clone(),
            "one_step": rollout_onestep(model, pm, embed, std, mean,
                                        x_mean, x_std),
            "tf_chain": rollout_chain(model, pm, embed, std, mean,
                                      x_mean, x_std),
        }
        target = c["h40"][s:s + k].float().flatten(1)
        diag = {}
        for name, pred in preds.items():
            pf = pred.float().flatten(1)
            diag[name] = {
                "rel_l2_mean": rel_l2(pf, target).mean().item(),
                "cos_mean": torch.nn.functional.cosine_similarity(
                    pf, target, dim=1).mean().item(),
            }
        artifact = {key: c[key] for key in (
            "category", "split", "S", "tokens", "prompt",
            "snapshot_revision", "capture_path", "capture_sha256",
            "prompt_sha256")}
        artifact.update({
            "schema_version": 1, "K": k, "verification_width_N": None,
            "target_state": a.target_state,
            "h40_committed": c["h40"][:s].clone(),
            "predicted_h40": preds,
            "h40_target": c["h40"][s:s + k].clone(),
            "h42_target": c["h42"][:s + k].clone(),
            "chunk": None,
            "chunk_feedback": "none; one-step token-conditioned "
                              "teacher-forced export",
            "checkpoint_path": str((a.out / "student.pt").resolve()),
            "student_cpu_timing": None,
        })
        torch.save(artifact, a.out / f"{c['category']}.predictions.pt")
        rows.append({"category": c["category"], "K": k,
                     "state_diagnostics": diag})
        print(c["category"], json.dumps(diag), flush=True)

    report = {
        "classification": "D3_V1_TOKEN_CONDITIONED_DELTA_STUDENT",
        "student_pass": False, "production_certified": False,
        "source_tail_acceptance": None,
        "architecture": f"DeltaNet hidden={a.hidden} depth={a.depth} "
                        f"rank={a.rank or 'full'}",
        "training": "teacher-forced one-step; per-dim standardized "
                    "delta MSE; final-epoch export (no metric "
                    "selection)",
        "train_prompts": [c["category"] for c in train],
        "val_prompts": [c["category"] for c in val],
        "train_transitions": tr["x"].shape[0],
        "val_transitions": va["x"].shape[0],
        "loss_history": hist,
        "per_prompt": rows,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
    }
    (a.out / "diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": str(a.out),
                      "train_transitions": tr["x"].shape[0]}))


if __name__ == "__main__":
    main()
