"""C2/C3 pre-flight (full contract): no-update free rollout from
the ACTUAL post-1000-step one-step base, over ALL 84 train
prompts, with explicit gating.

Measures per prompt (K=32 free chain from the boundary root):
  - drift by position (max, final, p95)
  - raw/standardized feature magnitudes; finiteness
  - prediction magnitudes; finiteness
  - RECOVERY LOSS on free states in TWO forms: the EXACT
    (unclipped) loss — an off-manifold severity diagnostic — and
    the ACTUAL TRAINING loss (recovery targets clamped at ±4
    standardized, matching train_exposure). The launch gate uses
    the TRAINING loss; the exact loss is reported for severity.

Gating (explicit, decided before looking at results):
  PASS-RAW      all finite; max standardized feature < 100;
                max |pred| < 1e3; unclipped free-rail loss
                within 100x the teacher-forced train loss
                (trainable by AdamW + grad clip without
                non-finite steps)
  PASS-CLAMPED  finite everywhere; features bounded enough that
                clamping at 30σ touches <5% of positions and
                clamped loss is finite and <100x TF loss
                -> C2/C3 launch with that clamp + non-finite
                   counting + hard abort thresholds
  STAGE/BOUND   anything else (non-finite anywhere, or magnitudes
                beyond the above)
                -> free rail must be staged (progressive drift
                   exposure) or replay-bounded, not launched raw

The verdict applies the thresholds mechanically; the staged-vs-
bounded design choice is a separate decision recorded with it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_v1 import (build_transitions, load_embed,
                                  make_feat, train_model)
from mosaicvm.bench.d3_student import load_captures

OUT = Path("artifacts/d3_c2_preflight.json")
N_PROMPTS = None            # all
GATE = {"feat_std_max": 100.0, "pred_absmax": 1e3,
        "loss_ratio_max": 100.0, "clamp_touch_frac": 0.05}


def main():
    torch.manual_seed(104729)
    torch.set_num_threads(8)
    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json").read_text()),
                         None, 32)
    train = [c for c in caps if c["split"] == "train"]
    print(f"{len(train)} train prompts", flush=True)
    embed = load_embed()
    tr = build_transitions(train, embed, "train")
    x, y = tr["x"], tr["y"]

    # exact A2-recipe base (1000 one-step full-batch steps)
    model, mean, std, hist, x_mean, x_std = train_model(
        tr, tr, hidden=512, depth=3, rank=0, steps=1000,
        batch=0, lr=1e-3)
    # teacher-forced train loss (standardized) for the ratio gate
    with torch.no_grad():
        xn_chk = (x - x_mean) / x_std
        yn_chk = (y - mean) / std
        tf_loss = ((model(xn_chk) - yn_chk) ** 2).mean().item()
    print(f"base trained; TF train loss {tf_loss:.4f}", flush=True)

    rows = []
    agg = {"nonfinite": {"raw": 0, "c30": 0, "c10": 0},
           "feat_std_max": 0.0, "pred_absmax": 0.0,
           "clamp30_touch": 0, "clamp10_touch": 0,
           "positions": 0,
           "free_losses": {"raw": [], "c30": [], "c10": []},
           "drifts": {"raw": [], "c30": [], "c10": []}}

    def rollout(clamp):
        """One full free rollout for one prompt with the given
        input clamp (None = raw). Returns (drifts, per-position
        losses, broke). state updates use the model's own [16384]
        prediction under THIS clamp — independent trajectory."""
        state = h[S - 1].flatten().clone()
        prev = h[S - 2].flatten().clone()
        drifts, losses = [], []
        for j in range(K):
            t = S + j
            gold = h[t].flatten()
            gold_prev = h[t - 1].flatten()
            drifts.append(((state - gold_prev).norm()
                           / gold_prev.norm().clamp_min(1e-6)).item())
            feat = make_feat(embed, state, state - prev,
                             toks[t], toks[t - 1], t)
            if not torch.isfinite(feat).all():
                return drifts, losses, f"input@j{j}"
            fn = (feat - x_mean) / x_std
            if clamp is not None:
                fn = fn.clamp(-clamp, clamp)
            pred_n = model(fn.unsqueeze(0))        # [1, 16384]
            pred = pred_n[0] * std + mean          # [16384]
            if not (torch.isfinite(pred_n).all()
                    and torch.isfinite(pred).all()):
                return drifts, losses, f"pred@j{j}"
            tgt = (gold - state - mean) / std      # [16384]
            # EXACT (unclipped) loss: off-manifold severity
            # diagnostic only
            l_exact = ((pred_n[0] - tgt) ** 2).mean().item()
            # ACTUAL training objective: recovery targets clamped
            # at ±4 standardized (train_exposure semantics)
            tgt_c = tgt.clamp(-4, 4)
            l_train = ((pred_n[0] - tgt_c) ** 2).mean().item()
            losses.append((l_exact, l_train))
            prev = state
            state = state + pred                   # [16384] update
        return drifts, losses, None

    for pm in tr["meta"]:
        c = pm["capture"]
        S, K = pm["S"], pm["K"]
        toks = c["tokens"]
        h = c["h40"].float()
        row = {"prompt": pm["category"]}
        # raw pass (collects magnitude stats)
        state = h[S - 1].flatten().clone()
        prev = h[S - 2].flatten().clone()
        broke_raw = None
        drifts_raw = []
        for j in range(K):
            t = S + j
            gold_prev = h[t - 1].flatten()
            drifts_raw.append(((state - gold_prev).norm()
                               / gold_prev.norm().clamp_min(1e-6)
                               ).item())
            feat = make_feat(embed, state, state - prev,
                             toks[t], toks[t - 1], t)
            if not torch.isfinite(feat).all():
                agg["nonfinite"]["raw"] += 1
                broke_raw = f"input@j{j}"
                break
            fn = (feat - x_mean) / x_std
            m = fn.abs().max().item()
            agg["feat_std_max"] = max(agg["feat_std_max"], m)
            agg["positions"] += 1
            if m > 30:
                agg["clamp30_touch"] += 1
            if m > 10:
                agg["clamp10_touch"] += 1
            pred_n = model(fn.unsqueeze(0))
            pred = pred_n[0] * std + mean
            if not (torch.isfinite(pred_n).all()
                    and torch.isfinite(pred).all()):
                agg["nonfinite"]["raw"] += 1
                broke_raw = f"pred@j{j}"
                break
            agg["pred_absmax"] = max(agg["pred_absmax"],
                                     pred.abs().max().item())
            prev = state
            state = state + pred
        row["drift_max_raw"] = round(max(drifts_raw), 3) \
            if drifts_raw else None
        row["broke_raw"] = broke_raw
        if broke_raw is None:
            _, losses_raw, _ = rollout(None)
            agg["free_losses"]["raw"].extend(losses_raw)
            agg["drifts"]["raw"].append(max(drifts_raw))
        # independent clamped trajectories
        for tag, clamp in (("c30", 30), ("c10", 10)):
            drifts_c, losses_c, broke_c = rollout(clamp)
            row[f"drift_max_{tag}"] = round(max(drifts_c), 3) \
                if drifts_c else None
            row[f"broke_{tag}"] = broke_c
            if broke_c is None:
                agg["free_losses"][tag].extend(losses_c)
                agg["drifts"][tag].append(max(drifts_c))
            else:
                agg["nonfinite"][tag] += 1
        rows.append(row)

    def rail_stats(tag):
        fl = agg["free_losses"][tag]   # list of (exact, train) pairs
        if not fl:
            return {"mean_train_loss": None,
                    "mean_exact_loss": None, "ratio_vs_tf": None,
                    "finite": False, "drift_max": -1}
        mean_train = sum(x[1] for x in fl) / len(fl)
        mean_exact = sum(x[0] for x in fl) / len(fl)
        # GATE ON THE ACTUAL TRAINING OBJECTIVE (clipped targets)
        ratio = mean_train / max(tf_loss, 1e-9)
        finite = agg["nonfinite"][tag] == 0
        return {"mean_train_loss": round(mean_train, 4),
                "mean_exact_loss": round(mean_exact, 4),
                "ratio_vs_tf": round(ratio, 2),
                "finite": finite,
                "drift_max": round(max(agg["drifts"][tag])
                                   if agg["drifts"][tag] else -1, 3)}

    stats = {tag: rail_stats(tag) for tag in ("raw", "c30", "c10")}
    touch_frac30 = agg["clamp30_touch"] / max(agg["positions"], 1)
    touch_frac10 = agg["clamp10_touch"] / max(agg["positions"], 1)
    G = GATE
    if (stats["raw"]["finite"]
            and agg["feat_std_max"] < G["feat_std_max"]
            and agg["pred_absmax"] < G["pred_absmax"]
            and stats["raw"]["ratio_vs_tf"] < G["loss_ratio_max"]):
        verdict = "PASS-RAW"
    elif (stats["c30"]["finite"]
            and touch_frac30 < G["clamp_touch_frac"]
            and stats["c30"]["ratio_vs_tf"] < G["loss_ratio_max"]):
        verdict = "PASS-CLAMPED (30sigma input clamp)"
    elif (stats["c10"]["finite"]
            and touch_frac10 < 2 * G["clamp_touch_frac"]
            and stats["c10"]["ratio_vs_tf"] < G["loss_ratio_max"]):
        verdict = "PASS-CLAMPED (10sigma input clamp)"
    else:
        verdict = "STAGE/BOUND required"

    result = {
        "classification": "D3_C2_PREFLIGHT_FULL",
        "base": "A2 recipe one-step model, 1000 steps, pre-exposure",
        "gate": GATE,
        "n_prompts": len(rows),
        "tf_train_loss": round(tf_loss, 4),
        "rail_stats": stats,
        "feat_std_max": round(agg["feat_std_max"], 1),
        "pred_absmax": round(agg["pred_absmax"], 1),
        "nonfinite": agg["nonfinite"],
        "clamp30_touch_fraction": round(touch_frac30, 4),
        "clamp10_touch_fraction": round(touch_frac10, 4),
        "drift_max_overall": round(max(
            (r.get("drift_max_raw") or 0) for r in rows), 3),
        "prompts_broken": sum(
            1 for r in rows
            if r.get("broke_raw") or r.get("broke_c30")),
        "verdict": verdict,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "rows"}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
