"""TCS-FD projection oracle: gold-assisted correction along
per-position TRUE DELTA directions (the FD scan's signal).

The FD scan found:
  delta-own (true h40 motion at same position): 6.64x random
  own-err (A2 error at same position):           2.15x random
  everything else:                               ~1x random

This oracle builds a basis from PER-POSITION true deltas (the
highest-sensitivity directions) and tests whether correcting
A2's predicted h40 along those directions improves exact-tail
decisions.

Correction: h40_corrected = h40_pred - lambda * B_pos @ B_pos^T @ e
where B_pos is the rank-r basis of true deltas AT THAT POSITION's
domain/context, and e = pred - gold (oracle-only gold info).

Basis construction options tested:
  A) per-position delta: B = [delta_t] rank-1 (the single delta
     at this position — the highest-sensitivity direction)
  B) domain-delta SVD: rank-r SVD of all deltas within the
     prompt's domain
  C) global-delta SVD: rank-r SVD of all train deltas

Kill gate: dS1 >= 0.10 (same as the error-SVD oracle).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import DeltaNet, make_feat, HORIZON
from mosaicvm.bench.d1b_tail_robustness import prefix_metrics
from mosaicvm.bench.source_tail import SourceTail
from mosaicvm.bench.d3_bridge import load_embed

TEACHER = Path("artifacts/d3_r2_teacher")
OUT = Path("artifacts/d3_tcs_fd_projection.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambdas", default="0.25,0.5,1.0")
    a = p.parse_args()
    lams = [float(x) for x in a.lambdas.split(",")]
    torch.set_num_threads(4)
    torch.manual_seed(104729)

    t0 = time.time()
    ck = torch.load("artifacts/d3_v1_A2/student.pt",
                    map_location="cpu", weights_only=True)
    a2 = DeltaNet(ck["hidden"], ck["depth"], ck["rank"])
    a2.load_state_dict(ck["model"])
    a2.eval()
    t_mean, t_std = ck["target_mean"], ck["target_std"]
    x_mean, x_std = ck["x_mean"], ck["x_std"]
    embed = load_embed()

    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    train = [c for c in caps if c["split"] == "train"]
    print(f"val {len(val)} / train {len(train)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    tail = SourceTail(device="cuda", backend="reference",
                      max_seq=64, cache_cap=96)

    @torch.no_grad()
    def a2_one_step(c):
        S = c["S"]
        K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]
        h = c["h40"].float()
        preds = []
        for j in range(K):
            t = S + j
            feat = make_feat(embed, h[t - 1].flatten(),
                             h[t - 1].flatten() - h[t - 2].flatten(),
                             toks[t], toks[t - 1], t)
            fn = (feat - x_mean) / x_std
            pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
            preds.append(h[t - 1].flatten() + pred)
        return torch.stack(preds).view(K, 4, 4096), K

    def eval_traj(c, h40_test, K):
        S = c["S"]
        exact = torch.cat([c["h40"][:S], h40_test])
        h42 = tail.replay(exact, c["tokens"], S)
        lg = tail.logits(h42)
        target = torch.tensor(c["tokens"][S + 1:S + K + 1],
                              dtype=torch.long)
        hits = (lg[S:S + K].argmax(-1) == target).tolist()
        return prefix_metrics([int(x) for x in hits], K)

    # ---- build per-prompt delta references from TRAIN ----
    # For each val prompt, find same-domain train prompts and
    # use their delta vectors as the basis pool
    train_by_dom = {}
    for c in train:
        d = c["category"].rsplit("_", 1)[0]
        h40_gold = c["h40"][c["S"]:].float()
        deltas = (h40_gold[1:] - h40_gold[:-1]).flatten(1)
        train_by_dom.setdefault(d, []).append(deltas)
    pools = {k: len(v) for k, v in train_by_dom.items()}
    print(f"delta pools: {pools} ({time.time()-t0:.0f}s)",
          flush=True)

    results = {"classification": "D3_TCS_FD_PROJECTION_ORACLE",
               "lambdas": lams, "methods": {}, "baseline": {},
               "production_timing": None}

    # ---- baseline (uncorrected) ----
    base_rows = []
    for c in val:
        h40_pred, K = a2_one_step(c)
        m = eval_traj(c, h40_pred, K)
        base_rows.append(m)
    base_ea = sum(r["E_A"] for r in base_rows) / len(base_rows)
    base_s1 = sum(r["Sk"][0] for r in base_rows) / len(base_rows)
    results["baseline"] = {"mean_E_A": round(base_ea, 4),
                           "S1": round(base_s1, 4)}
    print(f"baseline: E[A]={base_ea:.3f} S1={base_s1:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- method A: per-position rank-1 delta correction ----
    # For each position j, the basis is the TRUE delta at j
    # (from the gold trajectory — oracle info). This is the
    # highest-sensitivity direction per the FD scan.
    for lam in lams:
        key = f"per-pos-delta_lam{lam}"
        rows = []
        for c in val:
            S = c["S"]
            h40_pred, K = a2_one_step(c)
            h40_gold = c["h40"][S:S + K].float()
            corrected = []
            for j in range(K):
                if j < K - 1:
                    delta_j = (h40_gold[j + 1]
                               - h40_gold[j]).flatten()
                else:
                    delta_j = (h40_gold[j]
                               - h40_gold[j - 1]).flatten()
                e = (h40_pred[j].float().flatten()
                     - h40_gold[j].flatten())
                # project error onto the delta direction
                d_norm = delta_j.norm().clamp_min(1e-8)
                d_hat = delta_j / d_norm
                proj = (e @ d_hat) * d_hat
                corrected.append(
                    h40_pred[j].float().flatten()
                    - lam * proj)
            h40_corr = torch.stack(corrected).view(K, 4, 4096)
            m = eval_traj(c, h40_corr, K)
            rows.append(m)
        ea = sum(r["E_A"] for r in rows) / len(rows)
        s1 = sum(r["Sk"][0] for r in rows) / len(rows)
        results["methods"][key] = {
            "mean_E_A": round(ea, 4), "S1": round(s1, 4),
            "delta_S1": round(s1 - base_s1, 4),
            "delta_E_A": round(ea - base_ea, 4)}
        print(f"  {key}: E[A]={ea:.3f} S1={s1:.3f} "
              f"(dS1={s1-base_s1:+.3f}) "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- method B: domain-delta SVD rank-8 ----
    for lam in lams:
        key = f"dom-svd8_lam{lam}"
        rows = []
        for c in val:
            S = c["S"]
            h40_pred, K = a2_one_step(c)
            h40_gold = c["h40"][S:S + K].float()
            dom = c["category"].rsplit("_", 1)[0]
            # pool all train deltas from this domain
            pool = torch.cat(train_by_dom.get(dom, []))
            # SVD: top-8 right singular vectors
            _, _, Vh = torch.linalg.svd(
                pool, full_matrices=False)
            B = Vh[:8].t()               # [16384, 8]
            corrected = []
            for j in range(K):
                e = (h40_pred[j].float().flatten()
                     - h40_gold[j].flatten())
                proj = B @ (B.t() @ e)
                corrected.append(
                    h40_pred[j].float().flatten()
                    - lam * proj)
            h40_corr = torch.stack(corrected).view(K, 4, 4096)
            m = eval_traj(c, h40_corr, K)
            rows.append(m)
        ea = sum(r["E_A"] for r in rows) / len(rows)
        s1 = sum(r["Sk"][0] for r in rows) / len(rows)
        results["methods"][key] = {
            "mean_E_A": round(ea, 4), "S1": round(s1, 4),
            "delta_S1": round(s1 - base_s1, 4),
            "delta_E_A": round(ea - base_ea, 4)}
        print(f"  {key}: E[A]={ea:.3f} S1={s1:.3f} "
              f"(dS1={s1-base_s1:+.3f}) "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- kill gate ----
    best = max(results["methods"].values(),
               key=lambda x: x.get("delta_S1", 0))
    best_key = max(results["methods"],
                   key=lambda k: results["methods"][k]
                   .get("delta_S1", 0))
    results["best"] = {"key": best_key, **best}
    results["kill_gate"] = {
        "threshold": 0.10,
        "best_delta_S1": best.get("delta_S1", 0),
        "passed": best.get("delta_S1", 0) >= 0.10}
    results["verdict"] = (
        "PASS — FD-basis correction materially improves "
        "decisions; TCS-FD student training justified"
        if best.get("delta_S1", 0) >= 0.10 else
        "FAIL — even FD-derived correction does not improve "
        "decisions enough; TCS-FD CLOSED")

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\n=== FD PROJECTION VERDICT: {results['verdict']} ===")
    print(f"best dS1 = {best.get('delta_S1', 0):+.3f} at "
          f"{best_key} (gate >= 0.10)")
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
