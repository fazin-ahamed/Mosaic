"""TCS-FD alpha-sweep oracle: the direct decision-recovery curve.

Interpolates from A2's predicted h40 toward the gold state:
  h40_corrected = h40_pred + alpha * (h40_gold - h40_pred)

At alpha=0: A2's prediction (known acceptance).
At alpha=1.0: gold state (perfect by D1-C).

Sweeps alpha to measure WHERE decisions recover:
  - Sharp recovery at low alpha (<= 0.25): error concentrated
    in decision-critical directions → TCS-FD viable
  - Gradual recovery (only near alpha ~ 1): error diffuse
    → TCS-FD closed → direct decision representation

Also probes ORTHOGONAL interpolations (moving the same DISTANCE
in random directions) as a control: if random directions recover
equally, the recovery is just error-norm reduction, not
direction-specific correction.

Uses the exact source tail + FP32 head; canonical statistics.
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

OUT = Path("artifacts/d3_tcs_fd_oracle.json")
ALPHAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-prompts", type=int, default=21)
    p.add_argument("--control", action="store_true",
                   help="also probe random-direction controls")
    a = p.parse_args()
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
    val = [c for c in caps if c["split"] == "val"][:a.n_prompts]
    print(f"val prompts: {len(val)} ({time.time()-t0:.0f}s)",
          flush=True)

    tail = SourceTail(device="cuda", backend="reference",
                      max_seq=64, cache_cap=96)

    @torch.no_grad()
    def a2_predictions(c):
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

    def eval_traj(c, h40_pred, K):
        S = c["S"]
        exact = torch.cat([c["h40"][:S], h40_pred])
        h42 = tail.replay(exact, c["tokens"], S)
        lg = tail.logits(h42)
        target = torch.tensor(c["tokens"][S + 1:S + K + 1],
                              dtype=torch.long)
        hits = (lg[S:S + K].argmax(-1) == target).tolist()
        m = prefix_metrics([int(x) for x in hits], K)
        return m, hits

    results = {"classification": "D3_TCS_FD_ALPHA_ORACLE",
               "alphas": list(ALPHAS),
               "gold_interp": {}, "random_control": {},
               "production_timing": None}

    # ---- gold-interpolation sweep ----
    for alpha in ALPHAS:
        rows = []
        all_hits = []
        for c in val:
            h40_pred, K = a2_predictions(c)
            h40_gold = c["h40"][c["S"]:c["S"] + K].float()
            if alpha == 0.0:
                h40_test = h40_pred
            elif alpha == 1.0:
                h40_test = h40_gold.to(torch.bfloat16)
            else:
                h40_test = (h40_pred.float()
                            + alpha * (h40_gold - h40_pred.float()
                                       )).to(torch.bfloat16)
            m, hits = eval_traj(c, h40_test, K)
            rows.append(m)
            all_hits.extend(hits)
        ea = sum(r["E_A"] for r in rows) / len(rows)
        s1 = sum(r["Sk"][0] for r in rows) / len(rows)
        results["gold_interp"][str(alpha)] = {
            "mean_E_A": round(ea, 4), "S1": round(s1, 4)}
        print(f"alpha={alpha:.3f}: E[A]={ea:.3f} S1={s1:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- random-direction control (same norms, wrong direction) --
    if a.control:
        for alpha in (0.25, 0.5):
            rows = []
            for c in val:
                h40_pred, K = a2_predictions(c)
                h40_gold = c["h40"][c["S"]:c["S"] + K].float()
                # random direction with SAME per-position norm
                # as the gold-pred difference
                diff_norm = (h40_gold - h40_pred.float()
                             ).flatten(1).norm(dim=1)
                rand = torch.randn_like(h40_pred.float())
                rf = rand.flatten(1)
                rf = rf / rf.norm(dim=1,
                                  keepdim=True).clamp_min(1e-6)
                rf = rf * diff_norm.view(-1, 1)
                rand = rf.view_as(h40_pred)
                h40_test = (h40_pred.float()
                            + alpha * rand).to(torch.bfloat16)
                m, _ = eval_traj(c, h40_test, K)
                rows.append(m)
            ea = sum(r["E_A"] for r in rows) / len(rows)
            s1 = sum(r["Sk"][0] for r in rows) / len(rows)
            results["random_control"][str(alpha)] = {
                "mean_E_A": round(ea, 4), "S1": round(s1, 4)}
            print(f"random alpha={alpha:.3f}: E[A]={ea:.3f} "
                  f"S1={s1:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)

    # ---- analysis ----
    g = results["gold_interp"]
    base_ea = g["0.0"]["mean_E_A"]
    gold_ea = g["1.0"]["mean_E_A"]
    # recovery fraction at each alpha
    recovery = {a_: round((g[a_]["mean_E_A"] - base_ea)
                          / max(gold_ea - base_ea, 1e-6), 4)
                for a_ in g}
    results["recovery_fraction"] = recovery
    # TCS-FD gate: does alpha=0.25 recover >= 50% of the gap?
    r25 = recovery.get("0.25", 0)
    results["gate"] = {
        "alpha_0.25_recovery": r25,
        "threshold": 0.5,
        "passed": r25 >= 0.5}
    results["verdict"] = (
        "TCS-FD VIABLE — decisions recover at low correction "
        "fraction; error is concentrated in decision-critical "
        "directions" if r25 >= 0.5 else
        "TCS-FD MARGINAL/CLOSED — gradual recovery; error is "
        "diffuse across decision-irrelevant directions")

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\n=== TCS-FD VERDICT: {results['verdict']} ===")
    print(f"recovery at alpha=0.25: {r25:.1%} "
          f"(gate: >= 50%)")
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
