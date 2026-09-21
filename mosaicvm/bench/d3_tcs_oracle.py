"""TCS-4: gold-assisted projection oracle (roadmap S28-S30).

THE kill gate for the entire TCS program. Before training any
student on a tail-sensitive objective, test whether correcting
A2's predicted-h40 errors in a low-rank basis ACTUALLY improves
exact-tail decisions.

Method:
  1. Build candidate bases (rank r) from TRAIN-split A2 errors
     via SVD: e_t = h40_pred_os[t] - h40_gold[t], flattened to
     [16384]. Right singular vectors = basis directions.
  2. For each VAL prompt, take A2's one-step predicted h40
     trajectory and apply gold-assisted correction:
       h40_corrected = h40_pred - lambda * B @ B^T @ e
     where e = h40_pred - h40_gold (the TRUE error, oracle-only).
  3. Replay the corrected trajectory through the EXACT source
     L41/L42 tail + FP32 head; measure prefix survival.
  4. Compare with the uncorrected baseline (A2 os-variant).

If low-rank correction materially improves exact-tail decisions,
the TCS hypothesis is validated. If not, TCS is closed.

The oracle uses gold information ONLY for the correction (never
in production); this is an upper bound on what a TCS-trained
student could achieve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import DeltaNet, make_feat, HORIZON
from mosaicvm.bench.d1b_tail_robustness import prefix_metrics, \
    prompt_mean_ci
from mosaicvm.bench.source_tail import SourceTail
from mosaicvm.bench.d3_bridge import load_embed

TEACHER = Path("artifacts/d3_r2_teacher")
AUDIT = Path("artifacts/d3_r2_teacher_audit.json")
OUT = Path("artifacts/d3_tcs_oracle.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranks", default="8,16,32")
    p.add_argument("--lambdas", default="0.25,0.5,1.0")
    a = p.parse_args()
    ranks = [int(x) for x in a.ranks.split(",")]
    lams = [float(x) for x in a.lambdas.split(",")]
    torch.set_num_threads(4)

    t0 = time.time()
    # ---- hash gate ----
    ver = json.loads(AUDIT.read_text())
    assert ver["verdict"] == "PASS"
    for cat, h in list(ver["record_hashes_full"].items())[:5]:
        cur = hashlib.sha256(
            (TEACHER / f"{cat}.pt").read_bytes()).hexdigest()
        assert cur == h, f"teacher drift {cat}"
    print(f"hash gate: PASS ({time.time()-t0:.0f}s)", flush=True)

    # ---- load A2 (for generating predicted trajectories on val) ----
    ck = torch.load("artifacts/d3_v1_A2/student.pt",
                    map_location="cpu", weights_only=True)
    a2 = DeltaNet(ck["hidden"], ck["depth"], ck["rank"])
    a2.load_state_dict(ck["model"])
    a2.eval()
    t_mean, t_std = ck["target_mean"], ck["target_std"]
    x_mean, x_std = ck["x_mean"], ck["x_std"]
    embed = load_embed()

    # ---- build error bases from TRAIN split ----
    train_errors = []   # [n_train_positions, 16384]
    for cat in sorted(ver["record_hashes_full"]):
        rec = torch.load(TEACHER / f"{cat}.pt",
                         map_location="cpu", weights_only=True)
        for j in range(rec["K"]):
            e = (rec["h40_pred_os"][j].float()
                 - rec["h40_gold_future"][j].float()).flatten()
            train_errors.append(e)
    E = torch.stack(train_errors)         # [n_train, 16384]
    print(f"train error matrix: {tuple(E.shape)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    # SVD → right singular vectors = basis directions
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    bases = {}
    for r in ranks:
        bases[r] = Vh[:r].t()             # [16384, r]
        print(f"  rank {r}: top singular values "
              f"{S[:min(r,4)].tolist()[:4]}", flush=True)

    # ---- load val captures ----
    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    print(f"val prompts: {len(val)} ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- exact tail ----
    tail = SourceTail(device="cuda", backend="reference",
                      max_seq=64, cache_cap=96)

    # ---- generate A2 one-step predictions on val + evaluate ----
    @torch.no_grad()
    def a2_one_step_trajectory(c):
        """A2 one-step predictions on a val capture."""
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

    def eval_trajectory(c, h40_pred, K):
        """Replay a predicted trajectory through the exact tail;
        return prefix-survival metrics."""
        S = c["S"]
        exact = torch.cat([c["h40"][:S], h40_pred])
        h42 = tail.replay(exact, c["tokens"], S)
        lg = tail.logits(h42)
        # K speculative decisions EXCLUDING the exact root:
        # lg[S+j] predicts tokens[S+j+1], j = 0..K-1
        target = torch.tensor(c["tokens"][S + 1:S + K + 1],
                              dtype=torch.long)
        hits = (lg[S:S + K].argmax(-1) == target).tolist()
        return prefix_metrics([int(x) for x in hits], K)

    results = {"classification": "D3_TCS_PROJECTION_ORACLE",
               "ranks": ranks, "lambdas": lams,
               "baseline": {}, "corrected": {},
               "kill_gate": "Delta S1 >= 0.10 at rank <= 32",
               "production_timing": None}

    # baseline: uncorrected A2 predictions
    print(f"\n=== baseline (uncorrected A2 one-step) ===",
          flush=True)
    base_rows = []
    for c in val:
        h40_pred, K = a2_one_step_trajectory(c)
        m = eval_trajectory(c, h40_pred, K)
        base_rows.append({"cat": c["category"], "m": m})
        print(f"  {c['category']}: E[A]={m['E_A']}", flush=True)
    base_ea = sum(r["m"]["E_A"] for r in base_rows) / len(base_rows)
    base_s1 = sum(r["m"]["Sk"][0] for r in base_rows) / len(base_rows)
    results["baseline"] = {
        "mean_E_A": round(base_ea, 4), "S1": round(base_s1, 4)}
    print(f"  BASELINE: mean E[A]={base_ea:.3f} "
          f"S1={base_s1:.3f}", flush=True)

    # corrected: for each rank and lambda
    for r in ranks:
        B = bases[r]                        # [16384, r]
        for lam in lams:
            key = f"r{r}_lam{lam}"
            print(f"\n=== {key} ===", flush=True)
            rows = []
            for c in val:
                S = c["S"]
                h40_pred, K = a2_one_step_trajectory(c)
                h40_gold = c["h40"][S:S + K].float()
                # gold-assisted correction (ORACLE ONLY)
                corrected = []
                for j in range(K):
                    e = (h40_pred[j].float().flatten()
                         - h40_gold[j].flatten())
                    proj = B @ (B.t() @ e)
                    corrected.append(
                        h40_pred[j].float().flatten()
                        - lam * proj)
                h40_corr = torch.stack(corrected).view(K, 4, 4096)
                m = eval_trajectory(c, h40_corr, K)
                rows.append({"cat": c["category"], "m": m})
            ea = sum(r_["m"]["E_A"] for r_ in rows) / len(rows)
            s1 = sum(r_["m"]["Sk"][0] for r_ in rows) / len(rows)
            results["corrected"][key] = {
                "mean_E_A": round(ea, 4), "S1": round(s1, 4),
                "delta_S1": round(s1 - base_s1, 4),
                "delta_E_A": round(ea - base_ea, 4),
                "per_prompt": {r_["cat"]: r_["m"]["E_A"]
                                for r_ in rows}}
            print(f"  {key}: E[A]={ea:.3f} S1={s1:.3f} "
                  f"(dS1={s1-base_s1:+.3f} "
                  f"dEA={ea-base_ea:+.3f})", flush=True)

    # ---- kill gate evaluation ----
    best_dS1 = max(v.get("delta_S1", 0)
                   for v in results["corrected"].values())
    best_key = max(results["corrected"],
                   key=lambda k: results["corrected"][k]
                   .get("delta_S1", 0))
    results["best_correction"] = {
        "key": best_key,
        "delta_S1": best_dS1,
        "kill_gate_passed": best_dS1 >= 0.10}
    results["verdict"] = (
        "PASS — low-rank correction materially improves "
        "exact-tail decisions; TCS student training justified"
        if best_dS1 >= 0.10 else
        "FAIL — low-rank correction does not improve decisions; "
        "TCS CLOSED")

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\n=== ORACLE VERDICT: {results['verdict']} ===")
    print(f"best dS1 = {best_dS1:+.3f} at {best_key} "
          f"(gate: >= 0.10)")
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
