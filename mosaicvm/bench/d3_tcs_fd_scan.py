"""TCS-2: exact-tail finite-difference sensitivity scan.

Probes the REAL source L41/L42 tail's decision-margin response
to perturbations along candidate directions, at multiple epsilon
values. This is the experiment the interpolation oracle could
NOT substitute for: it measures DIRECTION-LEVEL sensitivity,
not full-vector recovery.

At each probe position t (with exact KV prefix from gold states):
  h40_test = h40_gold_t +/- eps * b_i
  margin = l_y(tail(h40_test)) - l_top_comp(tail(h40_test))
  FD_sensitivity = (m(+) - m(-)) / (2*eps)

Candidate direction types (8 per position):
  own-err    A2's error at THIS position
  x-err      A2 errors from OTHER positions (2)
  delta      true h40 deltas at this + other positions (2)
  random     isotropic Gaussian (3) — sensitivity baseline

Telemetry per probe:
  margin change, greedy flip, rank change, route changes (L41/L42)

If random directions show the SAME sensitivity as error/delta
directions, the tail is isotropically sensitive and no basis
helps. If error/delta directions are MORE sensitive, TCS-FD has
signal and the projection oracle should run on an FD basis.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import DeltaNet, make_feat, HORIZON
from mosaicvm.bench.source_tail import SourceTail
from mosaicvm.bench.d3_bridge import load_embed
from mosaicvm.bench.d1b_tail_robustness import reset_decode_state

TEACHER = Path("artifacts/d3_r2_teacher")
OUT = Path("artifacts/d3_tcs_fd_scan.json")
N_PROMPTS = 7          # one per domain
POSITIONS_PER_PROMPT = 2  # early + mid
EPS_ALPHAS = (0.5, 1.0)   # fraction of the direction's own norm
D_STATE = 16384


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(104729)

    # ---- load A2 + embed for error computation ----
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
    # one per domain
    by_dom = {}
    for c in val:
        d = c["category"].rsplit("_", 1)[0]
        if d not in by_dom:
            by_dom[d] = c
    prompts = list(by_dom.values())[:N_PROMPTS]
    print(f"{len(prompts)} prompts (one per domain) "
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

    # ---- collect candidate directions per prompt ----
    # precompute A2 errors for cross-prompt directions
    all_errors = {}  # cat -> [K, 16384]
    all_deltas = {}
    for c in prompts:
        h40_pred, K = a2_one_step(c)
        h40_gold = c["h40"][c["S"]:c["S"] + K].float()
        err = (h40_pred.float() - h40_gold).flatten(1)
        delta = (h40_gold[1:] - h40_gold[:-1]).flatten(1)
        all_errors[c["category"]] = err
        all_deltas[c["category"]] = delta

    other_cats = list(all_errors.keys())

    def probe_margin(c, h40_pert, t_pos, S):
        """Full tail replay with ONE perturbed position; return
        the margin (true_logit - top_comp_logit) at position t."""
        K = min(HORIZON, c["h40"].shape[0] - S)
        # build trajectory: exact prefix + exact states up to t-1
        # + perturbed at t + exact for t+1.. (the perturbation
        # only affects position t's own decision via its h42)
        traj = c["h40"][:S + K].clone()
        traj[t_pos] = h40_pert.to(torch.bfloat16)
        h42 = tail.replay(traj, c["tokens"], S)
        lg = tail.logits(h42[t_pos:t_pos + 1])[0]
        true_tok = c["tokens"][t_pos + 1]
        tl = float(lg[true_tok])
        top_idx = int(lg.argmax())
        if top_idx == true_tok:
            # already correct: margin vs second-best
            sorted_lg = lg.sort(descending=True)
            comp = int(sorted_lg.indices[1])
        else:
            comp = top_idx
        margin = tl - float(lg[comp])
        return margin, (top_idx == true_tok)

    results = {"classification": "D3_TCS_FD_SCAN",
               "n_prompts": len(prompts),
               "probes": [], "summary": {}}
    type_sens = {}   # direction_type -> list of |FD| values

    for c in prompts:
        S = c["S"]
        K = min(HORIZON, c["h40"].shape[0] - S)
        cat = c["category"]
        h40_gold = c["h40"][S:S + K].float()
        err = all_errors[cat]
        delta = all_deltas[cat]

        # probe positions: first speculative + mid
        probe_js = sorted(set([0, K // 2]))
        for j in probe_js:
            t = S + j
            h_gold = h40_gold[j].flatten()       # [16384]
            e_own = err[j]                        # [16384]

            # cross-prompt directions
            other = [oc for oc in other_cats if oc != cat]
            e_x1 = all_errors[other[j % len(other)]][j]
            e_x2 = all_errors[other[(j + 3) % len(other)]][
                min(j + 5, K - 1)]
            # true deltas
            d_own = delta[min(j, delta.shape[0] - 1)]
            d_x = all_deltas[other[j % len(other)]][
                min(j, delta.shape[0] - 1)]
            # random
            r1 = torch.randn(D_STATE)
            r2 = torch.randn(D_STATE)
            r3 = torch.randn(D_STATE)

            dirs = [
                ("own-err", e_own), ("x-err-1", e_x1),
                ("x-err-2", e_x2), ("delta-own", d_own),
                ("delta-x", d_x), ("random-1", r1),
                ("random-2", r2), ("random-3", r3),
            ]

            for dtype, b in dirs:
                b_norm = b.norm().item()
                if b_norm < 1e-8:
                    continue
                b_hat = b / b_norm
                for alpha in EPS_ALPHAS:
                    eps = alpha * b_norm
                    h_plus = (h_gold + eps * b_hat
                              ).view(4, 4096)
                    h_minus = (h_gold - eps * b_hat
                               ).view(4, 4096)
                    m_plus, ok_plus = probe_margin(
                        c, h_plus, t, S)
                    m_minus, ok_minus = probe_margin(
                        c, h_minus, t, S)
                    fd = (m_plus - m_minus) / (2 * eps)
                    rec = {
                        "cat": cat, "j": j, "t": t,
                        "dtype": dtype, "alpha": alpha,
                        "eps": round(eps, 4),
                        "m_plus": round(m_plus, 4),
                        "m_minus": round(m_minus, 4),
                        "fd": round(fd, 6),
                        "fd_abs": round(abs(fd), 6),
                        "greedy_plus": ok_plus,
                        "greedy_minus": ok_minus,
                    }
                    results["probes"].append(rec)
                    type_sens.setdefault(dtype, []).append(
                        abs(fd))
            print(f"  {cat} j={j}: done "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- summary: sensitivity by direction type ----
    for dtype, vals in sorted(type_sens.items()):
        results["summary"][dtype] = {
            "n": len(vals),
            "mean_abs_fd": round(sum(vals) / len(vals), 6),
            "max_abs_fd": round(max(vals), 6),
            "p90_abs_fd": round(sorted(vals)[
                int(0.9 * len(vals))], 6),
        }

    # ---- signal detection ----
    rand_sens = results["summary"].get("random-1", {}).get(
        "mean_abs_fd", 0)
    rand_sens = max(rand_sens, 1e-8)
    signal = {}
    for dtype, s in results["summary"].items():
        signal[dtype] = round(
            s["mean_abs_fd"] / rand_sens, 3)
    results["sensitivity_ratio_vs_random"] = signal

    # any direction type >= 3x random sensitivity?
    has_signal = any(v >= 3.0 for k, v in signal.items()
                     if "random" not in k)
    results["verdict"] = (
        "SIGNAL — specific direction types are >= 3x more "
        "decision-sensitive than random; build FD basis and run "
        "projection oracle" if has_signal else
        "NO SIGNAL — all direction types have similar "
        "sensitivity to random; the tail is isotropically "
        "sensitive and no basis construction helps")

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\n=== FD SCAN VERDICT: {results['verdict']} ===")
    print(f"sensitivity ratios vs random: {signal}")
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
