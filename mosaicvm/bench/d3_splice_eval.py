"""DD1+ps splice evaluator (roadmap S17-S22).

Zero retraining: DD-1 (h40 variant) drafts the FIRST speculative
token from the exact boundary state; if correct, frozen ps bridge
continues from position S+1 with the normal A2 recurrence.

Architecture:
  exact h40[S-1] + root token → DD-1 → token_1    (SPLICE)
  A2(h40[S-1], root) → h40[S]                      (recurrence)
  if token_1 correct:
    A2(h40[S], token_1) → h40[S+1] → ps → token_2  (continuation)
    ... (normal bridge cl_loop from j=1)

The bridge's j=0 draft is REPLACED by DD-1's; the A2 recurrence
still runs at j=0 (producing h40[S] needed for j=1). All feedback
semantics unchanged (cl_loop: own drafts fed back).

Verification requirement: DD-1's standalone S1 (9/21) must
reproduce exactly inside the splice harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

sys = None
import sys as _sys
_sys.path.insert(0, ".")
_sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_bridge import (BridgeNet, load_head,
                                      load_embed,
                                      make_bridge_feat)
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_v1 import (DeltaNet, make_feat, D_STATE,
                                  HORIZON, pos_encode)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d1b_tail_robustness import prefix_metrics, \
    prompt_mean_ci


def load_a2(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = DeltaNet(ck["hidden"], ck["depth"], ck["rank"],
                 aux_out=ck.get("h42_aux_weight", 0) > 0
                 or "aux.weight" in ck["model"])
    m.load_state_dict(ck["model"])
    m.eval()
    return (m, ck["target_mean"], ck["target_std"],
            ck["x_mean"], ck["x_std"])


def load_bridge(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = BridgeNet(ck["hidden"], ck["depth"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["x_mean"], ck["x_std"]


def load_dd1(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = BridgeNet(ck["hidden"], ck["depth"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["x_mean"], ck["x_std"], ck["state_field"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dd1", type=Path,
                   default=Path("artifacts/d3_dd1/dd1_h40/dd1.pt"))
    p.add_argument("--a2", type=Path,
                   default=Path("artifacts/d3_v1_A2/student.pt"))
    p.add_argument("--bridge", type=Path,
                   default=Path("artifacts/d3_bridge_r3b/"
                                "bridge_ps/bridge.pt"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)

    t0 = time.time()
    dd1, dd1_xm, dd1_xs, dd1_field = load_dd1(a.dd1)
    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(a.a2)
    bridge, b_xm, b_xs = load_bridge(a.bridge)
    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda()
    w_norm_g = w_norm.cuda()

    caps = load_captures(a.data,
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    print(f"val: {len(val)} ({time.time()-t0:.0f}s)", flush=True)

    @torch.inference_mode()
    def head_argmax(z):
        zg = z.cuda()
        zn = zg * torch.rsqrt(zg.square().mean() + 1e-6
                              ) * w_norm_g
        return int((w_head_g @ zn).argmax())

    def splice_run(c):
        """Full splice: DD-1 first draft + ps bridge continuation."""
        S = c["S"]
        K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]
        h = c["h40"].float()

        # DD-1 first decision (from exact boundary state)
        dd1_state = c[dd1_field][S - 1].float().flatten()
        dd1_feat = make_dd_feat(embed, dd1_state, toks[S],
                                toks[S - 1], S)
        dd1_fn = (dd1_feat - dd1_xm) / dd1_xs
        dd1_z = dd1(dd1_fn.unsqueeze(0))[0]
        dd1_draft = head_argmax(dd1_z)

        # A2 recurrence + bridge continuation
        state = h[S - 1].flatten()
        prev = h[S - 2].flatten()
        cur_tok = toks[S - 1]
        driver = toks[S]  # exact root

        hits = []
        for j in range(K):
            t = S + j
            # A2 predicts h40[t]
            feat = make_feat(embed, state, state - prev, driver,
                             cur_tok, t)
            fn = (feat - a2_xm) / a2_xs
            pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
            new_state = state + pred

            if j == 0:
                draft = dd1_draft  # SPLICE: DD-1 replaces bridge
            else:
                bf = make_bridge_feat(embed, new_state, driver,
                                      cur_tok, t)
                bfn = (bf - b_xm) / b_xs
                z = bridge(bfn.unsqueeze(0))[0]
                draft = head_argmax(z)

            hits.append(draft == toks[t + 1])
            cur_tok = driver
            driver = draft
            prev = state
            state = new_state
        return hits, dd1_draft

    # ---- wall measurements (S21): cuda.synchronize + warmup ----
    @torch.inference_mode()
    def wall_probe(c, reps=10):
        """Per-position DD1 / ps-continuation wall with proper
        GPU synchronization and warmup."""
        S = c["S"]
        toks = c["tokens"]
        h = c["h40"].float()
        # warmup
        for _ in range(3):
            dd1_state = c[dd1_field][S - 1].float().flatten()
            dd1_feat = make_dd_feat(embed, dd1_state, toks[S],
                                    toks[S - 1], S)
            dd1_fn = (dd1_feat - dd1_xm) / dd1_xs
            dd1_z = dd1(dd1_fn.unsqueeze(0))[0]
            head_argmax(dd1_z)
            feat = make_feat(embed, h[S - 1].flatten(),
                             h[S - 1].flatten()
                             - h[S - 2].flatten(),
                             toks[S], toks[S - 1], S)
            fn = (feat - a2_xm) / a2_xs
            pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
            new_state = h[S - 1].flatten() + pred
            bf = make_bridge_feat(embed, new_state, toks[S],
                                  toks[S - 1], S + 1)
            bfn = (bf - b_xm) / b_xs
            z = bridge(bfn.unsqueeze(0))[0]
            head_argmax(z)
        torch.cuda.synchronize()
        # DD-1 wall
        ts_dd1 = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            dd1_state = c[dd1_field][S - 1].float().flatten()
            dd1_feat = make_dd_feat(embed, dd1_state, toks[S],
                                    toks[S - 1], S)
            dd1_fn = (dd1_feat - dd1_xm) / dd1_xs
            dd1_z = dd1(dd1_fn.unsqueeze(0))[0]
            head_argmax(dd1_z)
            torch.cuda.synchronize()
            ts_dd1.append((time.perf_counter() - t) * 1e3)
        # ps continuation wall (A2 + bridge + head, one position)
        ts_ps = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            feat = make_feat(embed, h[S - 1].flatten(),
                             h[S - 1].flatten()
                             - h[S - 2].flatten(),
                             toks[S], toks[S - 1], S)
            fn = (feat - a2_xm) / a2_xs
            pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
            new_state = h[S - 1].flatten() + pred
            bf = make_bridge_feat(embed, new_state, toks[S],
                                  toks[S - 1], S + 1)
            bfn = (bf - b_xm) / b_xs
            z = bridge(bfn.unsqueeze(0))[0]
            head_argmax(z)
            torch.cuda.synchronize()
            ts_ps.append((time.perf_counter() - t) * 1e3)
        ts_dd1.sort(); ts_ps.sort()
        return ts_dd1[reps // 2], ts_ps[reps // 2]

    walls = [wall_probe(c) for c in val[:5]]
    w_dd1 = sorted(w[0] for w in walls)[len(walls) // 2]
    w_ps = sorted(w[1] for w in walls)[len(walls) // 2]
    w_total = w_dd1 + w_ps
    print(f"\nwall: DD1 {w_dd1:.2f} ms | ps-cont {w_ps:.2f} ms "
          f"| first-pos total {w_total:.2f} ms", flush=True)


    rows = []
    dd1_standalone_hits = 0
    for c in val:
        hits, dd1_draft = splice_run(c)
        m = prefix_metrics([int(x) for x in hits],
                           len(hits))
        # DD-1 standalone check (first hit)
        dd1_standalone_hits += hits[0]
        rows.append({"cat": c["category"], "m": m,
                     "hits": [int(x) for x in hits]})
        print(f"  {c['category']}: E[A]={m['E_A']} "
              f"hits={sum(hits)}/{len(hits)} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # DD-1 standalone S1 verification
    dd1_s1 = dd1_standalone_hits / len(val)
    print(f"\nDD-1 standalone S1 inside splice: "
          f"{dd1_standalone_hits}/{len(val)} = {dd1_s1:.4f} "
          f"(expected 9/21 = 0.4286)", flush=True)
    if abs(dd1_s1 - 9/21) > 0.01:
        print("WARNING: DD-1 S1 differs from standalone! "
              "Debug before interpreting splice results.",
              flush=True)

    # canonical statistics via d3_stats (no hand-rolled metrics)
    from mosaicvm.bench.d3_stats import stats as canon_stats
    # emit rows in the format d3_stats expects
    stat_rows = [{"category": r["cat"],
                  "cl_loop": {"E_A": r["m"]["E_A"],
                              "Sk": r["m"]["Sk"]}}
                 for r in rows]
    cs = canon_stats(stat_rows, "cl_loop", k=32)
    # conditional survival from individual Sk
    s1c = dd1_standalone_hits       # already counted exactly
    c2 = cs["S2"] / cs["S1"] if cs["S1"] > 0 else 0
    c3 = cs["S3"] / cs["S2"] if cs["S2"] > 0 else 0
    c4 = cs["S4"] / cs["S3"] if cs["S3"] > 0 else 0

    # mechanism gate: exact COUNT comparison (9/21, not 0.429)
    GATE_S1_COUNT = 9
    gate_s1 = s1c >= GATE_S1_COUNT
    gate_s4 = cs["S4"] > 0

    result = {
        "classification": "D3_DD1_PS_SPLICE_EVAL",
        "production_certified": False,
        "dd1_standalone_S1_count": f"{dd1_standalone_hits}/{len(val)}",
        "canonical_stats": cs,
        "conditional_survival": {
            "C2": round(c2, 4), "C3": round(c3, 4),
            "C4": round(c4, 4)},
        "mechanism_gate": {
            "S1_count": f"{s1c}/{len(val)}",
            "S1_gate": f">= {GATE_S1_COUNT}",
            "S1_passed": gate_s1,
            "S4": cs["S4"],
            "S4_passed": gate_s4,
            "overall": "PASS" if gate_s1 and gate_s4 else "FAIL"},
        "rows": [{"category": r["cat"], "E_A": r["m"]["E_A"],
                  "Sk": r["m"]["Sk"]} for r in rows],
        "dd1": str(a.dd1.resolve()),
        "bridge": str(a.bridge.resolve()),
        "a2": str(a.a2.resolve()),
        "wall_ms": {"dd1_first_pos": round(w_dd1, 3),
                    "ps_cont_per_pos": round(w_ps, 3),
                    "first_pos_total": round(w_total, 3)},
    }
    a.output.write_text(json.dumps(result, indent=1))
    print(f"\n=== SPLICE RESULT (canonical stats) ===")
    print(f"  mean E[A] = {cs['mean']} (ps was 2.667)")
    print(f"  S1 = {s1c}/{len(val)} (ps was 8/21, DD-1 was 9/21)")
    print(f"  S4 = {cs['S4']} (ps was 0.143)")
    print(f"  deepest = {cs['deepest']} (ps was 15)")
    print(f"  C2 = {c2:.3f} C3 = {c3:.3f} C4 = {c4:.3f}")
    print(f"  trim20 = {cs['trim20']} (ps was 1.539)")
    print(f"  loo_min = {cs['loo_min']} (ps was 2.000)")
    print(f"\n  mechanism gate (S1 >= 9/21 AND S4 > 0): "
          f"{'PASS' if gate_s1 and gate_s4 else 'FAIL'}")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
