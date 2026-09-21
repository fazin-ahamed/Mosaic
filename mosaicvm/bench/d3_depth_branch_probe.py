"""Depth-2/3 global-beam probe (roadmap Probe-1).

Gold-free: the beam is constructed from proposer scores only.
Gold is consulted AFTER the beam is frozen, to score the longest
gold prefix present in the beam (then greedy continuation from
that prefix, matching production verification).

Policy:
  depth 1 (root)  B=1 greedy HIST1001
  depth 2         global beam B, local TopK=B
  depth 3         global beam B, local TopK=B
  depth 4+        greedy continuation from the frozen best prefix
                  (production: verifier would pick the matching
                  gold-containing prefix if present; scoring uses
                  that AFTER freeze, never to choose branches)

Prefix score = sum log p_draft. No calibration.

B in {1, 2, 4, 8}. B=1 must reproduce HIST1001+ps greedy E[A]~2.2.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_dd1_hist import DDHistNet
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import (load_head, load_embed,
                                      make_bridge_feat)
from mosaicvm.bench.d3_v1 import make_feat, HORIZON
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_ws import load_a2, load_bridge
from mosaicvm.bench.d1b_tail_robustness import prefix_metrics
from mosaicvm.bench.d3_stats import stats as canon_stats

HIST = "artifacts/d3_scale1000_train/hist1001.pt"
A2 = "artifacts/d3_v1_A2/student.pt"
BRIDGE = "artifacts/d3_bridge_r3b/bridge_ps/bridge.pt"
BEAMS = (1, 2, 4, 8)
BRANCH_DEPTHS = (2, 3)   # 1-indexed speculative positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="artifacts/d3_scale1000_data")
    p.add_argument("--manifest",
                   default="artifacts/d3_scale1000/prompts_scale1000.json")
    p.add_argument("--split", default="val")
    p.add_argument("--bridge", default=BRIDGE)
    p.add_argument("--output",
                   default="artifacts/d3_scale1000_train/depth_branch_probe.json")
    a = p.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()

    ck = torch.load(HIST, map_location="cpu", weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"], ck["aux_dim"])
    hist.load_state_dict(ck["model"]); hist.eval()
    xm_mean, xm_std = ck["xm_mean"], ck["xm_std"]
    xa_mean, xa_std = ck["xa_mean"], ck["xa_std"]
    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(Path(A2))
    bridge, b_xm, b_xs = load_bridge(Path(a.bridge))
    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda(); w_norm_g = w_norm.cuda()

    caps = load_captures(a.data, json.loads(Path(a.manifest).read_text()),
                         None, HORIZON, require_both=False)
    val = [c for c in caps if c["split"] == a.split]
    print(f"val: {len(val)} prompts ({time.time()-t0:.0f}s)", flush=True)

    @torch.inference_mode()
    def hist_logits(mfn, afn):
        z = hist(mfn.unsqueeze(0), afn.unsqueeze(0)).cuda()
        zn = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)
        return (w_head_g @ (zn[0] * w_norm_g))

    @torch.inference_mode()
    def bridge_logits(z):
        zg = z.cuda()
        zn = zg * torch.rsqrt(zg.square().mean() + 1e-6) * w_norm_g
        return (w_head_g @ zn)

    def a2_step(state, prev, driver, cur_tok, t):
        feat = make_feat(embed, state, state - prev, driver, cur_tok, t)
        fn = (feat - a2_xm) / a2_xs
        pred = a2(fn.unsqueeze(0))[0] * t_std + t_mean
        return state + pred

    def cont_logits(new_state, driver, cur_tok, t):
        bf = make_bridge_feat(embed, new_state, driver, cur_tok, t)
        bfn = (bf - b_xm) / b_xs
        with torch.inference_mode():
            z = bridge(bfn.unsqueeze(0))[0]
        return bridge_logits(z)

    def log_softmax_cpu(lg):
        return F.log_softmax(lg.float(), dim=-1).detach().cpu()

    # beam prefix after consuming drafts:
    # (drafts, score, state, prev, cur_tok, driver)


    def greedy_tail(prefix, S, K, toks):
        """Greedy continuation from a frozen prefix to horizon K.
        GOLD NEVER TOUCHED."""
        drafts, score, state, prev, cur_tok, driver = prefix
        drafts = list(drafts)
        for j in range(len(drafts), K):
            t = S + j
            new_state = a2_step(state, prev, driver, cur_tok, t)
            lg = cont_logits(new_state, driver, cur_tok, t)
            tok = int(lg.argmax())
            drafts.append(tok)
            cur_tok, driver = driver, tok
            prev, state = state, new_state
        return drafts

    def gold_prefix_len(drafts, toks, S):
        n = 0
        for j, tok in enumerate(drafts):
            if tok != toks[S + j + 1]:
                break
            n += 1
        return n

    def run_beam(c, B):
        S = c["S"]; K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]; h = c["h40"].float()
        cur = h[S - 1].flatten(); aux = h[S - 2].flatten()
        mf = make_dd_feat(embed, cur, toks[S], toks[S - 1], S)
        mfn = (mf - xm_mean) / xm_std
        afn = (aux - xa_mean) / xa_std
        lg0 = hist_logits(mfn, afn)
        root = int(lg0.argmax())
        ls0 = log_softmax_cpu(lg0)
        root_lp = float(ls0[root])
        # depth-1 prefix after consuming the greedy root (B1 always)
        state0 = h[S - 1].flatten(); prev0 = h[S - 2].flatten()
        new_state0 = a2_step(state0, prev0, toks[S], toks[S - 1], S)
        beam = [((root,), root_lp, new_state0, state0, toks[S], root)]
        expansions = 1  # root greedy

        # depths 2 and 3: global beam
        for depth in range(2, min(4, K + 1)):  # 1-indexed
            local_k = 1 if depth not in BRANCH_DEPTHS else B
            cand = []
            for pref in beam:
                # pref = (drafts, score, state, prev, cur_tok, driver)
                new_state = a2_step(pref[2], pref[3], pref[5],
                                    pref[4], S + depth - 1)
                lg = cont_logits(new_state, pref[5], pref[4],
                                 S + depth - 1)
                ls = log_softmax_cpu(lg)
                k = min(local_k, ls.numel())
                vals, idxs = ls.topk(k)
                for v, tok in zip(vals.tolist(), idxs.tolist()):
                    tok = int(tok)
                    cand.append((pref[0] + (tok,), pref[1] + float(v),
                                 new_state, pref[2], pref[5], tok))
                    expansions += 1
            cand.sort(key=lambda x: x[1], reverse=True)
            beam = cand[:B]

        # freeze beam — gold consulted only from here
        gold_lens = [gold_prefix_len(p[0], toks, S) for p in beam]
        # production verifier would accept the matching gold-containing
        # prefix if present; else the greedy (best-score) prefix.
        # For E[A] we take the longest gold match among frozen prefixes
        # THEN greedy-continue that prefix (the verifier emits the
        # matching prefix then the rest of that path until mismatch).
        best_i = max(range(len(beam)), key=lambda i: (gold_lens[i],
                                                       beam[i][1]))
        chosen = beam[best_i]
        full = greedy_tail(chosen, S, K, toks)
        hits = [full[j] == toks[S + j + 1] for j in range(K)]
        m = prefix_metrics([int(x) for x in hits], K)
        d2_surv = (K >= 2 and gold_prefix_len(chosen[0][:2], toks, S) >= 2
                   if len(chosen[0]) >= 2 else False)
        d3_surv = (K >= 3 and gold_prefix_len(chosen[0][:3], toks, S) >= 3
                   if len(chosen[0]) >= 3 else False)
        return {
            "category": c["category"],
            "m": m,
            "accepted_draft": m["accepted_draft"],
            "E_A": m["E_A"],
            "d2_survived": bool(d2_surv),
            "d3_survived": bool(d3_surv),
            "gold_in_beam_d2": bool(any(
                gold_prefix_len(p[0][:2], toks, S) >= 2
                for p in beam if len(p[0]) >= 2)),
            "gold_in_beam_d3": bool(any(
                gold_prefix_len(p[0][:3], toks, S) >= 3
                for p in beam if len(p[0]) >= 3)),
            "n_prefixes": len(beam),
            "expansions": expansions,
        }

    results = {"classification": "D3_DEPTH_BRANCH_PROBE",
               "policy": {"root": 1, "branch_depths": list(BRANCH_DEPTHS),
                          "beams": list(BEAMS),
                          "score": "sum_logp", "gold": "post-freeze only"},
               "n_prompts": len(val), "by_B": {}}
    print(f"{'B':>3} {'E[A]':>7} {'dEA':>7} {'S1':>6} {'S2':>6} "
          f"{'S3':>6} {'S4':>6} {'d2surv':>7} {'d3surv':>7} "
          f"{'nodes':>6}")
    ea_b1 = None
    for B in BEAMS:
        rows = []
        for c in val:
            rows.append(run_beam(c, B))
        stat_rows = [{"category": r["category"],
                      "cl_loop": {"E_A": r["m"]["E_A"],
                                  "Sk": r["m"]["Sk"]}}
                     for r in rows]
        cs = canon_stats(stat_rows, "cl_loop", k=32)
        if B == 1:
            ea_b1 = cs["mean"]
        dEA = round(cs["mean"] - ea_b1, 4)
        n_node = sum(r["expansions"] for r in rows) / len(rows)
        d2 = sum(r["d2_survived"] for r in rows) / len(rows)
        d3 = sum(r["d3_survived"] for r in rows) / len(rows)
        gpn = ((cs["mean"] - ea_b1) / max(n_node - 1.0, 1e-9)
               if B > 1 else 0.0)
        rec = {
            "canonical": cs,
            "delta_EA": dEA,
            "S1": cs["S1"], "S2": cs["S2"], "S3": cs["S3"],
            "S4": cs["S4"],
            "d2_survived": round(d2, 4),
            "d3_survived": round(d3, 4),
            "mean_expansions": round(n_node, 2),
            "GainPerNode": round(gpn, 4),
            "per_prompt": {r["category"]: {
                "E_A": r["E_A"],
                "d2": r["d2_survived"], "d3": r["d3_survived"],
                "gold_in_beam_d2": r["gold_in_beam_d2"],
                "gold_in_beam_d3": r["gold_in_beam_d3"],
            } for r in rows},
        }
        results["by_B"][str(B)] = rec
        print(f"{B:>3} {cs['mean']:>7.3f} {dEA:>+7.3f} "
              f"{cs['S1']:>6.3f} {cs['S2']:>6.3f} {cs['S3']:>6.3f} "
              f"{cs['S4']:>6.3f} {d2:>7.3f} {d3:>7.3f} "
              f"{n_node:>6.1f}  ({time.time()-t0:.0f}s)", flush=True)

    b8 = results["by_B"]["8"]["delta_EA"]
    if b8 >= 1.0:
        verdict = "STRONG_PASS"
    elif b8 >= 0.50:
        verdict = "PASS"
    elif b8 >= 0.25:
        verdict = "WEAK_DIAGNOSTIC"
    else:
        verdict = "FAIL"
    results["verdict"] = verdict
    results["gate"] = {"PASS": "dEA_B8>=0.50", "STRONG": "dEA_B8>=1.00",
                       "observed_dEA_B8": b8}
    # frozen capability baselines (ps1001 + d3 B4); NEVER compare to own B1
    FROZEN = {"CAP-DEV70": 3.7571, "SCALE-DEV3": 4.0143}
    b4_mean = results["by_B"]["4"]["canonical"]["mean"]
    tag = "CAP-DEV70" if "capdev70" in str(a.data) else (
        "SCALE-DEV3" if "scale1000_data" in str(a.data) else None)
    if tag:
        base = FROZEN[tag]
        results["capability"] = {
            "set": tag, "frozen_ps1001_B4": base,
            "candidate_B4": round(b4_mean, 4),
            "delta_vs_frozen": round(b4_mean - base, 4),
            "gate_4.0_met": b4_mean >= 4.0,
            "readiness_4.15": b4_mean >= 4.15}
        print(f"CAPABILITY[{tag}] B4 {b4_mean:.3f} vs frozen {base:.3f} "
              f"= {b4_mean-base:+.3f}  gate4.0={'MET' if b4_mean>=4.0 else 'no'}",
              flush=True)
    Path(a.output).write_text(json.dumps(results, indent=1))
    print(f"verdict={verdict}  dEA_B8={b8}  wrote {a.output} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
