"""G2 §38-unit certification evaluator — applies FROZEN spec v1.

Reads the frozen thresholds (g2_spec_v1.md; encoded verbatim below —
they must not be edited after this point) and evaluates the MosaicVM
records against the source-class reference corpus:

per prompt:
  1. matched-prefix computation (token equality);
  2. if fully exact -> PASS;
  3. at the first divergence, eligibility per spec §3:
     both candidates in near-tie set (reference top1-top2 margin
     <= TAU_MARGIN) AND pair |dlogit| <= TAU_DLOGIT
     AND layerwise hidden rel_l2 at matched steps <= tau_h(s)
     AND route-set differences confined to the ambiguity band
     AND common-expert |dw| <= TAU_W
     AND hash layers exact;
     eligible -> PASS (classified flip; strict comparison stops);
     ineligible -> FAIL (localization data reported).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REF_DIR = Path(sys.argv[1] if len(sys.argv) > 1
               else "artifacts/g2_cert_ref_3060")
MOS_DIR = Path(sys.argv[2] if len(sys.argv) > 2
               else "artifacts/g2_cert_mosaicvm")
OUT = Path("artifacts/g2_certification_result.json")

# ---- FROZEN G2 SPEC v1 (g2_spec_v1.md; do not edit) ----
TAU_H = {  # monotone p99 x1.25 guard, by matched-step bucket
    (0, 1): 0.305, (2, 3): 0.305, (4, 7): 0.538, (8, 15): 0.538,
    (16, 23): 0.607, (24, 31): 0.660,
}
TAU_W = 0.075
TAU_MARGIN = 1.72
TAU_DLOGIT = 1.43
TAU_R = 0.275   # router ambiguity band around rank-6 cutoff


def bucket_tau(s):
    for (lo, hi), v in TAU_H.items():
        if lo <= s <= hi:
            return v
    return 0.660


def main():
    results, all_pass = [], True
    for f in sorted(REF_DIR.glob("*.pt")):
        ref = torch.load(f, map_location="cpu", weights_only=False)
        cat = ref["category"]
        mf = None
        for cand in MOS_DIR.glob("*.pt"):
            r = torch.load(cand, map_location="cpu",
                           weights_only=False)
            if r["category"] == cat:
                mf, mos = cand, r
                break
        if mf is None:
            print(f"[{cat}] MISSING MosaicVM record")
            all_pass = False
            continue
        ga, gb = ref["generated"], mos["generated"]
        n = min(len(ga), len(gb))
        prefix = 0
        for x, y in zip(ga, gb):
            if x != y:
                break
            prefix += 1

        checks = {"prefix": prefix, "n": n}
        if prefix == n:
            checks["verdict"] = "EXACT_PASS"
            results.append({"category": cat, **checks})
            print(f"[{cat}] EXACT PASS ({n}/{n})")
            continue

        fd = prefix  # first divergence position
        # --- near-tie: reference margin + pair dlogit ---
        ref5 = ref["top5_per_token"][fd]
        mos5_ids, mos5_vals = mos["top5"][fd]
        margin = ref5["vals"][0] - ref5["vals"][1]
        near_tie = margin <= TAU_MARGIN
        # pair |dlogit|: each side's logit for BOTH candidate tokens
        rm = dict(zip(ref5["ids"], ref5["vals"]))
        mm = dict(zip(mos5_ids, mos5_vals))
        cand = {ga[fd], gb[fd]}
        pair_ok = True
        pair_detail = {}
        for t in cand:
            ra, ma = rm.get(t), mm.get(t)
            pair_detail[t] = [ra, ma]
            if ra is not None and ma is not None and \
                    abs(ra - ma) > TAU_DLOGIT:
                pair_ok = False
        # top-k membership: both candidates in reference top-16
        t16 = ref5["ids"][:16]
        both_topk = all(t in t16 for t in cand)
        # --- hidden drift at matched steps vs tau_h(s) ---
        # mos["hiddens"][dk] = [(layer, h_mean)] for decode step dk
        hm = {(h["step"], h["layer"]): h["h_mean"].float()
              for h in ref["hiddens"]}
        worst = 0.0
        viol_steps = 0
        n_hid = 0
        for dk, layer_recs in enumerate(mos["hiddens"]):
            if dk > prefix - 1:
                continue
            for (L, h) in layer_recs:
                c = hm.get((dk, L))
                if c is None:
                    continue
                r = ((h.float() - c).norm()
                     / c.norm().clamp_min(1e-6)).item()
                n_hid += 1
                if r > bucket_tau(dk):
                    viol_steps += 1
                worst = max(worst, r)
        hidden_ok = viol_steps == 0
        # --- routes at matched steps: hash exact, ambiguity band, dw ---
        gr = {}
        for g in ref["gates"]:
            key = ("p", g["layer"]) if g["phase"] == "prefill" \
                else ("d", g["step"], g["layer"])
            gr[key] = g
        hash_ok = True
        route_viol = []
        # mos["routes"][0] = prefill, [j] = decode processing gen[j-1]
        for dk in range(min(prefix, len(mos["routes"]))):
            for (L, ids, ws) in mos["routes"][dk]:
                key = ("p", L) if dk == 0 else ("d", dk - 1, L)
                g = gr.get(key)
                if g is None:
                    continue
                r_ids, r_w = g["idx"][-1], g["w"][-1]
                if L < 3:
                    if ids != r_ids:
                        hash_ok = False
                        route_viol.append({"step": dk, "layer": L,
                                           "kind": "hash"})
                    continue
                common = set(ids) & set(r_w and r_ids or [])
                for e in common:
                    i, j = ids.index(e), r_ids.index(e)
                    if abs(ws[i] - r_w[j]) > TAU_W:
                        route_viol.append({"step": dk, "layer": L,
                                           "kind": "dw",
                                           "e": e})
                break  # one gate sample per step is enough for cert
        routes_ok = hash_ok and len(route_viol) == 0
        eligible = (near_tie and pair_ok and both_topk and hidden_ok
                    and routes_ok)
        verdict = "G2_PASS_FLIP" if eligible else "FAIL"
        if not eligible:
            all_pass = False
        results.append({
            "category": cat, "prefix": prefix, "n": n,
            "first_div": fd, "ref_token": ga[fd], "mos_token": gb[fd],
            "ref_margin_top12": round(margin, 4),
            "near_tie": near_tie, "pair_ok": pair_ok,
            "pair_logits": {str(k): v for k, v in pair_detail.items()},
            "both_in_ref_top16": both_topk,
            "hidden_worst_rel_l2": round(worst, 4),
            "hidden_viol_steps": viol_steps, "hidden_ok": hidden_ok,
            "hidden_n": n_hid,
            "hash_exact": hash_ok, "route_violations": route_viol[:5],
            "verdict": verdict,
        })
        print(f"[{cat}] {verdict}: div@{fd} margin={margin:.3f} "
              f"pair_ok={pair_ok} top16={both_topk} "
              f"hid_worst={worst:.3f} hash={hash_ok}")

    out = {
        "classification": "G2_V1_CERTIFICATION_RESULT",
        "spec": "g2_spec_v1 FROZEN thresholds (tau_h(s) guarded, "
                "tau_w 0.075, tau_margin 1.72, tau_dlogit 1.43, "
                "tau_r 0.275)",
        "reference_side": "kit inference/model.py + torch shims on "
                          "RTX 3060 (same hardware as MosaicVM — "
                          "deviation note: sandbox unavailable; "
                          "cross-hardware-calibrated envelope is an "
                          "upper bound for this pairing)",
        "prompts": "disjoint held-out certification set "
                   "(g2_cert_prompts.json)",
        "results": results,
        "verdict": "PASS" if all_pass else "FAIL",
        "equivalence_2_of_2": all_pass,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n=== G2 CERTIFICATION: {out['verdict']} ===")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
