"""HIST-1 causal attribution (roadmap S8-S21). No training.

Frozen HIST-1 model, six auxiliary conditions per DEV root:

  A0  zero aux                    [HIST-0M equivalent]
  A1  true previous state          [HIST-1]
  A2  older same-prompt state      [multi-step temporal]
  A3  same-domain other prompt     [domain/template cue]
  A4  cross-domain other prompt    [generic regularization]
  A5  current state as aux         [self-duplication]

Plus the full 21×21 aux-swap matrix (target-vs-top-competitor
margin), and aux-space interpolation (alpha in {0,.25,.5,.75,1}).

All conditions use ONLY committed states (no future leakage).
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

from mosaicvm.bench.d3_dd1_hist import DDHistNet, D_STATE
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import load_head, load_embed
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON

OUT = Path("artifacts/d3_hist_attribution.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    a = p.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()

    # ---- load frozen HIST-1 ----
    ckpt = Path("artifacts/d3_dd1_hist/dd1_hist_prev/dd1_hist.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = DDHistNet(ck["hidden"], ck["depth"], ck["aux_dim"])
    model.load_state_dict(ck["model"])
    model.eval()
    xm_mean, xm_std = ck["xm_mean"], ck["xm_std"]
    xa_mean, xa_std = ck["xa_mean"], ck["xa_std"]
    sfield = ck["state_field"]

    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda()
    w_norm_g = w_norm.cuda()

    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    print(f"{len(val)} DEV prompts ({time.time()-t0:.0f}s)",
          flush=True)

    @torch.inference_mode()
    def eval_aux(c, aux_state):
        """Evaluate DD-1 with a given aux state; return
        (pred, target_logit, top_comp_logit, margin, top16_hit,
        target_rank)."""
        S = c["S"]
        toks = c["tokens"]
        cur = c[sfield][S - 1].float().flatten()
        mf = make_dd_feat(embed, cur, toks[S], toks[S - 1], S)
        mfn = (mf - xm_mean) / xm_std
        afn = (aux_state - xa_mean) / xa_std
        z = model(mfn.unsqueeze(0), afn.unsqueeze(0))[0]
        zg = z.cuda()
        zn = zg * torch.rsqrt(zg.square().mean() + 1e-6
                              ) * w_norm_g
        lg = w_head_g @ zn
        true = toks[S + 1]
        tl = float(lg[true])
        pred = int(lg.argmax())
        sorted_lg = lg.sort(descending=True)
        if pred == true:
            comp = int(sorted_lg.indices[1])
        else:
            comp = pred
        margin = tl - float(lg[comp])
        rank = int((lg > tl).sum().item()) + 1
        top16 = int(true in lg.topk(16).indices.tolist())
        return {"pred": pred, "true": true, "hit": pred == true,
                "margin": round(margin, 4), "rank": rank,
                "top16": top16,
                "target_logit": round(tl, 4)}

    # ---- collect all prev states for swapping ----
    prev_states = {}
    older_states = {}
    for c in val:
        S = c["S"]
        prev_states[c["category"]] = \
            c[sfield][S - 2].float().flatten()
        # older: S-3 if exists (log fallback usage)
        if S - 3 >= 0:
            older_states[c["category"]] = \
                c[sfield][S - 3].float().flatten()
        else:
            older_states[c["category"]] = \
                c[sfield][S - 2].float().flatten()
            print(f"  A2 fallback: {cat} S-3 < 0, "
                  f"using S-2 (same as A1)", flush=True)

    # ---- condition A evaluation (all prompts × 6 conditions) ----
    conditions = {}
    for c in val:
        cat = c["category"]
        S = c["S"]
        cur = c[sfield][S - 1].float().flatten()
        domain = cat.rsplit("_", 1)[0]
        same_dom = [k for k in prev_states
                    if k.rsplit("_", 1)[0] == domain
                    and k != cat]
        cross_dom = [k for k in prev_states
                     if k.rsplit("_", 1)[0] != domain]
        # pick first same-domain and first cross-domain
        a3_key = same_dom[0] if same_dom else None
        a4_key = cross_dom[0] if cross_dom else None

        conds = {
            "A0_zero": eval_aux(c, torch.zeros(D_STATE)),
            "A1_true_prev": eval_aux(c, prev_states[cat]),
            "A2_older_same": eval_aux(c, older_states[cat]),
            "A3_same_domain": eval_aux(c, prev_states[a3_key])
            if a3_key else None,
            "A4_cross_domain": eval_aux(c, prev_states[a4_key])
            if a4_key else None,
            "A5_dup_current": eval_aux(c, cur),
        }
        conditions[cat] = conds
        print(f"  {cat}: A0={'H' if conds['A0_zero']['hit'] else 'M'}"
              f" A1={'H' if conds['A1_true_prev']['hit'] else 'M'}"
              f" A2={'H' if conds['A2_older_same']['hit'] else 'M'}"
              f" A3={'H' if conds['A3_same_domain'] and conds['A3_same_domain']['hit'] else 'M'}"
              f" A4={'H' if conds['A4_cross_domain'] and conds['A4_cross_domain']['hit'] else 'M'}"
              f" A5={'H' if conds['A5_dup_current']['hit'] else 'M'}"
              f" ({time.time()-t0:.0f}s)", flush=True)

    # ---- 21×21 aux-swap matrix ----
    print("\n21x21 matrix...", flush=True)
    matrix = {}
    for c_target in val:
        cat_t = c_target["category"]
        row = {}
        for c_source in val:
            cat_s = c_source["category"]
            r = eval_aux(c_target, prev_states[cat_s])
            row[cat_s] = {"margin": r["margin"],
                          "hit": r["hit"], "rank": r["rank"]}
        matrix[cat_t] = row
        print(f"  {cat_t} done ({time.time()-t0:.0f}s)",
              flush=True)

    # diagonal advantage: true aux vs median of swapped aux
    diag_adv = {}
    for cat_t, row in matrix.items():
        diag_m = row[cat_t]["margin"]
        off = [v["margin"] for k, v in row.items() if k != cat_t]
        off_sorted = sorted(off)
        med_off = off_sorted[len(off) // 2]
        diag_adv[cat_t] = round(diag_m - med_off, 4)
    mean_diag = sum(diag_adv.values()) / len(diag_adv)
    pos_diag = sum(1 for v in diag_adv.values() if v > 0)

    # ---- margin shift analysis (A1 vs A0) ----
    margin_shifts = {}
    flips_pos = flips_neg = 0
    for cat, conds in conditions.items():
        dm = conds["A1_true_prev"]["margin"] \
            - conds["A0_zero"]["margin"]
        margin_shifts[cat] = round(dm, 4)
        if conds["A1_true_prev"]["hit"] \
                and not conds["A0_zero"]["hit"]:
            flips_pos += 1
        elif conds["A0_zero"]["hit"] \
                and not conds["A1_true_prev"]["hit"]:
            flips_neg += 1
    pos_shifts = sum(1 for v in margin_shifts.values()
                     if v > 0)
    neg_shifts = sum(1 for v in margin_shifts.values()
                     if v < 0)

    # ---- json_14 specific analysis ----
    j14 = conditions.get("json_14", {})
    j14_detail = {
        "conditions": j14,
        "same_domain_rescue": {
            k: matrix["json_14"][k]["hit"]
            for k in ("json_13", "json_15")
            if k in matrix.get("json_14", {})
        },
        "diagonal_margin": matrix.get("json_14", {}).get(
            "json_14", {}).get("margin"),
        "off_diagonal_median_margin": sorted(
            [v["margin"] for k, v in
             matrix.get("json_14", {}).items()
             if k != "json_14"])[10]
        if "json_14" in matrix else None,
    }

    # ---- interpolation in AUX-STEM LEARNED SPACE (roadmap S18)
    # u(α) = u0 + α(u1 - u0), where u = aux_stem(standardized)
    interp = {}
    c_j14 = next(c for c in val
                 if c["category"] == "json_14")
    if c_j14:
        S = c_j14["S"]
        cur = c_j14[sfield][S - 1].float().flatten()
        prev = prev_states["json_14"]
        mf = make_dd_feat(embed, cur, c_j14["tokens"][S],
                          c_j14["tokens"][S - 1], S)
        mfn = (mf - xm_mean) / xm_std
        af_zero = (torch.zeros(D_STATE) - xa_mean) / xa_std
        af_true = (prev - xa_mean) / xa_std
        with torch.inference_mode():
            # aux-stem POST-ACTIVATION outputs (learned space;
            # matches model.forward's act(aux_stem(x)) path)
            u0 = model.act(model.aux_stem(af_zero.unsqueeze(0)))
            u1 = model.act(model.aux_stem(af_true.unsqueeze(0)))
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            u_alpha = u0 + alpha * (u1 - u0)
            with torch.inference_mode():
                h_main = model.act(model.main_stem(
                    mfn.unsqueeze(0)))
                # u_alpha is already post-activation; fusion
                # applies its own act to the combined vector
                h_fused = model.act(model.fusion(
                    torch.cat([h_main, u_alpha], dim=-1)))
                h = h_fused
                for m in model.mid:
                    h = h + model.act(m(h))
                z = model.head(h)
            zg = z[0].cuda()
            zn = zg * torch.rsqrt(
                zg.square().mean() + 1e-6) * w_norm_g
            lg = w_head_g @ zn
            true = c_j14["tokens"][S + 1]
            tl = float(lg[true])
            pred = int(lg.argmax())
            # margin: target vs best COMPETITOR (not vs max,
            # which is 0 on hits)
            if pred == true:
                sorted_lg = lg.sort(descending=True)
                comp = int(sorted_lg.indices[1])
            else:
                comp = pred
            margin = tl - float(lg[comp])
            interp[str(alpha)] = {
                "pred": pred, "hit": pred == true,
                "margin": round(margin, 4)}
            print(f"  json_14 alpha={alpha} (stem-space): "
                  f"{'HIT' if pred == true else 'MISS'} "
                  f"margin={margin:.4f}", flush=True)

    # ---- compile verdict ----
    # Pattern classification per roadmap S22-S25
    if flips_pos >= 3 and pos_shifts >= 12 and pos_diag >= 12:
        pattern = "BROAD_HIDDEN_SIGNAL"
    elif flips_pos >= 2 and pos_shifts >= 10:
        pattern = "MODERATE_TEMPORAL_SIGNAL"
    elif pos_diag < 8:
        pattern = "ISOLATED_FLIP"
    else:
        pattern = "MIXED"

    result = {
        "classification": "D3_HIST_ATTRIBUTION",
        "conditions": conditions,
        "swap_matrix": matrix,
        "diagonal_advantage": {
            "per_prompt": diag_adv,
            "mean": round(mean_diag, 4),
            "positive_count": f"{pos_diag}/21",
        },
        "margin_shift_A1_vs_A0": {
            "per_prompt": margin_shifts,
            "flips_positive": flips_pos,
            "flips_negative": flips_neg,
            "positive_shifts": pos_shifts,
            "negative_shifts": neg_shifts,
        },
        "json_14_detail": j14_detail,
        "interpolation": interp,
        "pattern": pattern,
        "verdict": {
            "BROAD_HIDDEN_SIGNAL": "history generalizes below "
            "Top1; hard-negative ranking on HIST-1 is NEXT",
            "MODERATE_TEMPORAL_SIGNAL": "some temporal value; "
            "consider ranking with history input retained",
            "ISOLATED_FLIP": "json_14 is an isolated event; do "
            "not deepen HIST; hard-negative ranking is NEXT",
            "MIXED": "mixed evidence; inspect per-prompt detail",
        }[pattern],
        "production_timing": None,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(f"\n=== ATTRIBUTION PATTERN: {pattern} ===")
    print(f"verdict: {result['verdict']}")
    print(f"flips: +{flips_pos}/-{flips_neg} | "
          f"margins: +{pos_shifts}/-{neg_shifts} | "
          f"diag advantage: {pos_diag}/21 positive, "
          f"mean {mean_diag:.4f}")
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
