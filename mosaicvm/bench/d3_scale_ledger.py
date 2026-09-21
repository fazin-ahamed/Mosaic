"""Paired scale ledger + gates (S28–S33).

Compares a frozen smaller-corpus baseline vs a scaled model
on the same holdout:
  G = #(base wrong, scaled correct)
  L = #(base correct, scaled wrong)
  ΔH = G - L
Gates: ΔH >= 4 AND ΔE[A] > 0 AND S4 > 0 (S28);
robust: trim20/LOO deltas, >=2 domain gains (S29);
strong: ΔH >= 6 (S30); plateau |ΔH| <= 2 (S32); fail ΔH <= -3 (S33).

Verdict strings are measured labels for the pairing just
written. After SCALE700, a strong pass means SCALE1000
already ran; the next authorized step is a scaling-law fit,
not another automatic 700.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(base_path, scaled_path, out_path):
    b = json.loads(Path(base_path).read_text())
    s = json.loads(Path(scaled_path).read_text())
    bp, sp = b["per_prompt"], s["per_prompt"]
    assert set(bp) == set(sp)
    G, L = [], []
    for cat in sorted(bp):
        bh, sh = bp[cat]["hit"], sp[cat]["hit"]
        if not bh and sh:
            G.append(cat)
        if bh and not sh:
            L.append(cat)
    dH = len(G) - len(L)
    n = len(bp)
    csb, css = b["canonical_stats"], s["canonical_stats"]
    dEA = css["mean"] - csb["mean"]
    dtrim = css["trim20"] - csb["trim20"]
    dloo = css["loo_min"] - csb["loo_min"]
    # domain-level gains
    dom_g, dom_l = {}, {}
    for cat in G:
        dom_g[cat.rsplit("_", 1)[0]] = \
            dom_g.get(cat.rsplit("_", 1)[0], 0) + 1
    for cat in L:
        dom_l[cat.rsplit("_", 1)[0]] = \
            dom_l.get(cat.rsplit("_", 1)[0], 0) + 1
    doms_net_gain = sum(
        1 for d in set(dom_g) | set(dom_l)
        if dom_g.get(d, 0) - dom_l.get(d, 0) > 0)

    robust = (dtrim > 0 and dloo > 0
              and doms_net_gain >= 2)
    result = {
        "classification": "D3_SCALE_LEDGER",
        "G_conversions": G, "L_losses": L,
        "G_count": len(G), "L_count": len(L),
        "delta_H": dH, "n": n,
        "S1_base": sum(v["hit"] for v in bp.values()),
        "S1_scaled": sum(v["hit"] for v in sp.values()),
        "delta_EA": round(dEA, 4),
        "delta_trim20": round(dtrim, 4),
        "delta_loo": round(dloo, 4),
        "S4_scaled": css["S4"],
        "domain_gains": dom_g, "domain_losses": dom_l,
        "domains_net_gain": doms_net_gain,
        "boot_ci95_base": csb["boot_ci95"],
        "boot_ci95_scaled": css["boot_ci95"],
        "gates": {
            "mechanism_S28": (dH >= 4 and dEA > 0
                              and css["S4"] > 0),
            "robust_S29": robust,
            "strong_S30": dH >= 6 and dEA > 0
                          and dtrim > 0 and dloo > 0,
            "plateau_S32": abs(dH) <= 2,
            "fail_S33": dH <= -3},
        "verdict": None,
    }
    if result["gates"]["strong_S30"]:
        result["verdict"] = "STRONG_PASS_SCALING_LAW_NEXT"
    elif result["gates"]["mechanism_S28"] \
            and result["gates"]["robust_S29"]:
        result["verdict"] = "PASS_SCALING_LAW_NEXT"
    elif result["gates"]["fail_S33"]:
        result["verdict"] = "FAIL_REVISIT_ARCHITECTURE"
    elif result["gates"]["plateau_S32"]:
        result["verdict"] = "PLATEAU_NO_AUTOMATIC_SCALE"
    else:
        result["verdict"] = "MIXED_INSPECT_DETAIL"
    Path(out_path).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("G_conversions",
                                   "L_losses")}, indent=1))
    print("G:", G)
    print("L:", L)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
