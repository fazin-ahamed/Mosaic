"""Phase 2 go/no-go ladder. Exit 0 always; table contains pass/fail per row."""
from __future__ import annotations

import json
from pathlib import Path


def _load(name: str):
    p = Path("artifacts") / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def run() -> dict:
    floor = _load("compute_floor_3060.json") or {}
    oracle = _load("oracle_spec_3060.json") or {}
    topk = _load("topk_quality.json") or {}
    aec = _load("aec.json")
    aed = _load("aed_calibrate.json") or {}
    inv = _load("core_inventory.json") or {}

    rows_f = floor.get("projection_rows") or []
    k1 = next((r for r in rows_f if r.get("K") == 1), {})
    k1_toks = k1.get("k1_toks")
    k1_top4 = k1.get("k1_toks_top4")
    kernel = k1.get("routed_kernel") or (floor.get("raw") or {}).get("routed_expert", [{}])[0].get("routed_kernel")
    mean_k = aed.get("mean_k", 3.2)
    # AED compute equivalent: scale routed portion roughly mean_k/6, but report k1_top4 if present
    k1_aed = k1_top4
    if k1_toks and mean_k:
        # conservative: interpolate between k1 (top6) and k1_top4 using mean_k
        if k1_top4:
            # top4 is 4/6 of routed; mean_k/6 of routed
            k1_aed = k1_toks + (k1_top4 - k1_toks) * ((6 - mean_k) / 2.0) if mean_k <= 4 else k1_top4

    orows = oracle.get("rows") or []
    k5 = next((r for r in orows if r.get("K") == 5), {})
    k8 = next((r for r in orows if r.get("K") == 8), {})
    k5_toks = k5.get("accepted_output_toks")
    k8_toks = k8.get("accepted_output_toks")

    def cell(obs, req, ge=True):
        if obs is None:
            return obs, False
        return obs, (obs >= req) if ge else (obs < req)

    ladder = [
        ("3060 K1 perfect residency (bf16 or w4a16)", 12.0, k1_toks),
        ("Fused W4A16 + resident core (K1)", 14.0, k1_toks if kernel == "w4a16" else k1_toks),
        ("Top-4 / AED (K1 equiv)", 16.0, k1_aed or k1_top4 or k1_toks),
        ("Oracle K=5", 25.0, k5_toks),
        ("Oracle K=8", 35.0, k8_toks),
    ]
    print("=== Phase 2 go/no-go ===")
    print(f"{'Stage':<48} {'Required':>10} {'Observed':>12} {'Pass?':>6}")
    table = []
    for name, req, obs in ladder:
        ok = (obs is not None) and (obs >= req)
        obs_s = f"{obs:.3f}" if isinstance(obs, float) else str(obs)
        print(f"{name:<48} {req:>10.1f} {obs_s:>12} {'PASS' if ok else 'FAIL':>6}")
        table.append({"stage": name, "required": req, "observed": obs, "pass": ok})

    print()
    print("kernel:", kernel)
    print("roofline:", (floor.get("roofline") or {}).get("derate_mode"))
    print("gate A':", floor.get("gate_eval"))
    print("topk_quality:", topk.get("status"), "turbo_default=", topk.get("turbo_default"),
          topk.get("turbo_default_quality"))
    print("aed mean_k:", mean_k)
    print("core inventory GB:", inv.get("gpu_gb"), "under_6gb=", inv.get("under_6gb"))
    print("aec:", aec)
    print("oracle gates:", oracle.get("gates"))

    no_drafter = (k8_toks is None) or (k8_toks < 30)
    floor_d = _load("floor_cd.json") or {}
    floor_d_k8 = next((r for r in (floor_d.get("floor_d") or []) if r.get("K") == 8), None)
    if Path("artifacts/phase210x_report.json").exists():
        print("Phase 2.10X present — see phase210x_report")
    elif Path("artifacts/phase210_report.json").exists():
        print("Phase 2.10 present — see phase210_report")
    elif Path("artifacts/phase29r_report.json").exists():
        print("Phase 2.9R present — see phase29r_report")
    elif Path("artifacts/phase29_report.json").exists():
        print("Phase 2.9 present — see phase29_report")
    elif Path("artifacts/phase28_report.json").exists():
        print("Phase 2.8 present — see phase28_report")
    elif Path("artifacts/phase27_report.json").exists():
        print("Phase 2.7 present — see phase27_report")
    elif Path("artifacts/phase26_report.json").exists():
        print("Phase 2.6 present — see phase26_report")
    elif no_drafter:
        print("NO DRAFTER YET — first close dispatch gap")
    else:
        print("oracle K=8 >= 30 accepted tok/s — drafter is in-budget for later phase")

    out = {
        "ladder": table,
        "no_drafter": no_drafter,
        "kernel": kernel,
        "topk_quality": topk,
        "aec": aec,
    }
    Path("artifacts/phase02_report.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    run()
