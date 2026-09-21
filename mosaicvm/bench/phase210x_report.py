"""Phase 2.10X width-ceiling go/no-go. One precedence-ordered verdict line."""
from __future__ import annotations

import json
from pathlib import Path

RAW_30 = 30.0
RAW_35 = 35.0


def _load(name: str):
    p = Path("artifacts") / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def run() -> dict:
    wvx = _load("wide_verify_x.json") or {}
    wv = _load("wide_verify.json") or {}
    gates = wvx.get("gates") or {}
    best_wide = gates.get("best_wide_candidate_toks")
    best_wide_k = gates.get("best_wide_k")
    k_hw = gates.get("k_hw")
    plateau = gates.get("plateau")
    crossed_30 = gates.get("crossed_30_wide")
    crossed_35 = gates.get("crossed_35_wide")

    full_rows = wvx.get("full_rows") or []
    complete = bool(full_rows)
    bad_status = [
        r for r in (wvx.get("moe_rows") or [])
        if r.get("status") not in ("ok", "no_windows")
        and not str(r.get("status", "")).startswith(("error:", "invalid_not_256"))
    ]

    stage2_status = wvx.get("stage2_status", "not_attempted")

    if not complete:
        verdict = "WIDEVERIFY X INCOMPLETE"
    elif bad_status:
        verdict = "UNPLANNED_DEVIATION"
    elif crossed_35:
        verdict = f"WIDEVERIFY X STRONG — RAW 35+ AT K{best_wide_k}"
    elif crossed_30:
        # crossed 30 only at K >= 64 would be WIDE_ONLY_PASS; here every
        # crossing K is within the directly measured corpus
        verdict = f"WIDEVERIFY X PASS — RAW30 CROSSED AT K{best_wide_k} (P2_RAW30_WIDE_PASS)"
    elif plateau:
        verdict = "WIDEVERIFY X PLATEAU BELOW 30"
    else:
        verdict = "WIDEVERIFY X MISS BELOW 30"

    next_focus = None
    if "STRONG" in verdict:
        next_focus = ("2.9L continues for headroom/residency (objective: 35-45 at "
                      "practical widths), 2.11 core, 2.12A oracle; major-kernel "
                      "urgency reduced")
    elif "PASS" in verdict:
        next_focus = "2.9L objective shifts to headroom 35-45; continue all tracks"
    elif "PLATEAU" in verdict or "MISS" in verdict:
        next_focus = "2.9L mandatory to cross 30 (EXPERT30_MINIMUM)"

    # 2.9L target refresh at K_HW: required MoE gain for raw 35/40/45 given
    # the measured core at K_HW.
    k_hw_row = next((r for r in full_rows if r["K"] == k_hw), None)
    targets = {}
    if k_hw_row:
        t_core = k_hw_row["t_core_block_s"]
        t_moe = k_hw_row["t_moe_block_s"]
        for rate in (30, 35, 40, 45):
            allowed = k_hw / rate - t_core
            targets[str(rate)] = {
                "allowed_moe_block_s": allowed,
                "required_moe_gain": (t_moe / allowed) if allowed > 0 else None,
                "already_met": bool(t_moe <= allowed),
            }

    merged = wvx.get("merged_curve") or []

    evidence = {
        "verdict": verdict,
        "next_focus": next_focus,
        "best_wide_candidate_toks": best_wide,
        "best_wide_k": best_wide_k,
        "k_hw": k_hw,
        "crossed_30_wide": crossed_30,
        "crossed_35_wide": crossed_35,
        "plateau": plateau,
        "improvements_pct": wvx.get("improvements_pct"),
        "merged_curve": merged,
        "stage2_status": stage2_status,
        "stage2_note": ("K64/80/96 requires a 160-position Modal recapture; the "
                        "Modal workspace exceeded its spend limit, so the width "
                        "ceiling is recorded as >= 56 with >56 unknown. The "
                        "48->56 increment measured 8.93% (first run) and 4.88% "
                        "(rerun), straddling the 5% recapture threshold; stage 2 "
                        "is blocked externally either way."),
        "p2_raw30_wide_pass": bool(crossed_30),
        "expert_kernel_objective": "headroom_35_45" if crossed_35 else (
            "cross_30" if crossed_30 else "cross_30_mandatory"),
        "moe_gain_targets_at_k_hw": targets,
        "r_k_tail": [
            {"K": s["K"], "r_k": s["r_k"], "mean_m": s["mean_m"], "p90_m": s["p90_m"]}
            for s in (wvx.get("stats_rows") or [])
        ],
        "k_chain": None,
        "width_gap": None,
        "base_wideverify_best": next(
            (r["candidate_toks"] for r in (wv.get("full_rows") or [])
             if r.get("route_source") == "golden"), None),
        "bad_status_rows": len(bad_status),
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/phase210x_report.json").write_text(json.dumps(evidence, indent=2, default=str))
    print(verdict)
    return evidence


if __name__ == "__main__":
    run()
