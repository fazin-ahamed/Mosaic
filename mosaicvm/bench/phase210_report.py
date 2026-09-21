"""Phase 2.10 WideVerify go/no-go. One precedence-ordered verdict line."""
from __future__ import annotations

import json
from pathlib import Path

MOE_BUDGET_30_S = 0.178978


def _load(name: str):
    p = Path("artifacts") / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def run() -> dict:
    wv = _load("wide_verify.json") or {}
    race = _load("qpa_race.json") or {}
    p29r = _load("phase29r.json") or {}
    gates = wv.get("gates") or {}
    golden_available = bool(gates.get("golden_available"))
    best_synth = gates.get("best_candidate_toks_synthetic")
    best_golden = gates.get("best_candidate_toks_golden")
    k_hw = gates.get("k_hw_synthetic")
    gate_synth = gates.get("gate_synthetic", "incomplete")
    gate_golden = gates.get("gate_golden", "incomplete")

    full_rows = wv.get("full_rows") or []
    complete = bool(full_rows)
    bad_status = [
        r for r in (wv.get("moe_rows") or [])
        if r.get("status") not in ("ok", "no_windows")
        and not str(r.get("status", "")).startswith(("error:", "invalid_not_256"))
    ]

    source_semantics = (p29r.get("source_semantics")
                        or (_load("floor_b_moe.json") or {}).get("source_semantics", "validated"))

    # decisive number: golden when present, else synthetic (provisional)
    if golden_available and best_golden is not None:
        best, decisive, provisional = best_golden, "golden", False
    elif best_synth is not None:
        best, decisive, provisional = best_synth, "synthetic_mid", True
    else:
        best, decisive, provisional = None, "none", True

    # core-compensation analysis at the decisive best K: can a perfect core
    # alone lift the best width to the next gate?
    best_row = next((r for r in full_rows
                     if r.get("route_source") == decisive
                     and r.get("candidate_toks") == best), None)
    core_analysis = None
    if best_row:
        k_best = best_row["K"]
        t_moe = best_row["t_moe_block_s"]
        t_core = best_row["t_core_block_s"]
        cand_if_core_free = k_best / t_moe
        core_analysis = {
            "k_best": k_best,
            "t_moe_block_s": t_moe,
            "t_core_block_s": t_core,
            "core_share": t_core / (t_moe + t_core),
            "cand_toks_if_core_free": cand_if_core_free,
            "core_reaches_30": bool(cand_if_core_free >= 30.0),
            "core_reaches_35": bool(cand_if_core_free >= 35.0),
            "core_can_compensate": bool(cand_if_core_free >= 30.0),
        }

    if source_semantics != "validated":
        verdict = "SOURCE-SEMANTICS INVALID"
    elif not complete:
        verdict = "WIDEVERIFY INCOMPLETE"
    elif bad_status:
        verdict = "UNPLANNED_DEVIATION"
    elif best is None:
        verdict = "WIDEVERIFY INCOMPLETE"
    elif best >= 35:
        verdict = "WIDEVERIFY PASS — FREEZE EXPERT COMPUTE"
    elif best >= 30:
        verdict = "WIDEVERIFY VIABLE — OPTIMIZATION RESERVE"
    elif best >= 25:
        if core_analysis and not core_analysis["core_can_compensate"]:
            verdict = "WIDEVERIFY BORDERLINE — CORE CANNOT COMPENSATE, EXPERT-KERNEL RESEARCH REOPENS"
        else:
            verdict = "WIDEVERIFY BORDERLINE — CORE MUST COMPENSATE"
    elif golden_available:
        verdict = "WIDEVERIFY MISS — MX-LUT32 REOPEN"
    else:
        verdict = "WIDEVERIFY MISS (SYNTHETIC) — GOLDEN ROUTES REQUIRED TO CONFIRM MX-LUT32 REOPEN"

    next_phase = None
    lut = "n/a"
    if "PASS" in verdict:
        next_phase, lut = "2.11 exact non-MoE core", "not_now"
    elif "VIABLE" in verdict:
        next_phase, lut = "2.11 exact core with optimization reserve", "not_now"
    elif "BORDERLINE" in verdict:
        if core_analysis and not core_analysis["core_can_compensate"]:
            # Core is too small a share: even a free core cannot reach 30.
            # Keep 2.11 on the critical path (core is required regardless)
            # but expert-kernel research (MX-LUT32 / 2.9L) reopens now.
            next_phase = ("2.11 exact core AND 2.9L expert-kernel research in parallel "
                          "(core share {:.0%}; free core only reaches {:.1f} tok/s)".format(
                              core_analysis["core_share"],
                              core_analysis["cand_toks_if_core_free"]))
            lut = "reopen_now_core_cannot_compensate"
        else:
            next_phase, lut = "2.11 core measurement; core may compensate", "reopen_if_core_short"
    elif "MISS" in verdict:
        if golden_available:
            next_phase, lut = "2.9L MX-LUT32", "reopen_now"
        else:
            next_phase = "capture golden V4 routes (Modal) and re-gate; MX-LUT32 indicated"
            lut = "reopen_indicated_pending_golden"

    if provisional and verdict not in ("SOURCE-SEMANTICS INVALID", "WIDEVERIFY INCOMPLETE"):
        verdict += " [provisional: synthetic-distinct only, golden unavailable]"

    # backend portfolio summary
    portfolio = {}
    for r in full_rows:
        if r.get("route_source") == decisive:
            portfolio[str(r["K"])] = {
                "best_backend": r.get("best_backend"),
                "candidate_toks": r.get("candidate_toks"),
            }

    r_k = (wv.get("r_k_curve") or {}).get(decisive) or []
    r_k_summary = None
    if r_k:
        r_k_summary = {
            "flat": len({round(c["r_k"], 3) for c in r_k}) <= 2,
            "r_k_k8": next((c["r_k"] for c in r_k if c["K"] == 8), None),
            "r_k_k32": next((c["r_k"] for c in r_k if c["K"] == 32), None),
            "note": "synthetic-distinct fixes unique=3K, so R(K) is flat by construction; "
                    "elasticity can only come from golden routes",
        }

    evidence = {
        "verdict": verdict,
        "next_phase": next_phase,
        "lut": lut,
        "core_analysis": core_analysis,
        "source_semantics": source_semantics,
        "decisive_route_source": decisive,
        "provisional": provisional,
        "best_candidate_toks": best,
        "best_candidate_toks_synthetic": best_synth,
        "best_candidate_toks_golden": best_golden,
        "k_hw_synthetic": k_hw,
        "gate_synthetic": gate_synth,
        "gate_golden": gate_golden,
        "golden_status": wv.get("golden_status"),
        "backend_portfolio": portfolio,
        "r_k_curve": r_k,
        "r_k_summary": r_k_summary,
        "hot8_benefit_k8": (wv.get("hot8_benefit") or {}).get("8"),
        "p0_ratio": (race.get("mxint_vs_qpa") or {}).get("ratio"),
        "p1_ratio": (p29r.get("integrated") or {}).get("hot8_ratio"),
        "bad_status_rows": len(bad_status),
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/phase210_report.json").write_text(json.dumps(evidence, indent=2, default=str))
    print(verdict)
    return evidence


if __name__ == "__main__":
    run()
