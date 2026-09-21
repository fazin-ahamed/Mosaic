"""Phase 2.8 go/no-go. One precedence-ordered verdict line."""
from __future__ import annotations

import json
from pathlib import Path

MOE_BUDGET_30_S = 0.178978
MOE_BUDGET_40_S = 0.112311


def _load(name: str):
    p = Path("artifacts") / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def run() -> dict:
    fb = _load("floor_b_moe.json") or {}
    race = _load("qpa_race.json") or {}
    source_semantics = fb.get("source_semantics", "invalid")
    kernel_numeric = fb.get("kernel_numeric", "numeric_parity")
    golden_greedy = fb.get("golden_greedy", "unavailable")
    kernel_path_complete = fb.get("kernel_path_complete", False)
    s3_med = fb.get("s3_block_median_sec")
    s3_auth = fb.get("s3_route_authority", "synthetic_distinct_interim")
    census = fb.get("launch_census") or {}
    ksplit = fb.get("ksplit_metrics") or {}

    gates = race.get("gates") or {}
    cold = gates.get("cold_dram_gate_up") or (race.get("roofline") or {}).get("gate_up", {}).get("cold_dram_gbs")
    r1_gbs = race.get("r1_gbs_m1") or gates.get("r1_gbs")
    r3_m4 = race.get("r3_gbs_m4") or gates.get("r3_m4_gbs")
    r3_m8 = race.get("r3_gbs_m8") or gates.get("r3_m8_gbs")
    r1_ok = bool(r1_gbs and cold and r1_gbs >= 0.50 * cold)
    r3_ok = bool(r3_m4 and r3_m8 and cold and r3_m4 >= 0.45 * cold and r3_m8 >= 0.45 * cold)
    s3_ok = s3_med is not None and s3_med <= MOE_BUDGET_30_S
    s4_ok = s3_med is not None and s3_med <= MOE_BUDGET_40_S

    if source_semantics != "validated":
        verdict = "SOURCE-SEMANTICS INVALID"
    elif not kernel_path_complete:
        verdict = "KERNEL PATH INCOMPLETE"
    elif s4_ok:
        verdict = "MOE STRONG BUDGET PASS — full-model exact floor pending"
    elif s3_ok:
        verdict = "MOE EXACT-30 BUDGET PASS — full-model exact floor pending"
    elif (not s3_ok) and (not r1_ok or not r3_ok):
        verdict = "KERNEL PERFORMANCE MISS"
    elif (not s3_ok) and r1_ok and r3_ok and s3_auth != "golden":
        verdict = "SATURATION MISS ON SYNTHETIC-MID"
    else:
        verdict = "MOE EXACT-30 BUDGET MISS — routed MoE block > 178.98 ms"

    evidence = {
        "verdict": verdict,
        "source_semantics": source_semantics,
        "kernel_numeric": kernel_numeric,
        "golden_greedy": golden_greedy,
        "kernel_path_complete": kernel_path_complete,
        "s3_block_median_sec": s3_med,
        "s3_block_p90_sec": fb.get("s3_block_p90_sec"),
        "s3_route_authority": s3_auth,
        "ksplit_metrics": ksplit,
        "launch_census": census,
        "r1_ok": r1_ok,
        "r3_ok": r3_ok,
        "r1_gbs": r1_gbs,
        "r3_gbs_m4": r3_m4,
        "r3_gbs_m8": r3_m8,
        "cold_dram_gbs": cold,
        "NO_QUALITY_LOSS_VERIFIED": source_semantics == "validated" and golden_greedy == "pass",
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/phase28_report.json").write_text(json.dumps(evidence, indent=2, default=str))
    print(verdict)
    return evidence


if __name__ == "__main__":
    run()
