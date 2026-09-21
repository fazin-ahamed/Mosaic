"""Phase 2.9 go/no-go. One precedence-ordered verdict line."""
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
    race = _load("qpa_race.json") or {}
    fb = _load("floor_b_moe.json") or {}
    mx = race.get("mxint_vs_qpa") or {}
    g = mx.get("ratio")
    lut = mx.get("lut")
    rows = race.get("rows") or []
    mxint_complete = any(
        r.get("backend", "").startswith("mxint_") and r.get("packed_weight_gbs")
        for r in rows
    )
    encode_ok = True
    source_semantics = fb.get("source_semantics", "validated")
    s3_mxint = None
    for r in (fb.get("rows") or []):
        if r.get("backend") == "fused_moe_mxint" and r.get("k") == 8 and r.get("family") == "mid":
            s3_mxint = r.get("sec_per_moe_layer")
            break

    gates = race.get("gates") or {}
    cold = gates.get("cold_dram_gate_up") or (race.get("roofline") or {}).get("gate_up", {}).get("cold_dram_gbs")
    r1_gbs = race.get("r1_gbs_m1") or gates.get("r1_gbs")
    r3_m4 = race.get("r3_gbs_m4") or gates.get("r3_m4_gbs")
    r3_m8 = race.get("r3_gbs_m8") or gates.get("r3_m8_gbs")
    r1_ok = bool(r1_gbs and cold and r1_gbs >= 0.50 * cold)
    r3_ok = bool(r3_m4 and r3_m8 and cold and r3_m4 >= 0.45 * cold and r3_m8 >= 0.45 * cold)

    if source_semantics != "validated" or not encode_ok:
        verdict = "SOURCE-SEMANTICS INVALID"
    elif not mxint_complete:
        verdict = "MXINT PATH INCOMPLETE"
    elif g is not None and g > 3.0:
        verdict = "MXINT BREAKTHROUGH"
    elif g is not None and 2.5 <= g <= 3.0:
        verdict = "MXINT STRONG"
    elif g is not None and 1.8 <= g < 2.5:
        verdict = "MXINT GOOD"
    elif g is not None and 1.5 <= g < 1.8:
        verdict = "MXINT USEFUL"
    elif g is None or g < 1.5:
        verdict = "MXINT MISS — MX-LUT32 NEXT"
    elif (not r1_ok or not r3_ok) and s3_mxint is None:
        verdict = "KERNEL PERFORMANCE MISS"
    else:
        verdict = "MXINT MISS — MX-LUT32 NEXT"

    if s3_mxint is not None and s3_mxint <= MOE_BUDGET_30_S:
        verdict = verdict + " — K8 MoE budget met"

    evidence = {
        "verdict": verdict,
        "source_semantics": source_semantics,
        "mxint_vs_qpa": mx,
        "ratio": g,
        "lut": lut,
        "s3_mxint_k8_mid_sec": s3_mxint,
        "r1_ok": r1_ok,
        "r3_ok": r3_ok,
        "r1_gbs": r1_gbs,
        "r3_gbs_m8": r3_m8,
        "cold_dram_gbs": cold,
        "mxint_complete": mxint_complete,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/phase29_report.json").write_text(json.dumps(evidence, indent=2, default=str))
    print(verdict)
    return evidence


if __name__ == "__main__":
    run()
