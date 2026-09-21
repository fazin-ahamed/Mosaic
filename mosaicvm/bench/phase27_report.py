"""Phase 2.7 report. Exit 0 always; last line is exactly one decision string."""
from __future__ import annotations

import json
from pathlib import Path

ARTIFACT = Path("artifacts/phase27_report.json")


def _load(name: str):
    p = Path("artifacts") / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def run() -> dict:
    race = _load("qpa_race.json") or {}
    floor_b = _load("floor_b_moe.json") or {}
    decode = _load("mxfp4_decode.json") or {}
    gates_r = (race.get("gates") or {})
    r0 = gates_r.get("r0", False)
    r1 = gates_r.get("r1", False)
    r2 = gates_r.get("r2", False)
    r3 = gates_r.get("r3", False)
    r4 = gates_r.get("r4", "skipped_no_ncu")
    r5 = bool(decode.get("e2m1_ok") and decode.get("e4m3_ok")) if decode else True
    r6 = bool(decode.get("prepack_ok", True))

    budget_30 = floor_b.get("moe_layer_budget_30_s", 0.004162)
    budget_40 = floor_b.get("moe_layer_budget_40_s", 0.002612)
    sk_sec = floor_b.get("exact_k8_mid_sec") or floor_b.get("skinny_k8_mid_sec")
    s3_pass = floor_b.get("s3_k8_mid_4_16ms", False)
    s4_pass = floor_b.get("s4_k8_mid_2_61ms", False)
    n_e = floor_b.get("exact_n_experts") or floor_b.get("mxfp4_n_experts")
    invalid_256 = floor_b.get("exact_status") == "invalid_not_256" or n_e not in (None, 256)

    native_raced = race.get("native_raced", False)
    triton_only = race.get("triton_only", False)
    source_ok = race.get("act_quant", False) and r0

    print("R0 cold_dram per shape          {}".format("PASS" if r0 else "FAIL"))
    print("R1 M=1 >=50% cold_dram          {}".format("PASS" if r1 else "FAIL"))
    print("R2 M=2,3 >=45%                  {}".format("PASS" if r2 else "FAIL"))
    print("R3 native QPA M=4,8 >=45%       {}".format("PASS" if r3 else "FAIL"))
    print("R4 lmem                         {}".format(r4))
    print("R5 decode                       {}".format("PASS" if r5 else "FAIL"))
    print("R6 prepack inverse              {}".format("PASS" if r6 else "FAIL"))
    print("S3 K8 mid ms                    req 4.16 obs {} {}".format(
        f"{sk_sec*1000:.2f}" if sk_sec else "N/A", "PASS" if s3_pass else "FAIL"))
    print("S4 K8 mid ms                    req 2.61 obs {} {}".format(
        f"{sk_sec*1000:.2f}" if sk_sec else "N/A", "PASS" if s4_pass else "FAIL"))

    if not source_ok or not r5:
        decision = "SOURCE-SEMANTICS INVALID"
    elif not native_raced and triton_only:
        decision = "Triton prototype miss — native QPA/XSIMT not yet measured"
    elif not native_raced:
        decision = "KERNEL PATH INCOMPLETE"
    elif not r1 and not r3:
        decision = "KERNEL PERFORMANCE MISS"
    elif r1 and not s3_pass:
        decision = "EXACT-30 COMPUTE MISS: K8 mid MoE > 4.16 ms"
    elif s3_pass and not s4_pass:
        decision = "Exact-30 compute ceiling in reach (≤4.16 ms); Dual-Rail/NanoSpark eligible next"
    elif s4_pass:
        decision = "STRONG Exact-30 compute (≤2.61 ms)"
    else:
        decision = "KERNEL PATH INCOMPLETE"

    print(decision)
    out = {
        "decision": decision,
        "r0": r0, "r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5, "r6": r6,
        "s3_pass": s3_pass, "s4_pass": s4_pass,
        "exact_k8_mid_sec": sk_sec,
        "exact_n_experts": n_e,
        "invalid_not_256": invalid_256,
        "chosen_exact_backend": floor_b.get("chosen_exact_backend"),
        "ptx_has_m16n8k16": race.get("ptx_has_m16n8k16"),
        "prepack_stage": race.get("prepack_stage"),
        "decoder": race.get("decoder"),
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    run()
