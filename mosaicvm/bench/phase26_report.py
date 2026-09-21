"""Phase 2.6 report: S1-S4 gates and the Exact-30 decision string.

Reads skinny_race.json, floor_b_moe.json, floor_cd.json, routing_coverage.json.
Exit 0 always; last line is exactly one decision string.
"""
from __future__ import annotations

import json
from pathlib import Path

ARTIFACT = Path("artifacts/phase26_report.json")


def _load(name: str):
    p = Path("artifacts") / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def run() -> dict:
    race = _load("skinny_race.json") or {}
    floor_b = _load("floor_b_moe.json") or {}
    floor_d = _load("floor_cd.json") or {}
    coverage = _load("routing_coverage.json") or {}

    gates = race.get("gates", {})
    s1_pass = gates.get("s1_pass", False)
    s1_obs = gates.get("s1_m1_gbs_obs")
    s2_ratio = gates.get("s2_m1_ratio_obs")

    s3_pass = floor_b.get("s3_k8_mid_4_16ms", False)
    sk_sec = floor_b.get("skinny_k8_mid_sec")
    s4_pass = floor_b.get("s4_k8_mid_2_61ms", False)

    budget_30 = floor_b.get("moe_layer_budget_30_s", 0.004162)
    budget_40 = floor_b.get("moe_layer_budget_40_s", 0.002612)

    rows = [
        {"gate": "S1", "metric": "M=1 gate_up packed GB/s", "req": 100.0,
         "obs": round(s1_obs, 2) if s1_obs else None, "pass": s1_pass},
        {"gate": "S2", "metric": "M=1 skinny <=0.5x grouped", "req": 2.0,
         "obs": round(s2_ratio, 2) if s2_ratio else None, "pass": gates.get("s2_pass", False),
         "comfort": True},
        {"gate": "S3", "metric": "K8 mid ms/MoE layer", "req": round(budget_30 * 1000, 2),
         "obs": round(sk_sec * 1000, 2) if sk_sec else None, "pass": s3_pass},
        {"gate": "S4", "metric": "K8 mid ms/MoE layer", "req": round(budget_40 * 1000, 2),
         "obs": round(sk_sec * 1000, 2) if sk_sec else None, "pass": s4_pass, "comfort": True},
    ]

    print("S1 M=1 gate_up packed GB/s     req 100    obs {}   {}".format(
        f"{s1_obs:.2f}" if s1_obs else "N/A", "PASS" if s1_pass else "FAIL"))
    print("S2 M=1 skinny <=0.5x grouped    req 2.0    obs {}   (comfort)".format(
        f"{s2_ratio:.2f}" if s2_ratio else "N/A"))
    print("S3 K8 mid ms/MoE layer         req {}   obs {}   {}".format(
        round(budget_30 * 1000, 2),
        f"{sk_sec*1000:.2f}" if sk_sec else "N/A",
        "PASS" if s3_pass else "FAIL"))
    print("S4 K8 mid ms/MoE layer         req {}   obs {}   (comfort)".format(
        round(budget_40 * 1000, 2),
        f"{sk_sec*1000:.2f}" if sk_sec else "N/A"))

    if not s1_pass:
        decision = "SKINNY M=1 < 100 GB/s — SIMT kernel still memory-starved"
    elif not s3_pass:
        decision = "NO-GO Exact-30: K8 mid MoE > 4.16 ms"
    elif not s4_pass:
        decision = "Exact-30 compute ceiling in reach (≤4.16 ms); Dual-Rail/NanoSpark eligible next"
    else:
        decision = "STRONG Exact-30 compute (≤2.61 ms)"

    print(decision)

    out = {
        "gates": rows,
        "decision": decision,
        "s1_pass": s1_pass,
        "s3_pass": s3_pass,
        "s4_pass": s4_pass,
        "skinny_k8_mid_sec": sk_sec,
        "chosen_exact_backend": floor_b.get("chosen_exact_backend"),
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
