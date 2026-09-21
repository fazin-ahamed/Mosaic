"""Phase 2.5 G1–G4 ladder. Exit 0 always."""
from __future__ import annotations

import json
from pathlib import Path


def _load(name: str):
    p = Path("artifacts") / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _decision(bps: float | None, toks: float | None, g1: bool | None) -> str:
    if g1 is False:
        # G1 failed still prints a Floor D band if present; last line prefers G1
        # only when Floor D is missing. When Floor D exists, band wins unless
        # candidate_toks is in the special 20–30 / <20 ranges.
        pass
    if toks is None or bps is None:
        return "NO DRAFTER YET — first close dispatch gap"
    if toks < 20:
        return "30 tok/s faithful V4 is not a realistic 3060 target"
    if 20 <= toks < 30:
        return "ACTIVATE Top-4/AED/AEC"
    if bps < 4:
        return "NO-GO for 30 via speculation alone"
    if bps < 5:
        return "BORDERLINE — exceptional drafter acceptance required"
    if bps <= 6:
        return "GO — NanoSpark eligible"
    return "STRONG GO"


def run() -> dict:
    prof = _load("dispatch_profile_k8.json") or {}
    floor_b = _load("floor_b_moe.json") or {}
    floor_cd = _load("floor_cd.json") or {}
    marlin = _load("marlin_status.json") or {}

    rows_b = floor_b.get("rows") or []
    seq = next((r for r in rows_b if r.get("backend") == "sequential" and r.get("k") == 8 and r.get("family") == "mid"), {})
    grp = next((r for r in rows_b if r.get("backend") == "gpu_grouped_triton" and r.get("k") == 8 and r.get("family") == "mid"), {})
    seq_s = seq.get("sec_per_moe_layer")
    grp_s = grp.get("sec_per_moe_layer")
    g1_obs = (seq_s / grp_s) if (seq_s and grp_s) else None
    g1_pass = bool(grp_s and seq_s and grp_s <= 0.5 * seq_s)

    moe_only_bps = grp.get("blocks_per_sec_proj") or seq.get("blocks_per_sec_proj")
    k8d = next((r for r in (floor_cd.get("floor_d") or []) if r.get("K") == 8), {})
    k8c = next((r for r in (floor_cd.get("floor_c") or []) if r.get("K") == 8), {})
    bps = k8d.get("blocks_per_sec")
    toks = k8d.get("candidate_toks")
    # G3: Floor C projected 43-layer candidate at K=8 (no lm_head)
    g3 = None
    if k8c.get("per_layer"):
        g3 = 8.0 / (k8c["per_layer"] * 43)

    print("=== Phase 2.5 G1–G4 ===")
    print(f"{'Gate':<42} {'req':>8} {'obs':>12} {'result':>8}")

    def line(name, req, obs, result):
        obs_s = f"{obs:.3f}" if isinstance(obs, float) else str(obs)
        req_s = f"{req:.1f}" if isinstance(req, float) else str(req)
        print(f"{name:<42} {req_s:>8} {obs_s:>12} {result:>8}")

    line("G1 grouped ≥2× sequential (K8 mid)", 2.0, g1_obs, "PASS" if g1_pass else "FAIL")
    g2_ok = bool(moe_only_bps and moe_only_bps >= 6.0)
    line("G2 MoE-only K8 blocks/s", 6.0, moe_only_bps, "PASS" if g2_ok else "FAIL")
    g3_ok = bool(g3 and g3 >= 40)
    line("G3 Floor C → 43-layer candidate K8", 40.0, g3, "PASS" if g3_ok else "FAIL")
    if bps is None:
        g4 = "missing"
    elif bps < 4:
        g4 = "NO-GO"
    elif bps < 5:
        g4 = "borderline"
    elif bps <= 6:
        g4 = "GO"
    else:
        g4 = "strong GO"
    line("G4 Floor D K8 candidate tok/s", "bands", toks, g4)

    print()
    print("dispatch launches/moe:", prof.get("cuda_kernel_launches_per_moe_layer"),
          "unique:", prof.get("unique_experts"))
    print("floor_b layout:", floor_b.get("expert_layout"),
          "chosen:", floor_b.get("chosen_backend"),
          "n_experts:", floor_b.get("n_experts"),
          "g1:", floor_b.get("g1_k8_mid"))
    print("grouped launches K8 mid:", grp.get("launches_per_moe_layer"))
    print("marlin:", marlin.get("status"), marlin.get("error"))
    print("lm_head:", floor_cd.get("lm_head"))
    print("amdahl_if_moe_inf_toks K8:", k8d.get("amdahl_if_moe_inf_toks"))

    last = _decision(bps, toks, g1_pass)
    print(last)

    out = {
        "g1": {"req": 2.0, "obs": g1_obs, "pass": g1_pass},
        "g2": {"req": 6.0, "obs": moe_only_bps, "pass": g2_ok, "comfort": True},
        "g3": {"req": 40.0, "obs": g3, "pass": g3_ok},
        "g4": {"obs_bps": bps, "obs_toks": toks, "band": g4},
        "last_line": last,
        "marlin": marlin,
    }
    Path("artifacts/phase25_report.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    run()
