"""Phase 2.9R go/no-go. One precedence-ordered verdict line."""
from __future__ import annotations

import json
from pathlib import Path

import torch

MOE_BUDGET_30_S = 0.178978
ALLOWED_STATUS = {"ok", "invalid_not_256", "compile_fail"}
ALLOWED_PREFIX = ("error:",)


def _load(name: str):
    p = Path("artifacts") / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _p1_parity_check() -> dict:
    """Tiny E=8 fixture: fused_moe_mxint (Hot8) vs fused_moe_qpa numeric parity."""
    import math

    from mosaicvm.kernels.fp4_source import pack_mxfp4_nk
    from mosaicvm.kernels.mxint import fused_moe_mxint
    from mosaicvm.kernels.qpa_mma import fused_moe_qpa

    torch.manual_seed(7)
    k, top_k, e, h, inter = 4, 2, 8, 128, 128
    x = torch.randn(k, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, e, (k, top_k), device="cuda")
    weights = torch.rand(k, top_k, device="cuda", dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    pgu, sgu, pdu, sdu = [], [], [], []
    for _ in range(e):
        pg, sg = pack_mxfp4_nk((torch.randn(2 * inter, h, device="cuda") / math.sqrt(h)).to(torch.bfloat16))
        pd, sd = pack_mxfp4_nk((torch.randn(h, inter, device="cuda") / math.sqrt(inter)).to(torch.bfloat16))
        pgu.append(pg); sgu.append(sg); pdu.append(pd); sdu.append(sd)
    packed_gu = torch.stack(pgu)
    scale_gu = torch.stack(sgu)
    packed_d = torch.stack(pdu)
    scale_d = torch.stack(sdu)
    yq = fused_moe_qpa(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d)
    ym = fused_moe_mxint(x, ids, weights, packed_gu, scale_gu, packed_d, scale_d,
                         weight_repr="hot8")
    af, bf = ym.float(), yq.float()
    fin = torch.isfinite(af) & torch.isfinite(bf)
    nan_inf = int((~torch.isfinite(af) ^ ~torch.isfinite(bf)).sum().item())
    return {
        "max_abs": float((af - bf).abs().max()),
        "nan_inf_mismatches": nan_inf,
        "shape": list(ym.shape),
    }


def run() -> dict:
    rep = _load("phase29r.json") or {}
    race = _load("qpa_race.json") or {}
    fb = _load("floor_b_moe.json") or {}
    integrated = rep.get("integrated") or {}
    g = integrated.get("hot8_ratio")
    if g is None and integrated.get("packed4_k8_mid_sec") and integrated.get("qpa_k8_mid_sec"):
        g = integrated["qpa_k8_mid_sec"] / integrated["packed4_k8_mid_sec"]
    g_p4 = integrated.get("packed4_ratio")
    s3 = rep.get("s3_mxint_block_median_sec")

    hot8_rows = [r for r in (rep.get("rows") or [])
                 if r.get("backend") == "fused_moe_mxint" and r.get("weight_repr") == "hot8"]
    hot8_ok = any(r.get("sec_per_moe_layer") for r in hot8_rows)
    hot8_p1_sec = integrated.get("hot8_k8_mid_sec")

    parity = None
    try:
        parity = _p1_parity_check()
    except Exception as exc:
        parity = {"error": f"{type(exc).__name__}:{str(exc)[:120]}"}

    bad_status = [
        r for r in (rep.get("rows") or [])
        if r.get("status") not in ALLOWED_STATUS
        and not str(r.get("status", "")).startswith(ALLOWED_PREFIX)
    ]
    numeric_bad = bool(parity and parity.get("nan_inf_mismatches", 1) > 0)

    source_semantics = fb.get("source_semantics", "validated")
    act_quant_fuse = "skipped_integrated_ge_1.5" if (g is not None and g >= 1.5) else "not_evaluated"

    packed4_much_slower = bool(g_p4 is not None and g is not None and g_p4 < 0.70 * g)

    if source_semantics != "validated":
        verdict = "SOURCE-SEMANTICS INVALID"
    elif not hot8_ok and not integrated.get("packed4_k8_mid_sec"):
        verdict = "MXINT P1 INCOMPLETE"
    elif bad_status or numeric_bad:
        verdict = "UNPLANNED_DEVIATION"
    elif g is None:
        verdict = "MXINT P1 INCOMPLETE"
    elif g > 3.0:
        verdict = "MXINT P1 BREAKTHROUGH"
    elif 2.5 <= g <= 3.0:
        verdict = "MXINT P1 STRONG"
    elif 1.8 <= g < 2.5:
        verdict = "MXINT P1 GOOD"
    elif 1.5 <= g < 1.8:
        verdict = "MXINT P1 USEFUL"
    elif 1.25 <= g < 1.5:
        verdict = "MXINT P1 MARGINAL"
    else:
        verdict = "MXINT P1 MISS — MX-LUT32 NEXT"

    next_phase = None
    if verdict == "MXINT P1 USEFUL" or "GOOD" in verdict or "STRONG" in verdict or "BREAKTHROUGH" in verdict:
        next_phase = "2.10 WideVerify"
        lut = "not_now"
    elif verdict == "MXINT P1 MARGINAL":
        next_phase = "act_quant+qA fusion then retest; still <1.5x -> MX-LUT"
        lut = "reopen_if_fuse_fails"
    elif verdict.startswith("MXINT P1 MISS"):
        next_phase = "2.9L MX-LUT32"
        lut = "build_lut_n16"
    else:
        next_phase = "stop_architectural_work"
        lut = "n/a"

    if packed4_much_slower:
        verdict += " — packed4 << Hot8, Hot8 is cache-only"
    if s3 is not None and s3 <= MOE_BUDGET_30_S:
        verdict += " — K8 MoE budget met"

    evidence = {
        "verdict": verdict,
        "next_phase": next_phase,
        "lut": lut,
        "source_semantics": source_semantics,
        "p0_ratio": (race.get("mxint_vs_qpa") or {}).get("ratio"),
        "integrated": integrated,
        "hot8_p1_k8_mid_sec": hot8_p1_sec,
        "hot8_status": rep.get("hot8_status"),
        "s3_mxint_block_median_sec": s3,
        "s3_mxint_block_p90_sec": rep.get("s3_mxint_block_p90_sec"),
        "stage_times_k8_hot8": rep.get("stage_times_k8_hot8"),
        "act_quant_fuse": act_quant_fuse,
        "packed4_much_slower": packed4_much_slower,
        "p1_parity_vs_qpa_fused": parity,
        "launch_census": rep.get("launch_census"),
        "bad_status_rows": len(bad_status),
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/phase29r_report.json").write_text(json.dumps(evidence, indent=2, default=str))
    print(verdict)
    return evidence


if __name__ == "__main__":
    run()
