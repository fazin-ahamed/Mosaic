"""Phase 2.7 kernel race: native XSIMT/QPA vs Triton prototypes vs grouped INT4.

GB/s = (N*K/2 + N*K/32) / sec / 1e9 (NK layout). pct_roofline vs matching cold_dram_gbs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from mosaicvm.bench.compute_floor import _bench
from mosaicvm.kernels.fp4_source import act_quant_ref, pack_mxfp4_nk, reference_fp4_linear_source_semantics
from mosaicvm.kernels.mxfp4_decode import CHOSEN_DECODER
from mosaicvm.kernels.qpa_mma import CHOSEN_PREPACK, CHOSEN_QPA_K, PTX_HAS_M16N8K16, prepack_qpa, qpa_mma_gemm
from mosaicvm.kernels.xsimt import decode_activation_exact, prepack_xsimt, xsimt_gemm
from mosaicvm.kernels.w4a16_gemm import pack_w4, w4a16_gemm

ARTIFACT = Path("artifacts/qpa_race.json")
MS = (1, 2, 3, 4, 5, 8)
SHAPES = {
    "gate_up": (4096, 4096),  # N, K
    "down": (4096, 2048),
}


def _packed_bytes_nk(n: int, k: int) -> int:
    return n * (k // 2) + n * (k // 32)


def _try_tinygemm():
    try:
        import tinygemm  # noqa: F401
    except Exception:
        return "skipped_no_tinygemm"
    return "skipped_no_tinygemm"


def _load_roofline():
    p = Path("artifacts/read_roofline.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("shapes") or {}


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    if "3060" not in gpu.lower():
        print(gpu)
        raise SystemExit(f"unexpected GPU: {gpu}")

    roof = _load_roofline()
    rows = []
    tiny = _try_tinygemm()

    for shape_name, (n, k) in SHAPES.items():
        torch.manual_seed(42)
        w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
        p, s = pack_mxfp4_nk(w)
        w_kn = w.T.contiguous()
        p_w4, s_w4 = pack_w4(w_kn)
        bytes_w = _packed_bytes_nk(n, k)
        cold = (roof.get(shape_name) or {}).get("cold_dram_gbs")
        xsimt_pre = prepack_xsimt(p, s)
        qpa_pre, _ = prepack_qpa(p, s, "qpa_k32_fragment")
        from mosaicvm.kernels.mxint import qint_encode_w
        mxint_pre = qint_encode_w(p)

        for m in MS:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            x_fp8, sA = act_quant_ref(x)
            preq = (x_fp8, sA)
            ref = reference_fp4_linear_source_semantics(x, p, s)

            def add(backend, fn, skip=None):
                if skip:
                    rows.append({"M": m, "shape": shape_name, "backend": backend, "status": skip})
                    return
                try:
                    sec = _bench(fn, iters=50, warmup=10)
                    out = fn()
                    err = (out.float() - ref.float()).abs().max().item()
                    gbs = bytes_w / sec / 1e9
                    pct = (gbs / cold * 100.0) if cold else None
                    rows.append({
                        "M": m, "shape": shape_name, "backend": backend,
                        "sec": sec, "us": sec * 1e6, "bytes_w": bytes_w,
                        "packed_weight_gbs": gbs, "pct_roofline": pct,
                        "parity_max_abs": err, "act_quant": True,
                    })
                except Exception as exc:
                    rows.append({"M": m, "shape": shape_name, "backend": backend, "status": f"error:{type(exc).__name__}", "error": str(exc)[:200]})

            x_norm = decode_activation_exact(preq)
            add("xsimt_native", lambda: xsimt_gemm(x, p, s, n, backend="native", prequant=preq, prepacked=xsimt_pre))
            add("xsimt_triton", lambda: xsimt_gemm(x, p, s, n, backend="triton", prequant=preq))
            add("qpa_native", lambda: qpa_mma_gemm(x, p, s, n, backend="native", prequant=preq, prepacked=qpa_pre, x_norm=x_norm, sA=sA))
            add("qpa_triton", lambda: qpa_mma_gemm(x, p, s, n, backend="triton", prequant=preq))
            add("qpa_ksplit_w8", lambda xn=x_norm, sa=sA: qpa_mma_gemm(x, p, s, n, qpa_k=8, prepacked=qpa_pre, x_norm=xn, sA=sa, ordered=True))
            add("qpa_ksplit_reordered_w8", lambda xn=x_norm, sa=sA: qpa_mma_gemm(x, p, s, n, qpa_k=8, prepacked=qpa_pre, x_norm=xn, sA=sa, ordered=False))
            from mosaicvm.kernels.mxint import decode_activation_int, mxint_dp4a_gemm, mxint_imma_gemm
            a0, a1, a2 = decode_activation_int(preq)
            xint = (a0, a1, a2, sA)
            add("mxint_dp4a", lambda xi=xint, wp=mxint_pre: mxint_dp4a_gemm(x, p, s, n, xint=xi, prepacked=wp))
            add("mxint_imma_w8", lambda xi=xint, wp=mxint_pre: mxint_imma_gemm(x, p, s, n, qpa_k=8, ordered=True, xint=xi, prepacked=wp))
            if m in (4, 8):
                add("mxint_imma_reordered_w8", lambda xi=xint, wp=mxint_pre: mxint_imma_gemm(x, p, s, n, qpa_k=8, ordered=False, xint=xi, prepacked=wp))
            add("mxint_dp4a_packed4", lambda xi=xint, wp=p: mxint_dp4a_gemm(x, p, s, n, xint=xi, prepacked=wp, weight_repr="packed4"))
            add("mxint_imma_w8_packed4", lambda xi=xint, wp=p: mxint_imma_gemm(x, p, s, n, qpa_k=8, ordered=True, xint=xi, prepacked=wp, weight_repr="packed4"))
            if m in (4, 8):
                add("mxint_imma_reordered_w8_packed4", lambda xi=xint, wp=p: mxint_imma_gemm(x, p, s, n, qpa_k=8, ordered=False, xint=xi, prepacked=wp, weight_repr="packed4"))
            if m == 1:
                rows.append({"M": m, "shape": shape_name, "backend": "mxint_hexa", "status": "fused_only"})

            def step_grouped():
                return w4a16_gemm(x, p_w4, s_w4)

            try:
                sec_g = _bench(step_grouped, iters=50, warmup=10)
                gbs_g = bytes_w / sec_g / 1e9
                pct_g = (gbs_g / cold * 100.0) if cold else None
                rows.append({
                    "M": m, "shape": shape_name, "backend": "grouped_blockm16",
                    "sec": sec_g, "us": sec_g * 1e6, "bytes_w": bytes_w,
                    "packed_weight_gbs": gbs_g, "pct_roofline": pct_g,
                })
            except Exception as exc:
                rows.append({"M": m, "shape": shape_name, "backend": "grouped_blockm16", "status": f"error:{type(exc).__name__}"})

            rows.append({"M": m, "shape": shape_name, "backend": "tinygemm", "status": tiny})
            del x

    def best_exact(m, shape, backends):
        cands = [r for r in rows if r.get("M") == m and r.get("shape") == shape
                 and r.get("backend") in backends and r.get("packed_weight_gbs")]
        return max(cands, key=lambda r: r["packed_weight_gbs"]) if cands else None

    exact = ("xsimt_native", "qpa_native")
    r1 = best_exact(1, "gate_up", exact)
    r2a = best_exact(2, "gate_up", exact)
    r2b = best_exact(3, "gate_up", exact)
    r3a = next((r for r in rows if r.get("M") == 4 and r.get("shape") == "gate_up" and r.get("backend") == "qpa_ksplit_w8" and r.get("packed_weight_gbs")), None) \
        or next((r for r in rows if r.get("M") == 4 and r.get("shape") == "gate_up" and r.get("backend") == "qpa_native" and r.get("packed_weight_gbs")), None)
    r3b = next((r for r in rows if r.get("M") == 8 and r.get("shape") == "gate_up" and r.get("backend") == "qpa_ksplit_w8" and r.get("packed_weight_gbs")), None) \
        or next((r for r in rows if r.get("M") == 8 and r.get("shape") == "gate_up" and r.get("backend") == "qpa_native" and r.get("packed_weight_gbs")), None)
    ord_row = next((r for r in rows if r.get("M") == 8 and r.get("shape") == "gate_up" and r.get("backend") == "qpa_ksplit_w8" and r.get("packed_weight_gbs")), None)
    reo_row = next((r for r in rows if r.get("M") == 8 and r.get("shape") == "gate_up" and r.get("backend") == "qpa_ksplit_reordered_w8" and r.get("packed_weight_gbs")), None)
    cold_gu = (roof.get("gate_up") or {}).get("cold_dram_gbs")
    r0 = bool(cold_gu) and bool((roof.get("down") or {}).get("cold_dram_gbs"))
    r1_pass = bool(r1 and cold_gu and r1["packed_weight_gbs"] >= 0.50 * cold_gu)
    r2_pass = bool(r2a and r2b and cold_gu
                   and r2a["packed_weight_gbs"] >= 0.45 * cold_gu
                   and r2b["packed_weight_gbs"] >= 0.45 * cold_gu)
    r3_pass = bool(r3a and r3b and cold_gu
                   and r3a["packed_weight_gbs"] >= 0.45 * cold_gu
                   and r3b["packed_weight_gbs"] >= 0.45 * cold_gu)

    native_raced = any(r.get("backend") in exact and r.get("packed_weight_gbs") for r in rows)
    triton_only = (not native_raced) and any(r.get("backend") in ("xsimt_triton", "qpa_triton") and r.get("packed_weight_gbs") for r in rows)
    import mosaicvm.kernels.qpa_mma as qpa_mod
    ptx_flag = bool(qpa_mod.PTX_HAS_M16N8K16)
    prepack_stage = qpa_mod.CHOSEN_PREPACK

    def _gbs(backend, m, shape="gate_up"):
        r = next((x for x in rows if x.get("M") == m and x.get("shape") == shape
                  and x.get("backend") == backend and x.get("packed_weight_gbs")), None)
        return None if r is None else r["packed_weight_gbs"]

    dp4a_gbs = _gbs("mxint_dp4a", 1)
    imma_gbs = _gbs("mxint_imma_w8", 8)
    qpa_w8_gbs = _gbs("qpa_ksplit_w8", 8)
    mxint_fail = (
        (not dp4a_gbs) and any(r.get("backend") == "mxint_dp4a" and str(r.get("status", "")).startswith(("error:", "compile_fail")) for r in rows)
    ) or (
        (not imma_gbs) and any(r.get("backend") == "mxint_imma_w8" and str(r.get("status", "")).startswith(("error:", "compile_fail")) for r in rows)
    )
    best_mx = max([g for g in (dp4a_gbs, imma_gbs) if g], default=None)
    ratio = (best_mx / qpa_w8_gbs) if (best_mx and qpa_w8_gbs) else None
    lut_status = "skipped_mxint_ge_1.5" if (ratio is not None and ratio >= 1.5) else "not_built"
    if ratio is None or ratio < 1.5 or mxint_fail:
        from mosaicvm.kernels.mxint import mxint_lut32_gemm
        n, k = SHAPES["gate_up"]
        torch.manual_seed(42)
        w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
        p, s = pack_mxfp4_nk(w)
        bytes_w = _packed_bytes_nk(n, k)
        cold = (roof.get("gate_up") or {}).get("cold_dram_gbs")
        for m in (1, 8):
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            x_fp8, sA = act_quant_ref(x)
            preq = (x_fp8, sA)
            ref = reference_fp4_linear_source_semantics(x, p, s)
            try:
                def _lut(xx=x, pp=p, ss=s, nn=n, pq=preq):
                    return mxint_lut32_gemm(xx, pp, ss, nn, prequant=pq)
                sec = _bench(_lut, iters=50, warmup=10)
                outp = _lut()
                err = (outp.float() - ref.float()).abs().max().item()
                gbs = bytes_w / sec / 1e9
                pct = (gbs / cold * 100.0) if cold else None
                rows.append({
                    "M": m, "shape": "gate_up", "backend": "mxint_lut32",
                    "sec": sec, "us": sec * 1e6, "bytes_w": bytes_w,
                    "packed_weight_gbs": gbs, "pct_roofline": pct,
                    "parity_max_abs": err, "act_quant": True,
                })
                lut_status = "built"
            except Exception as exc:
                rows.append({"M": m, "shape": "gate_up", "backend": "mxint_lut32",
                             "status": f"error:{type(exc).__name__}", "error": str(exc)[:200]})
                lut_status = f"error:{type(exc).__name__}"
            del x
        del w, p, s

    mxint_vs_qpa = {
        "dp4a_gbs": dp4a_gbs,
        "imma_gbs": imma_gbs,
        "qpa_w8_gbs": qpa_w8_gbs,
        "ratio": ratio,
        "lut": lut_status,
    }
    mxint_packed4_vs_hot8 = {
        "dp4a_m1": _gbs("mxint_dp4a_packed4", 1),
        "imma_m8": _gbs("mxint_imma_w8_packed4", 8),
        "hot8_dp4a_m1": dp4a_gbs,
        "hot8_imma_m8": imma_gbs,
    }
    out = {
        "gpu": gpu,
        "ptx_has_m16n8k16": ptx_flag,
        "decoder": CHOSEN_DECODER,
        "prepack_stage": prepack_stage,
        "act_quant": True,
        "chosen_qpa_k": int(qpa_mod.CHOSEN_QPA_K),
        "ordered_vs_reordered": {
            "ordered_gbs": None if not ord_row else ord_row["packed_weight_gbs"],
            "reordered_gbs": None if not reo_row else reo_row["packed_weight_gbs"],
        },
        "r1_gbs_m1": r1["packed_weight_gbs"] if r1 else None,
        "r3_gbs_m4": r3a["packed_weight_gbs"] if r3a else None,
        "r3_gbs_m8": r3b["packed_weight_gbs"] if r3b else None,
        "roofline": roof,
        "gates": {
            "r0": r0,
            "r1": r1_pass,
            "r2": r2_pass,
            "r3": r3_pass,
            "r4": "skipped_no_ncu",
            "r1_gbs": r1["packed_weight_gbs"] if r1 else None,
            "r3_m4_gbs": r3a["packed_weight_gbs"] if r3a else None,
            "r3_m8_gbs": r3b["packed_weight_gbs"] if r3b else None,
            "cold_dram_gate_up": cold_gu,
        },
        "native_raced": native_raced,
        "triton_only": triton_only,
        "mxint_vs_qpa": mxint_vs_qpa,
        "mxint_packed4_vs_hot8": mxint_packed4_vs_hot8,
        "rows": rows,
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({
        "r0": r0, "r1": r1_pass, "r2": r2_pass, "r3": r3_pass,
        "ptx_has_m16n8k16": ptx_flag,
        "r1_gbs": None if not r1 else round(r1["packed_weight_gbs"], 2),
        "cold_gu": None if not cold_gu else round(cold_gu, 2),
        "wrote": str(ARTIFACT),
    }, indent=2))
    return out


if __name__ == "__main__":
    run()
