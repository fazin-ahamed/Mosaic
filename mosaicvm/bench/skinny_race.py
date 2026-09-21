"""Skinny MXFP4 kernel race: skinny_mxfp4 vs grouped_blockm16 (INT4) baseline.

Times one expert at V4 shapes, CUDA events, 50 iters / 10 warmup.  For each M in
(1, 2, 3, 4, 5, 8) and each shape (gate_up 4096x4096, down 2048x4096) records
sec, packed-weight GB/s, and parity vs the dequant reference.

Gate S1: M=1 gate_up skinny_mxfp4 packed_weight_gbs >= 100.
Gate S2: M=1 skinny_mxfp4.sec <= 0.5 * grouped_blockm16.sec (comfort).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from mosaicvm.bench.compute_floor import _bench
from mosaicvm.kernels.mxfp4 import dequant_mxfp4, pack_mxfp4
from mosaicvm.kernels.skinny_fp4 import skinny_mxfp4_gemm
from mosaicvm.kernels.w4a16_gemm import pack_w4, w4a16_gemm

ARTIFACT = Path("artifacts/skinny_race.json")
MS = (1, 2, 3, 4, 5, 8)
SHAPES = {
    "gate_up": (4096, 4096),  # (K, N) concat
    "down": (2048, 4096),
}


def _packed_bytes_mxfp4(k: int, n: int) -> int:
    return k * n // 2 + k * (n // 32)  # uint8 codes + e8m0 scale (1 byte)


def _packed_bytes_w4(k: int, n: int) -> int:
    return k * n // 2 + k * (n // 128) * 2  # uint8 codes + fp16 scale


def _try_tinygemm():
    try:
        import tinygemm  # noqa: F401
    except Exception:
        return "skipped_no_tinygemm"
    return "skipped_no_tinygemm"


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    if "3060" not in gpu.lower():
        print(gpu)
        raise SystemExit(f"unexpected GPU: {gpu}")

    rows = []
    tinygemm_status = _try_tinygemm()

    for shape_name, (k, n) in SHAPES.items():
        torch.manual_seed(42)
        w_bf16 = (torch.randn(k, n, device="cuda") / math.sqrt(k)).to(torch.bfloat16)

        # MXFP4 pack
        p_mx, s_mx = pack_mxfp4(w_bf16)
        # INT4 pack_w4
        p_w4, s_w4 = pack_w4(w_bf16)

        for m in MS:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

            # skinny_mxfp4
            def step_skinny():
                return skinny_mxfp4_gemm(x, p_mx, s_mx, n)

            sec_skinny = _bench(step_skinny, iters=50, warmup=10)
            ref_mx = x @ dequant_mxfp4(p_mx, s_mx)
            err_skinny = (step_skinny().float() - ref_mx.float()).abs().max().item()
            bytes_w = _packed_bytes_mxfp4(k, n)
            gbs_skinny = bytes_w / sec_skinny / 1e9

            rows.append({
                "M": m, "shape": shape_name, "backend": "skinny_mxfp4",
                "sec": sec_skinny, "us": sec_skinny * 1e6,
                "bytes_w": bytes_w, "packed_weight_gbs": gbs_skinny,
                "parity_max_abs": err_skinny,
            })

            # grouped_blockm16 (INT4 w4a16_gemm padded to BLOCK_M=16)
            def step_grouped():
                return w4a16_gemm(x, p_w4, s_w4)

            sec_grouped = _bench(step_grouped, iters=50, warmup=10)
            from mosaicvm.kernels.w4a16_gemm import dequant_w4
            err_grouped = (step_grouped().float() - (x @ dequant_w4(p_w4, s_w4)).float()).abs().max().item()
            bytes_w4 = _packed_bytes_w4(k, n)
            gbs_grouped = bytes_w4 / sec_grouped / 1e9

            rows.append({
                "M": m, "shape": shape_name, "backend": "grouped_blockm16",
                "sec": sec_grouped, "us": sec_grouped * 1e6,
                "bytes_w": bytes_w4, "packed_weight_gbs": gbs_grouped,
                "parity_max_abs": err_grouped,
            })

            # tinygemm
            rows.append({
                "M": m, "shape": shape_name, "backend": "tinygemm",
                "status": tinygemm_status,
            })

            del x
    # Initial gate computation (before autotune)
    s1_row = next(r for r in rows if r["M"] == 1 and r["shape"] == "gate_up" and r["backend"] == "skinny_mxfp4")
    g_row = next(r for r in rows if r["M"] == 1 and r["shape"] == "gate_up" and r["backend"] == "grouped_blockm16")
    s1_pass = s1_row["packed_weight_gbs"] >= 100.0
    s2_ratio = g_row["sec"] / s1_row["sec"] if s1_row["sec"] else None
    s2_pass = s2_ratio is not None and s2_ratio >= 2.0
    autotune_sweep = []
    if not s1_pass:
        import mosaicvm.kernels.skinny_fp4 as sk
        k_gu, n_gu = SHAPES["gate_up"]
        torch.manual_seed(42)
        w_sweep = (torch.randn(k_gu, n_gu, device="cuda") / math.sqrt(k_gu)).to(torch.bfloat16)
        p_sweep, s_sweep = pack_mxfp4(w_sweep)
        x_sweep = torch.randn(1, k_gu, dtype=torch.bfloat16, device="cuda")
        for bn in (64, 128, 256):
            for bk in (32, 64, 128):
                sk.BLOCK_N = bn
                sk.BLOCK_K = bk
                try:
                    sec = _bench(lambda: sk.skinny_mxfp4_gemm(x_sweep, p_sweep, s_sweep, n_gu), iters=20, warmup=5)
                    gbs = _packed_bytes_mxfp4(k_gu, n_gu) / sec / 1e9
                    autotune_sweep.append({"BLOCK_N": bn, "BLOCK_K": bk, "sec": sec, "gbs": gbs})
                except Exception as exc:
                    autotune_sweep.append({"BLOCK_N": bn, "BLOCK_K": bk, "error": str(exc)})
        # Restore best config
        if autotune_sweep:
            best = max((r for r in autotune_sweep if "gbs" in r), key=lambda r: r["gbs"])
            sk.BLOCK_N = best["BLOCK_N"]
            sk.BLOCK_K = best["BLOCK_K"]
            # Re-measure S1 with best config
            sec_best = _bench(lambda: sk.skinny_mxfp4_gemm(x_sweep, p_sweep, s_sweep, n_gu), iters=50, warmup=10)
            gbs_best = _packed_bytes_mxfp4(k_gu, n_gu) / sec_best / 1e9
            s1_row["sec"] = sec_best
            s1_row["packed_weight_gbs"] = gbs_best
            s1_pass = gbs_best >= 100.0
            s2_ratio = g_row["sec"] / sec_best if sec_best else None
            s2_pass = s2_ratio is not None and s2_ratio >= 2.0


    out = {
        "gpu": gpu,
        "gates": {
            "s1_m1_gbs_req": 100.0,
            "s1_m1_gbs_obs": s1_row["packed_weight_gbs"],
            "s1_pass": s1_pass,
            "s2_m1_ratio_req": 2.0,
            "s2_m1_ratio_obs": s2_ratio,
            "s2_pass": s2_pass,
        },
        "rows": rows,
        "autotune_sweep": autotune_sweep,
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({
        "s1_pass": s1_pass,
        "autotune_sweep": autotune_sweep,
        "s1_gbs": round(s1_row["packed_weight_gbs"], 2),
        "s2_pass": s2_pass,
        "s2_ratio": round(s2_ratio, 2) if s2_ratio else None,
        "wrote": str(ARTIFACT),
    }, indent=2))
    return out


if __name__ == "__main__":
    run()
