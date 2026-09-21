"""FP8 materialized baseline: dequant_fp8 + bf16 GEMM on [4096,512], M=1-8.

Not Exact-30 Floor D. Reuses gpu_fp8_core_plus_lm_head_plus_router_gb ≈ 5.86.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mosaicvm.bench.compute_floor import _bench
from mosaicvm.runtime.fp8 import dequant_fp8

ARTIFACT = Path("artifacts/fp8_materialized_baseline.json")
MS = (1, 2, 3, 4, 5, 8)
SHAPE = (4096, 512)  # out, in  — attn.kv_proj-like


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    n, k = SHAPE
    w_fp8 = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.ones(n // 128 or 1, k // 128 or 1, device="cuda", dtype=torch.float32)
    if n % 128 != 0 or k % 128 != 0:
        scale = torch.ones(1, 1, device="cuda", dtype=torch.float32)
    w_bf16 = dequant_fp8(w_fp8, scale)
    rows = []
    for m in MS:
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

        def step():
            return x @ w_bf16.T

        sec = _bench(step, iters=50, warmup=10)
        rows.append({"M": m, "N": n, "K": k, "sec": sec, "us": sec * 1e6})
    inv = {}
    p = Path("artifacts/core_inventory.json")
    if p.exists():
        inv = json.loads(p.read_text())
    out = {
        "gpu": torch.cuda.get_device_name(0),
        "shape": list(SHAPE),
        "gpu_fp8_core_plus_lm_head_plus_router_gb": inv.get("gpu_fp8_core_plus_lm_head_plus_router_gb", 5.86),
        "rows": rows,
        "note": "not Exact-30 Floor D",
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("shape", "gpu_fp8_core_plus_lm_head_plus_router_gb")}, indent=2))
    print("rows", [{r["M"]: round(r["us"], 1)} for r in rows])
    return out


if __name__ == "__main__":
    run()
