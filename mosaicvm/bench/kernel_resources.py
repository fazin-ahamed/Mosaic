"""Driver occupancy + resource introspection for KSplit variants."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mosaicvm.kernels.fp4_source import pack_mxfp4_nk
from mosaicvm.kernels.ptx_jit import function_attributes, max_active_blocks
from mosaicvm.kernels.qpa_mma import _FN_CACHE, _get_fn, _ksplit_direct_src, _ksplit_grouped_src, _QPA_SRC, qpa_mma_gemm
from mosaicvm.kernels.xsimt import _ksplit_xsimt_src, decode_activation_exact, xsimt_gemm

ARTIFACT = Path("artifacts/kernel_resources.json")
SM86_MAX_WARPS = 48


def _occupancy(fn, tpb: int, smem: int = 0) -> dict:
    attrs = function_attributes(fn)
    blocks = max_active_blocks(fn, tpb, smem)
    occ_driver = blocks * (tpb // 32) / SM86_MAX_WARPS
    # occupancy_model: regs-limited estimate (65536 regs/SM on SM86)
    regs = attrs.get("num_regs", 0)
    warps_reg = (65536 // max(regs, 1)) // 32
    occ_model = min(1.0, warps_reg / SM86_MAX_WARPS)
    return {
        **{k.lower() if isinstance(k, str) else k: v for k, v in attrs.items()},
        "num_regs": attrs.get("num_regs", attrs.get("NUM_REGS".lower())),
        "local_memory_used": attrs.get("local_size_bytes", 0),
        "occupancy_driver": occ_driver,
        "occupancy_model": occ_model,
        "max_active_blocks": blocks,
        "tpb": tpb,
        "smem": smem,
    }


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    # force-compile variants via a tiny GEMM
    torch.manual_seed(0)
    n, k, m = 16, 128, 2
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    p, s = pack_mxfp4_nk(w)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    qpa_mma_gemm(x, p, s, n, qpa_k=1)
    for w_k in (2, 4, 8):
        qpa_mma_gemm(x, p, s, n, qpa_k=w_k, ordered=True)
        qpa_mma_gemm(x, p, s, n, qpa_k=w_k, ordered=False)
    xsimt_gemm(x, p, s, n)

    variants = {}
    # single-warp reference
    if "qpa_gemm_pre" in _FN_CACHE:
        variants["qpa_gemm_pre"] = _occupancy(_FN_CACHE["qpa_gemm_pre"], 32, 0)
    for w_k in (2, 4, 8):
        for ordered in (True, False):
            entry = f"qpa_ksplit_direct_w{w_k}" if ordered else f"qpa_ksplit_direct_reordered_w{w_k}"
            src = _ksplit_direct_src(w_k, ordered)
            fn = _get_fn(entry, src, entry)
            variants[entry] = _occupancy(fn, 32 * w_k, w_k * 32 * 4 * 4)
        for kind in ("gu", "down"):
            entry = f"qpa_{kind}_ksplit_ordered_w{w_k}"
            src = _ksplit_grouped_src(kind, w_k, True)
            fn = _get_fn(entry, src, entry)
            variants[entry] = _occupancy(fn, 32 * w_k, w_k * 32 * 4 * 4)
    for w_k in (4, 8):
        for kind in ("gu", "down"):
            entry = f"xsimt_{kind}_ksplit_ordered_w{w_k}"
            src = _ksplit_xsimt_src(kind, w_k, True)
            from mosaicvm.kernels.xsimt import _get_fn as xget
            fn = xget(entry, src, entry)
            variants[entry] = _occupancy(fn, 32 * w_k, w_k * 32 * 4)

    out = {"gpu": torch.cuda.get_device_name(0), "variants": variants}
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"n_variants": len(variants), "wrote": str(ARTIFACT)}, indent=2))
    return out


if __name__ == "__main__":
    run()
