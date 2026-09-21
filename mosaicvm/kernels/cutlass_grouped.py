"""CUTLASS grouped GEMM wrapper. No runnable Python grouped GEMM on this wheel."""
from __future__ import annotations

CUTLASS_STATUS = {
    "status": "blocked_no_cutlass",
    "reason": (
        "nvidia-cutlass 4.2.0.0 installs a DSL plus cutlass_library metadata "
        "(EmitGemmGroupedInstance). No callable grouped GEMM / GatherA / INT4 "
        "epilogue. Plan forbids vendoring the CUTLASS C++ tree or writing a "
        "raw CUDA GEMM scheduler."
    ),
}


def cutlass_moe(x, pool, expert_ids, expert_weights):
    raise RuntimeError(CUTLASS_STATUS["reason"])
