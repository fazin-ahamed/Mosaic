"""vLLM Marlin MoE wrapper. Skip unless our W4 can be documented-repacked."""
from __future__ import annotations

MARLIN_STATUS = {
    "status": "blocked_layout",
    "reason": (
        "vLLM fused_marlin_moe / gptq_marlin_moe_repack documents GPTQ packing "
        "along K (w13 qweight [E, K/pack, N], pack_factor=8 for 4-bit) plus "
        "marlin_moe_permute_scales. MosaicVM pack_w4 stores two consecutive "
        "N-values per uint8 (low nibble first) with per-group scale amax/7 "
        "along N. No documented conversion from that layout to GPTQ-Marlin."
    ),
    "probed": [
        "vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe",
        "vllm.model_executor.layers.fused_moe.experts.marlin_moe.MarlinExperts",
        "vllm.model_executor.layers.fused_moe.experts.marlin_moe.BatchedMarlinExperts",
        "vllm.model_executor.layers.fused_moe.oracle.int_wna16._process_weights_marlin",
    ],
}


def marlin_moe(x, pool, expert_ids, expert_weights, top_k=6):
    raise RuntimeError(MARLIN_STATUS["reason"])
