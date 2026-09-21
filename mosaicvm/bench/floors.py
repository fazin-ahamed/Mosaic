"""Named compute floors and the 30 tok/s equation used by every Phase 2.5 JSON."""
from __future__ import annotations

N_LAYERS = 43
HIDDEN = 4096
MOE_INTER = 2048
N_EXPERTS = 256
TOP_K = 6  # exact; do not read Router.active_k / aed / aec
K_VALUES = (1, 2, 3, 4, 5, 6, 8, 12, 16)
TARGET_TOKS = 30.0

# Floor tags written into every artifact
FLOOR_A = "same_expert_gemm_roofline"       # current compute_floor_3060.json
FLOOR_B = "real_distinct_expert_moe"        # one layer, 256 experts resident
FLOOR_C = "full_layer_ceiling"              # attn+router+shared+moe+hc
FLOOR_D = "full_model_perfect_residency"    # 43 * Floor C + embed/lm_head

# Phase 2.6 Exact-30 MoE budget (from Floor D K=8 mid, artifacts/floor_cd.json).
# t_block = 1.4944318413704636 s, amdahl_if_moe_inf_toks = 91.23195504603373.
NON_MOE_K8_S = 8.0 / 91.23195504603373                     # 0.087689 s
MOE_BUDGET_30_S = 8.0 / 30.0 - NON_MOE_K8_S                 # 0.178978 s total MoE
MOE_LAYER_BUDGET_30_S = MOE_BUDGET_30_S / N_LAYERS          # 0.004162 s = 4.16 ms
MOE_LAYER_BUDGET_40_S = (8.0 / 40.0 - NON_MOE_K8_S) / N_LAYERS  # 0.002612 s = 2.61 ms
GBS_GATE = 100.0


def candidate_toks(blocks_per_sec: float, k: int) -> float:
    return k * blocks_per_sec  # oracle: E[A_K] = K


def required_t_block_s(e_ak: float, t_draft: float = 0.0, t_runtime: float = 0.0) -> float:
    # 30 tok/s => T_draft + T_verify + T_runtime < E[A_K]/30
    return e_ak / TARGET_TOKS - t_draft - t_runtime
