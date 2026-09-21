"""Track D: split fused_moe dispatch into GPU work vs launch/sync.

Per fused_moe_mxint call at N16-32: wall-clock per call (CPU timer)
vs sum of CUDA-event stage times. The gap = launch/sync overhead
(the only part CUDA graphs can remove). Also records the per-stage
GPU work (layout/act_quant/decode_int) that graphs cannot remove.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")

from mosaicvm.bench.floors import HIDDEN, N_LAYERS
from mosaicvm.bench.thermal import warmup_steady
from mosaicvm.bench.phase29r import _allocate_packed4_experts
from mosaicvm.kernels.mxint import fused_moe_mxint
from mosaicvm.bench.wide_verify import _golden_windows
from mosaicvm.bench.wideverify_bootstrap import (
    layer_token_pools, bootstrap_window)

N_VALUES = (16, 20, 24, 28, 32, 56, 80, 112)
OUT = Path("artifacts/dispatch_decomposition.json")
GOLDEN = Path("artifacts/golden_routes.json")
BANK_E = 128
STAGES = ("layout_sec", "act_quant_sec", "decode_int_sec", "gu_sec",
          "swiglu_sec", "down_quant_sec", "down_sec", "reduction_sec")


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    warmup_steady(90.0)
    golden = json.loads(GOLDEN.read_text())
    bank = _allocate_packed4_experts(BANK_E)
    torch.cuda.empty_cache()
    pools = layer_token_pools(golden)
    gen_r = torch.Generator().manual_seed(909)
    rows = {}
    for K in N_VALUES:
        wins = _golden_windows(K, golden)
        if len(wins) >= 8 * N_LAYERS:
            step_ = max(1, len(wins) // (8 * N_LAYERS))
            routes = [
                (torch.tensor(w[0], device="cuda", dtype=torch.int64),
                 torch.tensor(w[1], device="cuda", dtype=torch.float32))
                for w in wins[::step_][:8 * N_LAYERS]]
        else:
            routes = [
                (lambda p: (p[0].to("cuda"), p[1].to("cuda")))(
                    bootstrap_window(pools[L % N_LAYERS], K, gen_r))
                for L in range(8 * N_LAYERS)]
        xs = [torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
              for _ in range(4)]
        # warm
        for i in range(8):
            fused_moe_mxint(xs[i % 4], routes[i][0] % BANK_E,
                            routes[i][1], bank=bank,
                            weight_repr="packed4", crossover=0, imma_k=8,
                            swiglu_limit=10.0)
        torch.cuda.synchronize()
        # timed loop: CPU wall per call + accumulated stage events
        n_iter = 200
        st = {}
        t0 = time.perf_counter()
        for i in range(n_iter):
            fused_moe_mxint(
                xs[i % 4], routes[i % len(routes)][0] % BANK_E,
                routes[i % len(routes)][1], bank=bank,
                weight_repr="packed4", crossover=0, imma_k=8,
                swiglu_limit=10.0, stage_times=st)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / n_iter
        gpu_work = sum(st[s] for s in STAGES)
        prep = st["layout_sec"] + st["act_quant_sec"] + st["decode_int_sec"]
        launch = wall - gpu_work
        rows[str(K)] = {
            "wall_ms": round(wall * 1e3, 3),
            "gpu_stage_sum_ms": round(gpu_work * 1e3, 3),
            "prep_gpu_ms_layout_quant_decode": round(prep * 1e3, 3),
            "gemm_gu_down_ms": round(
                (st["gu_sec"] + st["down_sec"]) * 1e3, 3),
            "launch_sync_gap_ms": round(launch * 1e3, 3),
            "launch_share_pct": round(100 * launch / wall, 1),
        }
        print(f"N={K:3d}: wall={wall*1e3:6.2f}ms stages={gpu_work*1e3:6.2f} "
              f"launch_gap={launch*1e3:5.2f}ms "
              f"({100*launch/wall:.0f}%)")
        del routes, xs
        torch.cuda.empty_cache()
    out = {
        "classification": "DISPATCH_LAUNCH_VS_GPU_WORK_SPLIT",
        "note": "launch_sync_gap = wall - sum(cuda-event stages) = the "
                "only CUDA-graph-removable part; layout/quant/decode "
                "are real GPU work",
        "per_call": rows,
        "per_block_x43": {k: {kk: (round(vv * 43, 1) if isinstance(vv, (int, float)) else vv)
                              for kk, vv in v.items()} for k, v in rows.items()},
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
