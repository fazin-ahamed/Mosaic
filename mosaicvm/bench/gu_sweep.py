"""Track D: GU parameter race at the Exact-30 widths (N16-32).

Sweeps fused_moe_mxint (imma_k x crossover) at N in {16, 20, 24, 28,
32, 56, 112}, reporting gu_sec/down_sec stage times per config. The
best config must later be validated as a full TRUE-P2 delta.
"""
from __future__ import annotations

import json
import sys
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

N_VALUES = (16, 20, 24, 28, 32, 56, 112)
CONFIGS = [(8, 0), (4, 0), (16, 0), (8, 4), (8, 16), (8, 32)]
OUT = Path("artifacts/gu_sweep.json")
GOLDEN = Path("artifacts/golden_routes.json")
BANK_E = 128
STAGES = ("gu_sec", "down_sec", "layout_sec", "act_quant_sec",
          "decode_int_sec", "swiglu_sec", "down_quant_sec",
          "reduction_sec")


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
        x = torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for (ik, xo) in CONFIGS:
            try:
                for i in range(6):  # warm
                    fused_moe_mxint(x, routes[i][0] % BANK_E,
                                    routes[i][1], bank=bank,
                                    weight_repr="packed4", crossover=xo,
                                    imma_k=ik, swiglu_limit=10.0)
                torch.cuda.synchronize()
                st = {}
                reps = 60
                for i in range(reps):
                    fused_moe_mxint(x, routes[i % len(routes)][0] % BANK_E,
                                    routes[i % len(routes)][1], bank=bank,
                                    weight_repr="packed4", crossover=xo,
                                    imma_k=ik, swiglu_limit=10.0,
                                    stage_times=st)
                torch.cuda.synchronize()
                tot = st["gu_sec"] + st["down_sec"]
                rows.setdefault(str(K), {})[f"imma{ik}_xo{xo}"] = {
                    "gu_ms": round(st["gu_sec"] * 1e3, 3),
                    "down_ms": round(st["down_sec"] * 1e3, 3),
                    "total_ms": round(tot * 1e3, 3),
                }
                print(f"N={K:3d} imma{ik}/xo{xo}: "
                      f"GU={st['gu_sec']*1e3:6.2f}ms "
                      f"DOWN={st['down_sec']*1e3:6.2f}ms", flush=True)
            except Exception as e:
                rows.setdefault(str(K), {})[f"imma{ik}_xo{xo}"] = \
                    {"error": str(e)[:120]}
                print(f"N={K:3d} imma{ik}/xo{xo}: ERROR {e}"[:100])
        del routes, x
        torch.cuda.empty_cache()
    # best per N
    best = {}
    for k, cfgs in rows.items():
        ok = {c: v["total_ms"] for c, v in cfgs.items()
              if "total_ms" in v}
        if ok:
            b = min(ok, key=ok.get)
            best[k] = {"best": b, "total_ms": ok[b]}
    OUT.write_text(json.dumps({"classification": "GU_PARAM_RACE_N16_32",
                               "configs": rows, "best": best}, indent=1))
    print("\nbest per N:", json.dumps(best, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
