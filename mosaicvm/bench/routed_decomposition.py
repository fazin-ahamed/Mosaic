"""Track D (roadmap §53-54): routed-expert intercept decomposition.

Runs the SAME routed-expert workload as the TRUE P2 sweep block
(43 layers x fused_moe_mxint on golden routes, packed4 bank, swiglu 10)
with the kernel's stage_times instrumentation, splitting the routed
intercept into:

  layout      token->expert-slot assignment / gather
  act_quant   E4M3 per-128 activation quant
  decode_int  activation int-code decode
  gu          GU GEMM (w1+w3, [M*6,4096]->[M*6,2048] x2)
  swiglu      SwiGLU + clamp
  down_quant  w2-input re-quant (E4M3 per-128)
  down        DOWN GEMM (w2, [M*6,2048]->[M*6,4096])
  reduction   route-weight + expert scatter/reduction

Classification: TRUE_P2_ROUTED_DECOMPOSITION.
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

N_VALUES = (1, 2, 4, 8, 16, 32, 56, 80, 112)
OUT = Path("artifacts/routed_expert_decomposition.json")
GOLDEN = Path("artifacts/golden_routes.json")
BANK_E = 128
STAGES = ("layout_sec", "act_quant_sec", "decode_int_sec", "gu_sec",
          "swiglu_sec", "down_quant_sec", "down_sec", "reduction_sec")


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    warmup_steady(120.0)
    golden = json.loads(GOLDEN.read_text())

    bank = _allocate_packed4_experts(BANK_E)
    torch.cuda.empty_cache()
    pools = layer_token_pools(golden)
    gen_r = torch.Generator().manual_seed(909)

    results = {
        "gpu": torch.cuda.get_device_name(0),
        "classification": "TRUE_P2_ROUTED_DECOMPOSITION",
        "setup": "43 layers x fused_moe_mxint per block, golden routes "
                 "remapped mod 128, packed4 bank, swiglu_limit=10 — "
                 "identical to the TRUE P2 sweep routed workload",
        "stages": list(STAGES),
        "rows": {},
    }

    for K in N_VALUES:
        wins = _golden_windows(K, golden)
        if len(wins) >= 16 * N_LAYERS:
            step_ = max(1, len(wins) // (16 * N_LAYERS))
            routes = [
                (torch.tensor(w[0], device="cuda", dtype=torch.int64),
                 torch.tensor(w[1], device="cuda", dtype=torch.float32))
                for w in wins[::step_][:16 * N_LAYERS]]
        else:
            routes = [
                (lambda p: (p[0].to("cuda"), p[1].to("cuda")))(
                    bootstrap_window(pools[L % N_LAYERS], K, gen_r))
                for L in range(16 * N_LAYERS)]

        # accumulate stage times across 43 layers x reps
        acc = {s: 0.0 for s in STAGES}
        reps = 6
        # warm
        for i in range(N_LAYERS):
            fused_moe_mxint(
                torch.randn(K, HIDDEN, dtype=torch.bfloat16,
                            device="cuda"),
                routes[i][0] % BANK_E, routes[i][1], bank=bank,
                weight_repr="packed4", crossover=0, imma_k=8,
                swiglu_limit=10.0)
        torch.cuda.synchronize()
        from mosaicvm.bench.p2_exact_dsa import time_layer

        state = {"i": 0}

        def block():
            i = state["i"]
            st = {}
            for L in range(N_LAYERS):
                x = torch.randn(K, HIDDEN, dtype=torch.bfloat16,
                                device="cuda")
                fused_moe_mxint(
                    x, routes[(i * N_LAYERS + L) % len(routes)][0] % BANK_E,
                    routes[(i * N_LAYERS + L) % len(routes)][1],
                    bank=bank, weight_repr="packed4", crossover=0,
                    imma_k=8, swiglu_limit=10.0, stage_times=st)
                for s in STAGES:
                    acc[s] += st[s]
            state["i"] += 1

        t_total = time_layer(block, reps=reps, warm=1)
        n_calls = reps * N_LAYERS
        row = {s: round(acc[s] / n_calls, 6) for s in STAGES}
        row["sum_stages"] = round(sum(acc[s] / n_calls for s in STAGES), 6)
        row["t_block_measured"] = round(t_total, 4)
        results["rows"][str(K)] = row
        gu = row["gu_sec"] * 1e3
        dn = row["down_sec"] * 1e3
        ep = (row["swiglu_sec"] + row["down_quant_sec"]
              + row["reduction_sec"]) * 1e3
        disp = (row["layout_sec"] + row["act_quant_sec"]
                + row["decode_int_sec"]) * 1e3
        print(f"N={K:3d}: GU={gu:6.2f}ms DOWN={dn:6.2f}ms "
              f"epilogue={ep:6.2f}ms dispatch={disp:6.2f}ms "
              f"sum={row['sum_stages']*1e3:7.2f}ms "
              f"block={t_total*1e3:7.1f}ms")
        del routes
        torch.cuda.empty_cache()

    # per-call at N=1 -> intercept-style view (43 calls/block)
    r1 = results["rows"]["1"]
    per_block = {s: round(r1[s] * 43, 4) for s in STAGES}
    results["intercept_view_N1_x43"] = per_block
    print("\nintercept view (N=1 x 43 layers, ms):")
    for s in STAGES:
        print(f"  {s:16s} {per_block[s]*1e3:8.2f}")

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
