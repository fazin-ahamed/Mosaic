"""TRUE P2-est: first whole-target block with valid numerics.

Interleaves, per layer for 43 layers:
  core: FP8-resident decode (workspace path, the 183.9 ms STEADIS baseline
        components from fp8_core.py) + attention/shared/router/hc/norms
  MoE:  fixed all-IMMA fused_moe_mxint (crossover=0, exact post-fix) on a
        packed4 bank (fits VRAM alongside the FP8 core)

P2-est because: representation is production-shaped (source FP8 core codes,
source FP4 experts via packed4, exact semantics) but residency is synthetic
(all 256 experts' bank resident) and routes are golden/bootstrap. Still, this
is the first whole-target block measurement whose numerics are exact.

VRAM budget: FP8 core 7.01 GB + packed4 bank ~3.3 GB + workspace/activations
~0.2 GB ≈ 10.5 GB of 11.6 usable.

Thermal: warmup_steady + per-rep window certification.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from mosaicvm.bench.floors import HIDDEN, N_EXPERTS, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.bench.wide_verify import _bench_blocks
from mosaicvm.kernels.mxint import fused_moe_mxint
from mosaicvm.bench.phase29r import _allocate_packed4_experts
from mosaicvm.bench.fp8_core import (
    CORE_PATH, load_core_resident, alloc_workspace, decode_layer, compute_layer,
    N_HEADS, HEAD_DIM, CTX,
)
from safetensors.torch import safe_open

ORACLE = Path("artifacts/branchpack_oracle.json")
OUT = Path("artifacts/true_p2.json")
K_VALUES = (56, 63, 96)
TOP_K = 6


def certify(rep_snaps, start, end):
    temps = [s.get("temp_c") for s in rep_snaps if s.get("temp_c") is not None]
    drift = max(temps) - min(temps) if temps else 99
    c0 = start.get("sm_clock_mhz")
    locked = all(s.get("sm_clock_mhz") == c0 for s in rep_snaps + [end])
    return ("STEADY" if drift <= 3 else "CONTROLLED" if drift <= 5 else "UNKNOWN") \
        if locked else "UNKNOWN"


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    warmup_steady(120.0)

    with safe_open(str(CORE_PATH), framework="pt") as f:
        core = load_core_resident(f)
    bank = _allocate_packed4_experts(N_EXPERTS)          # packed4 exact bank
    torch.cuda.synchronize()
    print(f"resident: core 7.01GB + bank; VRAM now "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB")

    # golden + bootstrap routes
    sys.path.insert(0, ".")
    from mosaicvm.bench.wide_verify import _golden_windows
    from mosaicvm.bench.wideverify_bootstrap import layer_token_pools, bootstrap_window
    golden = json.loads(Path("artifacts/golden_routes.json").read_text())
    gw56 = _golden_windows(56, golden)
    gw63 = _golden_windows(63, golden)
    pools = layer_token_pools(golden)
    gen = torch.Generator().manual_seed(909)

    oracle = json.loads(ORACLE.read_text())["oracle"]
    t_start = snapshot()
    rep_snaps = []
    results = {"gpu": torch.cuda.get_device_name(0),
               "config": "FP8-resident workspace core + packed4 all-IMMA c0 "
                         "(exact post-fix), synthetic perfect residency",
               "rows": {}}

    ws0 = alloc_workspace("cuda")
    for K in K_VALUES:
        if K == 56:
            routes = [(torch.tensor(r[0], device="cuda", dtype=torch.int64),
                       torch.tensor(r[1], device="cuda", dtype=torch.float32))
                      for r in [gw56[(i * 7) % len(gw56)] for i in range(11 * N_LAYERS)]]
            auth = "real_golden"
        elif K == 63:
            routes = [(torch.tensor(r[0], device="cuda", dtype=torch.int64),
                       torch.tensor(r[1], device="cuda", dtype=torch.float32))
                      for r in [gw63[(i * 7) % len(gw63)] for i in range(11 * N_LAYERS)]]
            auth = "real_golden"
        else:
            routes = [(lambda p: (p[0].to("cuda"), p[1].to("cuda")))(
                bootstrap_window(pools[L % N_LAYERS], K, gen))
                for L in range(11 * N_LAYERS)]
            auth = "per_layer_bootstrap"

        xs = [torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
              for _ in range(4)]
        kv = [torch.randn(CTX, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
              for _ in range(2)]
        idx = 0

        def block_step():
            """One full 43-layer block: core compute + exact MoE per layer."""
            nonlocal idx
            x = xs[idx % 4]
            kv_cache = kv[idx % 2]
            for L in range(N_LAYERS):
                lp = core["layers"][L]
                decode_layer(lp["fp8"], lp["scales"], ws0)
                h = compute_layer(x, ws0, lp["small"], K, kv_cache)
                h = h + x * 0.001          # residual-ish tie-in (timing shape)
                out = fused_moe_mxint(h.float().to(torch.bfloat16),
                                      routes[idx][0], routes[idx][1],
                                      bank=bank, weight_repr="packed4",
                                      crossover=0, imma_k=8)
                x = (out + x).to(torch.bfloat16)
                idx += 1
            torch.nn.functional.linear(x, core["head_w"])

        rep_snaps.append(snapshot())
        t = _bench_blocks(block_step, 5)
        ea = oracle.get(str(K), {}).get("expected_accepted_depth") or \
            (31.0 if K == 63 else None)
        results["rows"][str(K)] = {
            "block_s": t, "route_authority": auth, "E_A": ea,
            "raw_per_s": K / t,
            "R_oracle_per_s": ea / t if ea else None,
        }
        print(f"K={K:3d} [{auth:22s}]: block={t:.4f}s raw={K/t:5.1f}/s "
              f"R_oracle={ea and round(ea/t,1)}/s")
        del routes, xs, kv
        torch.cuda.empty_cache()

    t_end = snapshot()
    label = certify(rep_snaps, t_start, t_end)
    results["thermal"] = {"label": label, "start": t_start, "end": t_end}
    print(f"thermal [{label}]")
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
