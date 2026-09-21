"""Production-fit wide frontier: Mixed-A banks (gu=p4, down=h8) at N63 real
golden + N64-128 bootstrap. Tests whether Exact-30 compute feasibility moves
to wider widths under the VRAM fit (banks 4.53 GB + core 7.0 = 11.53 GB).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from mosaicvm.bench.wide_verify import _bench_blocks, _golden_windows
from mosaicvm.bench.floors import HIDDEN, N_EXPERTS, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.bench.wideverify_v3 import capture_layer_tokens, win_nowrap, win_wrap
from mosaicvm.kernels.mxint import fused_moe_mxint, MXINTExpertBank, MXINTPacked4Bank
from mosaicvm.bench.phase29r import _allocate_packed4_experts
from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts

GOLDEN = Path("artifacts/golden_routes.json")
OUT = Path("artifacts/prodfit_frontier.json")
ORACLE = Path("artifacts/branchpack_oracle.json")
CORE_PROD = 0.173
KS = (56, 63, 64, 80, 96, 112, 128)


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    warmup_steady(120.0)
    golden = json.loads(GOLDEN.read_text())
    oracle = json.loads(ORACLE.read_text())["oracle"]
    pools = capture_layer_tokens(golden)
    gw63 = _golden_windows(63, golden)
    gw56 = _golden_windows(56, golden)
    gen = torch.Generator().manual_seed(5150)

    # Mixed-A banks (gu=p4 full, down=h8 full): VRAM-fit partial objects
    h8_full = _allocate_mxint_experts(N_EXPERTS)
    h8_down = MXINTExpertBank(
        w_gu=h8_full.w_gu[:1].clone(), s_gu=h8_full.s_gu[:1].clone(),
        w_d=h8_full.w_d, s_d=h8_full.s_d,
        n_out_gu=h8_full.n_out_gu, n_out_d=h8_full.n_out_d)
    del h8_full
    p4_full = _allocate_packed4_experts(N_EXPERTS)
    p4_gu = MXINTPacked4Bank(
        p_gu=p4_full.p_gu, s_gu=p4_full.s_gu,
        p_d=p4_full.p_d[:1].clone(), s_d=p4_full.s_d[:1].clone(),
        n_out_gu=p4_full.n_out_gu, n_out_d=p4_full.n_out_d)
    del p4_full
    torch.cuda.empty_cache()
    print(f"banks: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    t0 = snapshot()
    rows = {}
    for K in KS:
        if K in (56, 63):
            src = gw56 if K == 56 else gw63
            routes = [(torch.tensor(r[0]), torch.tensor(r[1], dtype=torch.float32))
                      for r in [src[(i * 7) % len(src)] for i in range(11 * N_LAYERS)]]
            auth = "real_golden"
        else:
            routes = [win_wrap(pools[(i * 5) % len(pools)][2], K, gen)
                      for i in range(11 * N_LAYERS)]
            auth = "corpus_wrap"
        xs = [torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
        ids_t = [r[0].to("cuda") for r in routes]
        w_t = [r[1].to("cuda") for r in routes]
        idx = 0

        def step():
            nonlocal idx
            for _ in range(N_LAYERS):
                fused_moe_mxint(xs[idx % 4], ids_t[idx], w_t[idx],
                                gu_bank=p4_gu, down_bank=h8_down, crossover=0)
                idx += 1

        t = _bench_blocks(step, 10)
        ea = oracle.get(str(K), {}).get("expected_accepted_depth") or \
            (31.0 if K == 63 else None)
        rows[str(K)] = {"moe_s": t, "auth": auth, "E_A": ea,
                        "R_oracle_prodcore": ea / (t + CORE_PROD) if ea else None,
                        "raw_per_s": K / (t + CORE_PROD)}
        print(f"K={K:3d} [{auth:12s}]: moe={t:.4f}s R_oracle="
              f"{ea and round(ea/(t+CORE_PROD),1)}/s raw={K/(t+CORE_PROD):.1f}/s",
              flush=True)
        del routes, xs, ids_t, w_t
        torch.cuda.empty_cache()

    t1 = snapshot()
    OUT.write_text(json.dumps({"banks": "Mixed-A (gu=p4, down=h8) 4.53GB",
                               "rows": rows, "thermal": {"start": t0, "end": t1}},
                              indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
