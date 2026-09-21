"""NEXT 3: Hot8 allocation sweep — T_MoE vs bank budget Pareto frontier.

Masked kernel (validated 8/8 bitwise) driven through fused_moe_mxint on real
K56 golden routes. Frequency-ranked hot selection from golden expert counts.
Bank budget: full packed4 3.22 GB + 16.78MB*N_gu + 8.39MB*N_dn <= 4.55 GB
(core 7.0 + banks <= 11.55 GB nominal).
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
from mosaicvm.kernels.mxint import (
    MXINTExpertBank, MXINTMaskedBank, MXINTPacked4Bank, fused_moe_mxint,
)

GOLDEN = Path("artifacts/golden_routes.json")
OUT = Path("artifacts/mask_sweep.json")
CORE_PROD = 0.173
BUDGET_MB = 1440.0            # hot8 extra beyond full packed4
GRID = [  # (N_gu_hot, N_down_hot) — gu-heavy bias per 2.10 vs 1.75 ms/GB
    (0, 0),      # pure packed4 sanity (~0.9545)
    (85, 0),     # max gu-heavy
    (68, 34),
    (60, 51),
    (51, 69),
    (34, 103),
    (0, 171),    # max down-heavy
]


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    warmup_steady(120.0)
    golden = json.loads(GOLDEN.read_text())
    gw = _golden_windows(56, golden)

    counts = {}
    for c in golden["captures"]:
        for layer in c["layers"]:
            ids = layer["expert_ids"]
            flat = ids if not isinstance(ids[0], list) else [e for t in ids for e in t]
            for e in flat:
                counts[e] = counts.get(e, 0) + 1
    freq = sorted(counts, key=lambda e: -counts[e])

    # Build compact hot8 banks FIRST (full hot8 transiently resident)
    from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts
    full_h8 = _allocate_mxint_experts(N_EXPERTS)
    max_g, max_d = max(g for g, _ in GRID), max(d for _, d in GRID)
    freq_sorted = torch.tensor(freq, device="cuda", dtype=torch.long)
    h8_gu_rows = freq_sorted[:max_g]
    h8_dn_rows = freq_sorted[:max_d]
    h8_gu = MXINTExpertBank(
        w_gu=full_h8.w_gu[h8_gu_rows].contiguous(),
        s_gu=full_h8.s_gu[h8_gu_rows].contiguous(),
        w_d=full_h8.w_d[h8_dn_rows].contiguous(),
        s_d=full_h8.s_d[h8_dn_rows].contiguous(),
        n_out_gu=full_h8.n_out_gu, n_out_d=full_h8.n_out_d)
    # NOTE: gu/down compact banks share the same frequency-rank prefix, so
    # h8_gu.w_gu rows [0..max_g) correspond to freq-rank order; similarly w_d.
    del full_h8
    torch.cuda.empty_cache()

    from mosaicvm.bench.phase29r import _allocate_packed4_experts
    p4 = _allocate_packed4_experts(N_EXPERTS)
    gb0 = torch.cuda.memory_allocated() / 1e9
    print(f"resident after banks: {gb0:.2f} GB (hot8 compacts + full p4)")

    routes = [gw[(i * 7) % len(gw)] for i in range(11 * N_LAYERS)]
    xs = [torch.randn(56, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    ids_t = [torch.tensor(r[0], device="cuda", dtype=torch.int64) for r in routes]
    w_t = [torch.tensor(r[1], device="cuda", dtype=torch.float32) for r in routes]

    results = {"grid": [], "budget_mb": BUDGET_MB}
    t0 = snapshot()
    for n_gu, n_dn in GRID:
        gu_ids = torch.full((N_EXPERTS,), -1, device="cuda", dtype=torch.int32)
        dn_ids = torch.full((N_EXPERTS,), -1, device="cuda", dtype=torch.int32)
        for i in range(n_gu):
            gu_ids[freq[i]] = i
        for i in range(n_dn):
            dn_ids[freq[i]] = i
        mb = n_gu * 16.78 + n_dn * 8.39
        bank = MXINTMaskedBank(
            p4=p4,
            h8_gu_w=h8_gu.w_gu[:n_gu] if n_gu else h8_gu.w_gu[:1],
            h8_gu_s=h8_gu.s_gu[:n_gu] if n_gu else h8_gu.s_gu[:1],
            h8_down_w=h8_gu.w_d[:n_dn] if n_dn else h8_gu.w_d[:1],
            h8_down_s=h8_gu.s_d[:n_dn] if n_dn else h8_gu.s_d[:1],
            gu_hot_ids=gu_ids, down_hot_ids=dn_ids)

        idx = 0

        def step():
            nonlocal idx
            for _ in range(N_LAYERS):
                fused_moe_mxint(xs[idx % 4], ids_t[idx], w_t[idx],
                                masked_bank=bank, crossover=0)
                idx += 1

        try:
            t = _bench_blocks(step, 10)
            row = {"n_gu": n_gu, "n_dn": n_dn, "hot8_mb": mb,
                   "moe_block_s": t,
                   "R_oracle_prodcore": 28.27 / (t + CORE_PROD)}
            results["grid"].append(row)
            print(f"gu={n_gu:3d} dn={n_dn:3d} ({mb:6.0f}MB): {t:.4f} s  "
                  f"R_oracle={row['R_oracle_prodcore']:.1f}/s", flush=True)
        except Exception as exc:
            print(f"gu={n_gu} dn={n_dn}: ERROR {exc}"[:120])
            results["grid"].append({"n_gu": n_gu, "n_dn": n_dn,
                                    "error": str(exc)[:150]})
    t1 = snapshot()
    results["thermal"] = {"start": t0, "end": t1}
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
