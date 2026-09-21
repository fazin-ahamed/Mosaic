"""NEXT 1: hybrid-bank sweep — production-fit K56 MoE timing.

VRAM budget: core 7.0 GB (FP8-resident) + packed4 full bank 3.3 GB +
partial hot8 bank 6.5xF <= 11.6 GB  =>  F <= ~0.20.

Method (existing kernels, no surgery): per layer-window, split assignments
into hot (expert in top-F golden-frequency set; ids remapped to a compact
partial hot8 bank) and cold (full packed4 bank); two fused_moe_mxint calls
+ output sum. The F=0 point measures the pure split-call overhead (two
packed4 calls summed) against the single-call packed4 baseline.

hot8 partial banks are built by qint-prepacking ONLY the hot experts
(golden-frequency ranked). Exact arithmetic throughout (post-fix kernels).
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
    fused_moe_mxint, prepack_expert_bank_mxint_packed4,
    MXINTPacked4Bank, qint_encode_w, MXINTExpertBank,
)
from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts
from mosaicvm.bench.phase29r import _allocate_packed4_experts

GOLDEN = Path("artifacts/golden_routes.json")
OUT = Path("artifacts/hybrid_bank_sweep.json")
CORE_GB, BUDGET_GB = 7.0, 11.6


def hot8_partial_from_full(full: MXINTExpertBank, hot_ids) -> MXINTExpertBank:
    """Compact partial hot8 bank holding only hot experts (tensors sliced)."""
    idx = torch.tensor(sorted(hot_ids), device=full.w_gu.device, dtype=torch.long)
    return MXINTExpertBank(
        w_gu=full.w_gu[idx].contiguous(), s_gu=full.s_gu[idx].contiguous(),
        w_d=full.w_d[idx].contiguous(), s_d=full.s_d[idx].contiguous(),
        n_out_gu=full.n_out_gu, n_out_d=full.n_out_d,
    )

def split_window(ids, wts, hot_id_set, hot_pos):
    """ids [K,6] -> per-assignment rows: (tok_idx, ids[N,1], wts[N,1])."""
    flat_ids = ids.reshape(-1)
    flat_w = wts.reshape(-1).float()
    tok = torch.arange(flat_ids.numel()) // ids.shape[1]
    is_hot = torch.tensor([int(e) in hot_id_set for e in flat_ids.tolist()])
    hid = torch.tensor([hot_pos[int(e)] for e in flat_ids[is_hot].tolist()],
                       dtype=torch.int64).reshape(-1, 1)
    hw = flat_w[is_hot].reshape(-1, 1)
    cid = flat_ids[~is_hot].reshape(-1, 1)
    cw = flat_w[~is_hot].reshape(-1, 1)
    return tok[is_hot], hid, hw, tok[~is_hot], cid, cw


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    warmup_steady(120.0)
    golden = json.loads(GOLDEN.read_text())

    # golden expert frequencies -> hot sets
    counts = {}
    for c in golden["captures"]:
        for layer in c["layers"]:
            ids = layer["expert_ids"]
            flat = ids if not isinstance(ids[0], list) else [e for t in ids for e in t]
            for e in flat:
                counts[e] = counts.get(e, 0) + 1
    freq = sorted(counts, key=lambda e: -counts[e])

    # Pre-slice ALL partial hot8 banks while only the full hot8 is resident
    # (avoids full-hot8 + packed4 + slice co-residency spikes).
    full_h8 = _allocate_mxint_experts(N_EXPERTS)
    partials = {}
    for F in (0.05, 0.10, 0.15, 0.20):
        n_hot = max(1, int(F * N_EXPERTS))
        partials[F] = hot8_partial_from_full(full_h8, set(freq[:n_hot]))
        torch.cuda.empty_cache()
    del full_h8
    torch.cuda.empty_cache()

    p4 = _allocate_packed4_experts(N_EXPERTS)          # 3.3 GB full
    torch.cuda.synchronize()
    base_gb = torch.cuda.memory_allocated() / 1e9
    print(f"partials {sum(t.w_gu.numel() for t in partials.values())/1e9:.2f} GB "
          f"+ packed4 resident: {base_gb:.2f} GB")

    gw = _golden_windows(56, golden)
    results = {"budget_model": {"core_gb": CORE_GB, "packed4_gb": 3.3,
                                "hot8_per_expert_mb": 25.6, "budget_gb": BUDGET_GB},
               "rows": {}}

    for F in (0.0, 0.05, 0.10, 0.15, 0.20):
        n_hot = 0 if F == 0 else max(1, int(F * N_EXPERTS))
        hot_ids = set(freq[:n_hot])
        hot_pos = {e: i for i, e in enumerate(sorted(hot_ids))}

        bank_hot = partials.pop(F, None)
        gb = torch.cuda.memory_allocated() / 1e9
        fits = (CORE_GB + gb) <= BUDGET_GB
        print(f"\nF={F:.2f} ({n_hot} hot): banks {gb:.2f} GB + core {CORE_GB} = "
              f"{CORE_GB+gb:.2f} GB {'FITS' if fits else 'OVER BUDGET'}")

        # pre-split routes (per-assignment rows + token indices)
        rt = []
        for i in range(11 * N_LAYERS):
            ids, wts, _ = gw[(i * 7) % len(gw)]
            ids_t = torch.tensor(ids) if not isinstance(ids[0], list) else torch.tensor(ids)
            wts_t = torch.tensor(wts, dtype=torch.float32)
            ht, hi, hw, ct, ci, cw = split_window(ids_t, wts_t, hot_ids, hot_pos)
            rt.append(tuple(t.to("cuda") for t in (ht, hi, hw, ct, ci, cw)))

        xs = [torch.randn(56, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
        idx = 0

        def step():
            nonlocal idx
            for _ in range(N_LAYERS):
                x = xs[idx % 4]
                ht, hi, hw, ct, ci, cw = rt[idx]
                out = torch.zeros(56, HIDDEN, dtype=torch.bfloat16, device="cuda")
                if hi.numel():
                    oh = fused_moe_mxint(x[ht], hi, hw, bank=bank_hot,
                                         weight_repr="hot8", crossover=0)
                    out = out.index_add(0, ht, oh.to(torch.bfloat16))
                if ci.numel():
                    oc = fused_moe_mxint(x[ct], ci, cw, bank=p4,
                                         weight_repr="packed4", crossover=0)
                    out = out.index_add(0, ct, oc.to(torch.bfloat16))
                idx += 1

        t = _bench_blocks(step, 8)
        row = {"F": F, "n_hot": n_hot, "banks_gb": gb, "fits_budget": fits,
               "moe_block_s": t,
               "R_oracle_prodcore": 28.27 / (t + 0.173)}
        results["rows"][f"F{F}"] = row
        print(f"  K56 golden split-call moe: {t:.4f} s  "
              f"R_oracle={row['R_oracle_prodcore']:.1f}/s (prodcore)")
        del rt, xs, bank_hot
        torch.cuda.empty_cache()

    # single-call packed4 baseline for overhead attribution
    routes = [gw[(i * 7) % len(gw)] for i in range(9 * N_LAYERS)]
    xs = [torch.randn(56, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    ids_t = [torch.tensor(r[0], device="cuda", dtype=torch.int64) for r in routes]
    w_t = [torch.tensor(r[1], device="cuda", dtype=torch.float32) for r in routes]
    idx = 0

    def step1():
        nonlocal idx
        for _ in range(N_LAYERS):
            fused_moe_mxint(xs[idx % 4], ids_t[idx], w_t[idx], bank=p4,
                            weight_repr="packed4", crossover=0)
            idx += 1

    t_single = _bench_blocks(step1, 8)
    results["packed4_single_call_baseline_s"] = t_single
    results["split_call_overhead_s"] = results["rows"]["F0.0"]["moe_block_s"] - t_single
    print(f"\npacked4 single-call baseline: {t_single:.4f} s; "
          f"split-call overhead: {results['split_call_overhead_s']:.4f} s")

    results["interpretation"] = {
        "note": "Two-call split includes double layout/act-quant/reduce per layer; "
                "in-kernel expert-set hot_mask (single call) removes that overhead — "
                "the natural follow-up kernel task if the F-sweep shows signal.",
    }
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
