"""Track D: per-layer golden bootstrap surrogate, N64-128.

Route model per review spec: resample WHOLE real tokens (their actual top-6
expert sets + route weights) from each layer's golden corpus — preserving
per-layer expert frequency P(e), token-level top-6 co-occurrence structure,
and layer-dependent concentration. The ONLY extrapolation is window length.

Realism check: surrogate timing at K=56/63 is compared against the real
golden measurements (moe 0.6491 s / 0.7039 s, crossover=0) before the
N64-128 numbers are reported.

All timings: post-fix all-IMMA (crossover=0) hot8 — the promoted config —
plus c3 reference. Thermal: per-rep window certification.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from mosaicvm.bench.wide_verify import _bench_blocks
from mosaicvm.bench.floors import HIDDEN, N_EXPERTS, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.kernels.mxint import fused_moe_mxint
from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts

GOLDEN = Path("artifacts/golden_routes.json")
OUT = Path("artifacts/wideverify_stage2.json")
ORACLE = Path("artifacts/branchpack_oracle.json")
K_SWEEP = (64, 80, 96, 112, 124, 128)
CORE_K56_S = 0.1024
REAL_ANCHORS = {56: 0.6491, 63: 0.7039}   # post-fix c0 golden moe_block (s)


def layer_token_pools(golden):
    """Per layer: list of (ids[6], wts[6]) per real token across all captures."""
    pools = {L: [] for L in range(N_LAYERS)}
    for c in golden.get("captures", []):
        for li, layer in enumerate(c.get("layers", [])):
            ids = layer["expert_ids"]
            ws = layer["weights"]
            per_tok_ids = ids if isinstance(ids[0], list) else \
                [ids[i:i + 6] for i in range(0, len(ids), 6)]
            per_tok_ws = ws if isinstance(ws[0], list) else \
                [ws[i:i + 6] for i in range(0, len(ws), 6)]
            for t_ids, t_ws in zip(per_tok_ids, per_tok_ws):
                pools[li].append((t_ids, t_ws))
    return pools


def bootstrap_window(pool, k, gen):
    """Resample k whole tokens (with replacement) from this layer's pool."""
    idx = torch.randint(0, len(pool), (k,), generator=gen)
    ids = torch.tensor([pool[i][0] for i in idx], dtype=torch.int64)
    wts = torch.tensor([pool[i][1] for i in idx], dtype=torch.float32)
    return ids, wts


def window_stats(ids):
    counts = {}
    for e in ids.flatten().tolist():
        counts[e] = counts.get(e, 0) + 1
    u = len(counts)
    ms = sorted(counts.values(), reverse=True)
    p90 = ms[min(len(ms) - 1, int(0.1 * len(ms)))]
    return {"unique": u, "mean_m": sum(ms) / u, "p90_m": p90,
            "max_m": ms[0], "frac_m1": sum(1 for m in ms if m == 1) / u}


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
    golden = json.loads(GOLDEN.read_text())
    pools = layer_token_pools(golden)
    n_tokens = {L: len(p) for L, p in pools.items()}
    print(f"token pool per layer: min={min(n_tokens.values())} "
          f"max={max(n_tokens.values())}")
    gen = torch.Generator().manual_seed(4242)

    oracle = json.loads(ORACLE.read_text())["oracle"]
    bank = _allocate_mxint_experts(N_EXPERTS)
    t_start = snapshot()
    rep_snaps = []
    results = {"authority": "per-layer golden bootstrap (whole-token resampling; "
               "only window length extrapolated)",
               "real_anchors_c0": {str(k): v for k, v in REAL_ANCHORS.items()},
               "configs": {}}

    for cfg_name, cfg in (("c0_all_imma", {"crossover": 0}),
                          ("c3_owner", {"crossover": 3})):
        rows = {}
        for K in (56, 63, *K_SWEEP):
            routes = [bootstrap_window(pools[L % N_LAYERS], K, gen)
                      for L in range(11 * N_LAYERS)]
            stats = window_stats(routes[0][0])
            xs = [torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
                  for _ in range(4)]
            ids_t = [r[0].to("cuda") for r in routes]
            w_t = [r[1].to("cuda") for r in routes]
            idx = 0

            def step():
                nonlocal idx
                for _ in range(N_LAYERS):
                    fused_moe_mxint(xs[idx % 4], ids_t[idx], w_t[idx],
                                    bank=bank, weight_repr="hot8",
                                    imma_k=8, **cfg)
                    idx += 1

            rep_snaps.append(snapshot())
            t = _bench_blocks(step, 10)
            ea = oracle.get(str(K), {}).get("expected_accepted_depth")
            if ea is None and K == 63:
                ea = 31.0  # interpolation 28.27@56 -> 31.58@64 (flagged)
            t_target = t + CORE_K56_S
            rows[str(K)] = {"moe_block_s": t, "t_target_s": t_target,
                            "raw_per_s": K / t_target, "E_A": ea,
                            "R_oracle_per_s": ea / t_target if ea else None,
                            "route_stats": stats, "E_A_interpolated": K == 63}
            del routes, ids_t, w_t, xs
            torch.cuda.empty_cache()
        results["configs"][cfg_name] = rows
        r = rows["56"]
        print(f"[{cfg_name}] K56 bootstrap: {r['moe_block_s']:.4f}s "
              f"(real {REAL_ANCHORS[56]:.4f}s, ratio "
              f"{r['moe_block_s']/REAL_ANCHORS[56]:.3f}) "
              f"uniq={r['route_stats']['unique']} meanM={r['route_stats']['mean_m']:.2f}")
        for K in K_SWEEP:
            r = rows[str(K)]
            print(f"  K={K:3d}: moe={r['moe_block_s']:.4f}s "
                  f"raw={r['raw_per_s']:5.1f}/s R_oracle="
                  f"{r['R_oracle_per_s'] and round(r['R_oracle_per_s'],1)}/s "
                  f"uniq={r['route_stats']['unique']} "
                  f"meanM={r['route_stats']['mean_m']:.2f} p90M={r['route_stats']['p90_m']}")

    t_end = snapshot()
    label = certify(rep_snaps, t_start, t_end)
    results["thermal"] = {"label": label, "start": t_start, "end": t_end}
    print(f"thermal [{label}]")

    old = json.loads(OUT.read_text())
    old["per_layer_bootstrap"] = results
    OUT.write_text(json.dumps(old, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
