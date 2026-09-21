"""Track D local surrogate: golden-MATCHED WideVerify sweep N56-128.

Golden captures end at 63 positions/layer, so golden windows beyond K63 are
impossible without a new sandbox capture. This sweep builds surrogate windows
whose ROUTE STRUCTURE matches golden statistics:
  - per-window expert pool size U(K) from the measured golden unique(K) fit
  - each token picks 6 DISTINCT experts from the pool (golden-like
    multiplicity distribution)
Anchored by measured golden K56. Timing-valid under both MoE configs
(numeric validity caveat: grouped-IMMA path is correctness-blocked; all
numbers here are TIMING cost estimates per AGENTS §7).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from mosaicvm.bench.wide_verify import _golden_windows, _bench_blocks
from mosaicvm.bench.floors import HIDDEN, N_EXPERTS, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.kernels.mxint import fused_moe_mxint
from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts

GOLDEN = Path("artifacts/golden_routes.json")
OUT = Path("artifacts/wideverify_stage2.json")
ORACLE = Path("artifacts/branchpack_oracle.json")
K_SWEEP = (56, 64, 80, 96, 112, 124, 128)
CORE_K56_S = 0.1024


def golden_unique_by_k(golden, ks):
    """Median unique experts per K-wide window from the golden corpus."""
    stats = {}
    for k in ks:
        uniq = []
        for c in golden.get("captures", []):
            for layer in c.get("layers", []):
                ids = layer["expert_ids"]
                flat = [e for tok in ids for e in tok] if isinstance(ids[0], list) else ids
                for s in range(0, len(flat) - k * 6 + 1, k * 6):
                    win = flat[s:s + k * 6]
                    uniq.append(len(set(win)))
        if uniq:
            stats[k] = statistics.median(uniq)
    return stats


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    warmup_steady(120.0)
    golden = json.loads(GOLDEN.read_text())

    # fit unique(K) from golden at several K (windows of k tokens x 6)
    gk = golden_unique_by_k(golden, (8, 16, 24, 32, 40, 48, 56))
    print("golden unique(K):", {k: int(v) for k, v in gk.items()})
    import math
    xs = [math.log(k) for k in gk]
    ys = [math.log(gk[k]) for k in gk]
    n = len(xs)
    slope = (n * sum(a * b for a, b in zip(xs, ys)) - sum(xs) * sum(ys)) / \
        (n * sum(a * a for a in xs) - sum(xs) ** 2)
    itc = (sum(ys) - slope * sum(xs)) / n

    def uniq_est(k):
        return min(N_EXPERTS, int(round(math.exp(itc) * (k ** slope))))

    print(f"fit: unique(K) ~ {math.exp(itc):.1f} * K^{slope:.3f}")

    g = torch.Generator(device="cuda").manual_seed(1234)

    def surrogate_window(k):
        pool = torch.randperm(N_EXPERTS, generator=g, device="cuda")[:uniq_est(k)]
        ids = torch.stack([pool[torch.randperm(len(pool), generator=g, device="cuda")[:6]]
                           for _ in range(k)])
        wts = torch.rand(k, 6, generator=g, device="cuda")
        wts = wts / wts.sum(-1, keepdim=True)
        return ids.cpu(), wts.float().cpu()

    bank = _allocate_mxint_experts(N_EXPERTS)
    gw56 = _golden_windows(56, golden)
    oracle = json.loads(ORACLE.read_text())

    res = {"gpu": torch.cuda.get_device_name(0),
           "route_authority": "golden-matched surrogate (pool size from golden unique(K) fit; distinct top-6 per token); anchored by measured golden K56",
           "unique_fit": {"coef": math.exp(itc), "slope": slope, "golden_points": {str(k): v for k, v in gk.items()}},
           "validity": "TIMING cost estimates only; grouped-IMMA numerics correctness-blocked (AGENTS §7)",
           "configs": {}}
    rep_snaps = []
    t_start = snapshot()

    for cfg_name, cfg in (("c3_owner", {"crossover": 3}),
                          ("c0_candidate", {"crossover": 0})):
        rows = {}
        # golden K56 anchor
        routes = [gw56[(i * 7) % len(gw56)] for i in range(6 * N_LAYERS)]
        xs = [torch.randn(56, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
        ids_t = [torch.tensor(r[0], device="cuda", dtype=torch.int64) for r in routes]
        w_t = [torch.tensor(r[1], device="cuda", dtype=torch.float32) for r in routes]
        idx = 0

        def step():
            nonlocal idx
            for _ in range(N_LAYERS):
                fused_moe_mxint(xs[idx % 4], ids_t[idx], w_t[idx], bank=bank,
                                weight_repr="hot8", imma_k=8, **cfg)
                idx += 1

        t_gold = _bench_blocks(step, 5)
        del routes, ids_t, w_t, xs
        torch.cuda.empty_cache()
        rows["56_golden_anchor_s"] = t_gold
        print(f"\n[{cfg_name}] golden K56 anchor: {t_gold:.4f} s")

        for k in K_SWEEP:
            routes = [surrogate_window(k) for _ in range(6 * N_LAYERS)]
            xs = [torch.randn(k, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
            ids_t = [r[0].to("cuda") for r in routes]
            w_t = [r[1].to("cuda") for r in routes]
            idx = 0

            def stepk():
                nonlocal idx
                for _ in range(N_LAYERS):
                    fused_moe_mxint(xs[idx % 4], ids_t[idx], w_t[idx], bank=bank,
                                    weight_repr="hot8", imma_k=8, **cfg)
                    idx += 1

            rep_snaps.append(snapshot())
            t = _bench_blocks(stepk, 5)
            t_target = t + CORE_K56_S
            ea = oracle.get("oracle", {}).get(str(k), {}).get("expected_accepted_depth")
            r_or = ea / t_target if ea else None
            rows[str(k)] = {"moe_block_s": t, "t_target_s": t_target,
                            "raw_per_s": k / t_target,
                            "E_A_oracle": ea,
                            "R_oracle_per_s": r_or}
            print(f"  K={k:3d} (uniq~{uniq_est(k):3d}): moe={t:.4f}s "
                  f"T_target={t_target:.3f}s raw={k/t_target:5.1f}/s "
                  f"R_oracle={r_or and round(r_or,1)}/s")
            del routes, ids_t, w_t, xs
            torch.cuda.empty_cache()
        res["configs"][cfg_name] = rows

    t_end = snapshot()
    temps = [s.get("temp_c") for s in rep_snaps if s.get("temp_c") is not None]
    drift = max(temps) - min(temps) if temps else 99
    clock0 = t_start.get("sm_clock_mhz")
    locked = all(s.get("sm_clock_mhz") == clock0 for s in rep_snaps + [t_end])
    label = "STEADY" if drift <= 3 and locked else ("CONTROLLED" if drift <= 5 and locked else "UNKNOWN")
    res["thermal"] = {"start": t_start, "end": t_end, "label": label,
                      "window_drift_c": drift}

    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    old["golden_matched_surrogate"] = res
    OUT.write_text(json.dumps(old, indent=2, default=str))
    print(f"\nthermal [{label}] drift={drift}°C clocks={clock0}->{t_end.get('sm_clock_mhz')}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
