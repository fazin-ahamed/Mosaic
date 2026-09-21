"""Track D v2: consecutive-window golden bootstrap, N64-128.

Fixes the v1 pool-mixing defect (all prompts pooled per layer inflated
U(K) 116 vs real ~86 at K56): pools are now per (capture, layer) and each
window resamples a CONSECUTIVE run of real tokens (wrap within capture),
preserving within-prompt expert co-occurrence and the real U(K) growth.

Anchors: real golden K56 0.6491 s / K63 0.7039 s (crossover=0 hot8).
Headline R_oracle reported with BOTH the BF16-core constant (0.1024) and
the production-core projection (0.173). Track D remains BLOCKED for
authoritative golden N64-128.
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
K_SWEEP = (64, 80, 96, 112, 124, 128)
CORE_BF16 = 0.1024
CORE_PROD = 0.173
REAL_ANCHORS = {56: 0.6491, 63: 0.7039}


def capture_layer_tokens(golden):
    """[(capture, layer, tokens)] where tokens = [(ids6, wts6), ...] in order."""
    out = []
    for c in golden.get("captures", []):
        for li, layer in enumerate(c.get("layers", [])):
            ids = layer["expert_ids"]
            ws = layer["weights"]
            per_ids = ids if isinstance(ids[0], list) else \
                [ids[i:i + 6] for i in range(0, len(ids), 6)]
            per_ws = ws if isinstance(ws[0], list) else \
                [ws[i:i + 6] for i in range(0, len(ws), 6)]
            toks = list(zip(per_ids, per_ws))
            if len(toks) >= 32:
                out.append((c.get("category", "?"), li, toks))
    return out


def consecutive_window(toks, k, gen):
    """Start at random offset, take k consecutive tokens (wrap within capture)."""
    n = len(toks)
    start = int(torch.randint(0, n, (1,), generator=gen))
    idx = [(start + i) % n for i in range(k)]
    ids = torch.tensor([toks[i][0] for i in idx], dtype=torch.int64)
    wts = torch.tensor([toks[i][1] for i in idx], dtype=torch.float32)
    return ids, wts


def window_stats(ids):
    counts = {}
    for e in ids.flatten().tolist():
        counts[e] = counts.get(e, 0) + 1
    u = len(counts)
    ms = sorted(counts.values(), reverse=True)
    return {"unique": u, "mean_m": sum(ms) / u,
            "p90_m": ms[min(len(ms) - 1, int(0.1 * len(ms)))]}


def certify(snaps, start, end):
    temps = [s.get("temp_c") for s in snaps if s.get("temp_c") is not None]
    drift = max(temps) - min(temps) if temps else 99
    c0 = start.get("sm_clock_mhz")
    locked = all(s.get("sm_clock_mhz") == c0 for s in snaps + [end])
    return ("STEADY" if drift <= 3 else "CONTROLLED" if drift <= 5 else "UNKNOWN") \
        if locked else "UNKNOWN"


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    warmup_steady(120.0)
    golden = json.loads(GOLDEN.read_text())
    pools = capture_layer_tokens(golden)
    n_tok = {cat: len(t) for cat, _, t in pools[:0]} or None
    lens = sorted({len(t) for _, _, t in pools})
    print(f"capture-layer pools: {len(pools)}, token counts: {lens}")
    gen = torch.Generator().manual_seed(777)

    oracle = json.loads(Path("artifacts/branchpack_oracle.json").read_text())["oracle"]
    bank = _allocate_mxint_experts(N_EXPERTS)
    t_start = snapshot()
    snaps = []
    results = {"authority": "consecutive-window golden bootstrap v2 "
               "(per-capture pools, wrap-within-capture; U(K)-realistic)",
               "configs": {}}

    for cfg_name, cfg in (("c0_all_imma", {"crossover": 0}),):
        rows = {}
        for K in (56, 63, *K_SWEEP):
            routes = []
            for i in range(11 * N_LAYERS):
                cat, li, toks = pools[(i * 5) % len(pools)]
                routes.append(consecutive_window(toks, K, gen))
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

            snaps.append(snapshot())
            t = _bench_blocks(step, 10)
            ea = oracle.get(str(K), {}).get("expected_accepted_depth") or \
                (31.0 if K == 63 else None)
            rows[str(K)] = {
                "moe_block_s": t,
                "raw_bf16core_per_s": K / (t + CORE_BF16),
                "raw_prodcore_per_s": K / (t + CORE_PROD),
                "R_oracle_bf16core_per_s": ea / (t + CORE_BF16) if ea else None,
                "R_oracle_prodcore_per_s": ea / (t + CORE_PROD) if ea else None,
                "E_A": ea, "route_stats": stats,
            }
            del routes, ids_t, w_t, xs
            torch.cuda.empty_cache()
        results["configs"][cfg_name] = rows
        r56 = rows["56"]
        print(f"[{cfg_name}] K56: {r56['moe_block_s']:.4f}s "
              f"(real {REAL_ANCHORS[56]:.4f}s ratio {r56['moe_block_s']/REAL_ANCHORS[56]:.3f}) "
              f"uniq={r56['route_stats']['unique']} (real ~86)")
        for K in (63, *K_SWEEP):
            r = rows[str(K)]
            ro_p = r["R_oracle_prodcore_per_s"]
            ro_b = r["R_oracle_bf16core_per_s"]
            print(f"  K={K:3d}: moe={r['moe_block_s']:.4f}s uniq={r['route_stats']['unique']:3d} "
                  f"meanM={r['route_stats']['mean_m']:.2f} "
                  f"R_oracle={ro_b and round(ro_b,1)}/{ro_p and round(ro_p,1)} "
                  f"(bf16core/prodcore)")

    t_end = snapshot()
    results["thermal"] = {"label": certify(snaps, t_start, t_end),
                          "start": t_start, "end": t_end}
    print(f"thermal [{results['thermal']['label']}]")

    old = json.loads(OUT.read_text())
    old["consecutive_bootstrap_v2"] = results
    OUT.write_text(json.dumps(old, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
