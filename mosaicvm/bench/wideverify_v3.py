"""Track D v3: no-wrap anchors + two explicit bounds for N64-128.

Per canonical §29-32:
  K <= 63 : consecutive windows, start in [0, n-K] — NO WRAP. Stats are
            medians over ALL measured windows (fixes the v2 single-sample
            uniq-134-vs-median-86 bookkeeping conflict).
  K > 63  : two labeled bounds (authoritative golden still BLOCKED):
            OPTIMISTIC  — single-capture wrap (U capped at capture total)
            PESSIMISTIC — cross-capture same-layer token mix (higher U)
Configs: post-fix all-IMMA (crossover=0) hot8, exact. Dual headline cores.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from mosaicvm.bench.wide_verify import _bench_blocks, _golden_windows
from mosaicvm.bench.floors import HIDDEN, N_EXPERTS, N_LAYERS
from mosaicvm.bench.thermal import snapshot, warmup_steady
from mosaicvm.kernels.mxint import fused_moe_mxint
from mosaicvm.bench.floor_b_moe import _allocate_mxint_experts

GOLDEN = Path("artifacts/golden_routes.json")
OUT = Path("artifacts/wideverify_stage2.json")
K_BOUNDS = (64, 80, 96, 112, 124, 128)
CORE_BF16, CORE_PROD = 0.1024, 0.173
REAL = {56: 0.6491, 63: 0.7039}


def capture_layer_tokens(golden):
    out = []
    for c in golden.get("captures", []):
        for li, layer in enumerate(c.get("layers", [])):
            ids, ws = layer["expert_ids"], layer["weights"]
            pi = ids if isinstance(ids[0], list) else [ids[i:i+6] for i in range(0, len(ids), 6)]
            pw = ws if isinstance(ws[0], list) else [ws[i:i+6] for i in range(0, len(ws), 6)]
            out.append((c.get("category", "?"), li, list(zip(pi, pw))))
    return out


def win_nowrap(toks, k, gen):
    start = int(torch.randint(0, len(toks) - k + 1, (1,), generator=gen))
    sel = toks[start:start + k]
    return (torch.tensor([t[0] for t in sel], dtype=torch.int64),
            torch.tensor([t[1] for t in sel], dtype=torch.float32))


def win_wrap(toks, k, gen):
    n = len(toks)
    start = int(torch.randint(0, n, (1,), generator=gen))
    sel = [toks[(start + i) % n] for i in range(k)]
    return (torch.tensor([t[0] for t in sel], dtype=torch.int64),
            torch.tensor([t[1] for t in sel], dtype=torch.float32))


def win_mixed(pools_same_layer, k, gen):
    toks = pools_same_layer[int(torch.randint(0, len(pools_same_layer), (1,), generator=gen))]
    return win_wrap(toks, k, gen)  # diversity via prior mixing


def u_of(ids):
    return len(set(ids.flatten().tolist()))


def certify(snaps, start, end):
    temps = [s.get("temp_c") for s in snaps if s.get("temp_c") is not None]
    drift = max(temps) - min(temps) if temps else 99
    c0 = start.get("sm_clock_mhz")
    locked = all(s.get("sm_clock_mhz") == c0 for s in snaps + [end])
    return ("STEADY" if drift <= 3 else "CONTROLLED" if drift <= 5 else "UNKNOWN") if locked else "UNKNOWN"


def main():
    assert "3060" in torch.cuda.get_device_name(0).lower()
    warmup_steady(120.0)
    golden = json.loads(GOLDEN.read_text())
    pools = capture_layer_tokens(golden)
    # group same-layer captures for the pessimistic mixer
    by_layer = {}
    for cat, li, toks in pools:
        by_layer.setdefault(li, []).append(toks)
    gen = torch.Generator().manual_seed(31337)

    oracle = json.loads(Path("artifacts/branchpack_oracle.json").read_text())["oracle"]
    bank = _allocate_mxint_experts(N_EXPERTS)
    t_start = snapshot()
    snaps = []

    def measure(K, routes):
        xs = [torch.randn(K, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
        ids_t = [r[0].to("cuda") for r in routes]
        w_t = [r[1].to("cuda") for r in routes]
        idx = 0
        def step():
            nonlocal idx
            for _ in range(N_LAYERS):
                fused_moe_mxint(xs[idx % 4], ids_t[idx], w_t[idx], bank=bank,
                                weight_repr="hot8", imma_k=8, crossover=0)
                idx += 1
        snaps.append(snapshot())
        t = _bench_blocks(step, 10)
        u_med = statistics.median(u_of(r[0]) for r in routes)
        del xs, ids_t, w_t
        torch.cuda.empty_cache()
        return t, u_med

    res = {"authority": "v3: no-wrap anchors (K<=63) + OPTIMISTIC wrap / "
           "PESSIMISTIC mixed bounds (K>63); stats = medians over all windows",
           "rows": {}}

    # --- anchors (no wrap) ---
    for K in (56, 63):
        routes = [win_nowrap(pools[(i * 5) % len(pools)][2], K, gen)
                  for i in range(11 * N_LAYERS)]
        t, u = measure(K, routes)
        ea = oracle.get(str(K), {}).get("expected_accepted_depth") or (31.0 if K == 63 else None)
        res["rows"][f"{K}_nowrap_anchor"] = {
            "moe_s": t, "u_median": u, "E_A": ea,
            "ratio_vs_real": t / REAL[K],
            "R_oracle_prodcore": ea / (t + CORE_PROD) if ea else None,
            "R_oracle_bf16core": ea / (t + CORE_BF16) if ea else None,
        }
        print(f"K={K} NO-WRAP anchor: moe={t:.4f}s ratio={t/REAL[K]:.3f} "
              f"U_med={u:.0f} R_oracle={ea/(t+CORE_PROD):.1f}/s (prodcore)")

    # --- bounds K>63 ---
    for K in K_BOUNDS:
        for label, sampler in (
            ("optimistic_wrap", lambda li: win_wrap(pools[(li * 7) % len(pools)][2], K, gen)),
            ("pessimistic_mixed", lambda li: win_wrap(
                sum(by_layer[li % N_LAYERS], []), K, gen)),
        ):
            routes = [sampler(i) for i in range(11 * N_LAYERS)]
            t, u = measure(K, routes)
            ea = oracle.get(str(K), {}).get("expected_accepted_depth")
            key = f"{K}_{label}"
            res["rows"][key] = {"moe_s": t, "u_median": u, "E_A": ea,
                                "R_oracle_prodcore": ea / (t + CORE_PROD) if ea else None,
                                "R_oracle_bf16core": ea / (t + CORE_BF16) if ea else None}
            print(f"K={K:3d} {label:18s}: moe={t:.4f}s U_med={u:.0f} "
                  f"R_oracle={ea and round(ea/(t+CORE_PROD),1)}/s (prodcore)")

    t_end = snapshot()
    res["thermal"] = {"label": certify(snaps, t_start, t_end)}
    old = json.loads(OUT.read_text())
    old["v3_bounds"] = res
    OUT.write_text(json.dumps(old, indent=2, default=str))
    print(f"thermal [{res['thermal']['label']}]; wrote {OUT}")


if __name__ == "__main__":
    main()
