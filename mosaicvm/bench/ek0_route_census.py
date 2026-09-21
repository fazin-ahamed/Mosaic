"""EK0 — route / M census + compulsory-movement roofline.

Uses REAL per-layer top-6 routed-expert IDs from artifacts/source_corpus/*.pt
(gates: per (layer, decode-position) idx=[6 expert ids]). 43 MoE blocks, 256
routed experts, 6 active + 1 shared per layer.

Simulates the exact d3-B4 speculative verification geometry:
  B1: 1 draft path, verify W positions.
  B2/B4: branch at d3 -> up to B distinct paths; positions across paths that
  differ route independently. For the union we take, per layer, the set of
  experts touched by ALL candidate positions in the cycle.

Outputs:
  P(M): distribution of rows-per-expert within a layer (batching opportunity).
  per-layer expert-union sizes, compulsory packed-FP4 bytes/cycle.
  T_move_min = compulsory cold bytes / 12.90 GB/s (H2D roofline).
  Compares B1/B2/B4 incremental movement.

Bytes per routed expert (packed FP4, from route_manifest): gu 8912896 +
down 4456448 = 13369344 B. Shared expert (hot8-class) larger; shared experts
are permanent-VRAM candidates so excluded from compulsory cold movement.
"""
from __future__ import annotations

import json
import glob
from collections import Counter, defaultdict
from pathlib import Path

import torch

H2D_GBs = 12.90
BYTES_PER_ROUTED = 8912896 + 4456448      # packed FP4 gu+down = 13.37 MB
N_LAYERS = 43
WIN = 6          # speculative window ~ E[A] 3.757 -> verify ~6 positions
VRAM_BUDGET_GB = 12.0


def load_routes():
    """Return {domain: [ per-position { layer: [6 ids] } ]} for decode."""
    out = {}
    for f in sorted(glob.glob("artifacts/source_corpus/*.pt")):
        cat = Path(f).stem
        if cat == "manifest":
            continue
        c = torch.load(f, map_location="cpu", weights_only=False)
        # gates: flat list of {layer, phase, step, idx, w}; decode phase only
        pos = defaultdict(dict)   # step -> {layer: ids}
        for r in c["gates"]:
            if r["phase"] != "decode":
                continue
            step = r["step"]; layer = r["layer"]
            idx = r["idx"]
            if idx and isinstance(idx[0], list):
                idx = idx[0]           # decode: [[6 ids]] -> [6 ids]
            pos[step][layer] = list(idx)
        seq = [pos[s] for s in sorted(pos) if len(pos[s]) == N_LAYERS]
        out[cat] = seq
    return out

def m_histogram(routes):
    """Within a single decode position's layer, each of 6 experts gets 1 row.
    Across a W-position window sharing a path (B1), rows-per-expert per layer
    accumulate. P(M) = dist of that count per (layer, expert, window)."""
    hist = Counter()
    for cat, seq in routes.items():
        for start in range(0, max(1, len(seq) - WIN + 1)):
            window = seq[start:start + WIN]
            for L in range(N_LAYERS):
                cnt = Counter()
                for posmap in window:
                    for e in posmap.get(L, []):
                        cnt[e] += 1
                for e, m in cnt.items():
                    hist[m] += 1
    return hist


def union_bytes(routes, branch=1):
    """Per-cycle compulsory bytes: union of experts across the W-window.
    branch>1 approximates B4 by treating d3 branch positions as extra distinct
    routes (uses the next `branch` positions at depth 3 as independent paths)."""
    per_cat = {}
    for cat, seq in routes.items():
        if len(seq) < WIN:
            continue
        # B1: one window. Branch: also union the alternative d3+ positions.
        layer_union = defaultdict(set)
        window = seq[:WIN]
        for posmap in window:
            for L in range(N_LAYERS):
                for e in posmap.get(L, []):
                    layer_union[L].add(e)
        if branch > 1:
            # extra distinct routes at d3..d3+branch (positions 2..2+branch)
            for extra in seq[2:2 + branch]:
                for L in range(N_LAYERS):
                    for e in extra.get(L, []):
                        layer_union[L].add(e)
        total_experts = sum(len(s) for s in layer_union.values())
        per_cat[cat] = total_experts
    return per_cat


def main():
    routes = load_routes()
    print("domains:", {c: len(s) for c, s in routes.items()})
    ncat = len(routes)

    # --- P(M) ---
    hist = m_histogram(routes)
    tot = sum(hist.values())
    print("\n=== P(M): rows-per-expert per layer over W=%d window ===" % WIN)
    for m in sorted(hist):
        print(f"  M={m}: {hist[m]/tot:.4f}  ({hist[m]})")
    mean_m = sum(m * n for m, n in hist.items()) / tot
    print(f"  mean M = {mean_m:.2f}")

    # --- expert-union + roofline ---
    print("\n=== compulsory expert-union (B1) ===")
    b1 = union_bytes(routes, 1)
    b4 = union_bytes(routes, 4)
    mean_experts_b1 = sum(b1.values()) / len(b1)
    mean_experts_b4 = sum(b4.values()) / len(b4)
    bytes_b1 = mean_experts_b1 * BYTES_PER_ROUTED
    bytes_b4 = mean_experts_b4 * BYTES_PER_ROUTED
    print(f"  B1 mean unique routed-expert slots/cycle: {mean_experts_b1:.0f}"
          f"  ({mean_experts_b1/N_LAYERS:.1f}/layer)")
    print(f"  B4 mean unique routed-expert slots/cycle: {mean_experts_b4:.0f}")
    print(f"  B1 compulsory bytes (all cold): {bytes_b1/1e9:.3f} GB")
    print(f"  B4 compulsory bytes (all cold): {bytes_b4/1e9:.3f} GB")

    print("\n=== H2D roofline @ %.2f GB/s ===" % H2D_GBs)
    t_b1 = bytes_b1 / (H2D_GBs * 1e9) * 1e3
    t_b4 = bytes_b4 / (H2D_GBs * 1e9) * 1e3
    print(f"  T_move_min B1 (all cold): {t_b1:.1f} ms/cycle")
    print(f"  T_move_min B4 (all cold): {t_b4:.1f} ms/cycle")
    print(f"  Exact-30 budget @ E[A]=3.757: 125.2 ms/cycle")

    # --- residency: can the whole routed bank stay resident? ---
    total_routed = N_LAYERS * 256 * BYTES_PER_ROUTED
    print(f"\n=== residency ===")
    print(f"  full routed bank (43*256 experts, packed FP4): "
          f"{total_routed/1e9:.1f} GB (vs {VRAM_BUDGET_GB} GB VRAM)")
    # union of experts EVER touched across all domains (working set)
    ever = defaultdict(set)
    for cat, seq in routes.items():
        for posmap in seq:
            for L in range(N_LAYERS):
                for e in posmap.get(L, []):
                    ever[L].add(e)
    ws = sum(len(s) for s in ever.values())
    print(f"  observed working set (7 domains, 32 pos): {ws} expert-slots "
          f"= {ws*BYTES_PER_ROUTED/1e9:.2f} GB packed")

    out = {"classification": "EK0_ROUTE_CENSUS", "window": WIN,
           "n_layers": N_LAYERS, "bytes_per_routed": BYTES_PER_ROUTED,
           "P_M": {str(m): hist[m] / tot for m in sorted(hist)},
           "mean_M": round(mean_m, 3),
           "mean_experts_B1": round(mean_experts_b1, 1),
           "mean_experts_B4": round(mean_experts_b4, 1),
           "bytes_B1_GB": round(bytes_b1 / 1e9, 4),
           "bytes_B4_GB": round(bytes_b4 / 1e9, 4),
           "T_move_min_B1_ms": round(t_b1, 2),
           "T_move_min_B4_ms": round(t_b4, 2),
           "budget_ms": 125.2,
           "full_bank_GB": round(total_routed / 1e9, 2),
           "working_set_GB": round(ws * BYTES_PER_ROUTED / 1e9, 3)}
    Path("artifacts/ek0_route_census.json").write_text(json.dumps(out, indent=1))
    print("\nwrote artifacts/ek0_route_census.json")


if __name__ == "__main__":
    main()
