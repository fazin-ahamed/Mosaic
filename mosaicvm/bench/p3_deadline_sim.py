"""Track P3-A: per-layer H2D deadline simulator v2 (cache-aware).

Correct model: a 32-token block needs ~62 unique experts PER LAYER
(32 tokens x top-6, partial overlap), but only 256 unique experts per
BLOCK total (heavy cross-layer reuse). Movement volume therefore
depends on the VRAM staging cache capacity C and the layer access
order:

  - C = 256 (3.4 GB): total loads = 256 experts = 3.43 GB/block
  - C < 256: evictions force reloads; movement = 256 + reloads

Simulation: per layer L, demand set D_L (from real traces). Experts
not resident at layer-L FFN-start must H2D in time. Prefetch policies:

  JIT   reactive: transfer starts at layer-L route-ready
        (window = T_pre_ffn = 3.3 ms)
  O1    one-layer-ahead ORACLE (upper bound on any predictor):
        layer L+1's demand known during layer L's compute
        (window = full layer = 14.2 ms)

H2D stream serializes at 12.9 GB/s. Cache evicts LRU at capacity.
Metric: unhidden bubble ms per N32 block (mean/p50/p90/max) and total
movement GB.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

GOLDEN = Path("artifacts/golden_routes.json")
OUT = Path("artifacts/p3_deadline_sim.json")

T_LAYER = 14.2e-3
T_PRE_FFN = 3.3e-3
BW = 12.9e9
EXPERT_BYTES = int(13.4e6)
N = 32
T_VERIFY = 0.610


def load_blocks():
    g = json.loads(GOLDEN.read_text())
    by_prompt = defaultdict(dict)
    for r in g["windows"]:
        by_prompt[r["prompt_idx"]][r["layer_name"]] = r["expert_ids"]

    def key(nm):
        try:
            return int(nm.split(".")[2])
        except (IndexError, ValueError):
            return 0

    blocks = []
    for p, layers in by_prompt.items():
        names = sorted(layers.keys(), key=key)
        L = len(next(iter(layers.values())))
        for o in range(0, L - N + 1, N):
            blk = []
            for nm in names:
                u = set()
                for row in layers[nm][o:o + N]:
                    u.update(row)
                blk.append(u)
            blocks.append(blk)
    return blocks


def simulate(block, C, policy, warm=None):
    """Warm-start cache (top-C by frequency, from `warm` set) +
    serialized H2D. Returns (bubble_s, loads, moved_bytes, per-layer
    bubble list). Bubble per layer counted once: max lateness of that
    layer's missing experts at its FFN start."""
    resident = set(warm) if warm else set()
    loads = 0
    moved = 0
    n_layers = len(block)
    t_ffn = [L * T_LAYER + T_PRE_FFN for L in range(n_layers)]
    stream_free = 0.0
    ready_of = {}          # expert -> ready time (this block)
    layer_bubble = [0.0] * n_layers

    # oracle O1: loads for layer L may start one layer early
    for L, demand in enumerate(block):
        need = demand - resident
        for e in sorted(need):
            loads += 1
            moved += EXPERT_BYTES
            if policy == "JIT":
                start = max(stream_free, t_ffn[L] - T_PRE_FFN)
            else:
                start = max(stream_free,
                            (L - 1) * T_LAYER if L > 0 else 0.0)
            ready = start + EXPERT_BYTES / BW
            stream_free = ready
            ready_of[e] = ready
            resident.add(e)
        # per-layer bubble = lateness of missing experts at FFN start
        for e in demand:
            r = ready_of.get(e)
            if r is not None and r > t_ffn[L]:
                layer_bubble[L] = max(layer_bubble[L], r - t_ffn[L])
    return sum(layer_bubble), loads, moved, layer_bubble


def main():
    blocks = load_blocks()
    print(f"{len(blocks)} N{N} blocks")
    # per-layer new-expert profile (cold, C=infinite)
    prof = []
    for blk in blocks:
        seen = set()
        new = []
        for d in blk:
            new.append(len(d - seen))
            seen |= d
        prof.append(new)
    mean_new = [sum(x[i] for x in prof) / len(prof)
                for i in range(43)]
    print("new experts per layer (cumulative-coverage view): "
          f"L0={mean_new[0]:.0f} L1={mean_new[1]:.0f} "
          f"L4={mean_new[4]:.0f} L10={mean_new[10]:.0f} "
          f"L21={mean_new[21]:.0f} L42={mean_new[42]:.0f}")

    # warm cache: top-C experts by global use frequency (steady state
    # after cross-block retention; cold-start affects only block 1)
    from collections import Counter
    freq = Counter()
    for blk in blocks:
        for d in blk:
            for e in d:
                freq[e] += 1
    top = [e for e, _ in freq.most_common()]

    results = {"blocks": len(blocks),
               "per_layer_new_experts_mean": [round(x, 1)
                                              for x in mean_new],
               "policies": {}}
    for policy in ("JIT", "O1"):
        for C in (64, 96, 128, 160, 192, 256):
            bubbles, loads, moved = [], [], []
            for blk in blocks:
                warm = set(top[:C]) if C < 256 else set(top)
                b, ld, mv, _ = simulate(blk, C, policy, warm)
                bubbles.append(b)
                loads.append(ld)
                moved.append(mv)
            bubbles.sort()
            n = len(bubbles)
            results["policies"][f"{policy}_C{C}"] = {
                "bubble_mean_ms": round(sum(bubbles) / n * 1e3, 1),
                "bubble_p50_ms": round(bubbles[n // 2] * 1e3, 1),
                "bubble_p90_ms": round(
                    bubbles[int(.9 * (n - 1))] * 1e3, 1),
                "bubble_max_ms": round(bubbles[-1] * 1e3, 1),
                "mean_loads_per_block": round(sum(loads) / n, 1),
                "mean_moved_gb": round(sum(moved) / n / 1e9, 2),
                "h2d_time_ms": round(sum(moved) / n / BW * 1e3, 1),
            }
    OUT.write_text(json.dumps({
        "classification": "P3_PER_LAYER_DEADLINE_SIM_V2",
        "model": {"T_layer_s": T_LAYER, "T_pre_ffn_s": T_PRE_FFN,
                  "BW_GB_s": BW / 1e9, "expert_MB": 13.4, "N": N},
        **results}, indent=1))
    for k, v in results["policies"].items():
        print(f"{k:10s}: bubble mean={v['bubble_mean_ms']:7.1f}ms "
              f"p90={v['bubble_p90_ms']:7.1f}ms "
              f"loads={v['mean_loads_per_block']:6.1f} "
              f"moved={v['mean_moved_gb']:5.2f}GB "
              f"({v['h2d_time_ms']:6.1f}ms H2D)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
