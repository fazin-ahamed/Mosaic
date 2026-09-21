"""Track R: Belady-optimal residency upper bound on GENUINE routes.

Access stream: per prompt, per token position t, per layer L, the six
(layer, expert) objects the router selected. Belady evicts the object
with the furthest next use. Capacities swept in BYTES (VRAM-tier
0.5-4 GB; combined VRAM+RAM tiers where RAM misses are counted
separately).

Outputs per capacity:
  byte hit rate (VRAM tier), H2D bytes/block-equivalent,
  reuse-distance distribution, per-layer hit rates.

Honest caveat: 32-token traces = cold-window Belady (conservative;
steady conversations would score higher). This is a FEASIBILITY GATE,
not a production predictor.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

OUT = Path("artifacts/belady_upper_bound.json")
EXPERT_MB = 13.4
CATS = ["coding", "agent_tool_use", "math_reasoning",
        "general_conversation", "long_context_retrieval",
        "long_generation", "structured_json"]
CAPS_GB = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0)


def load_streams():
    """{cat: [ (token, layer, expert) ... ] in execution order."""
    streams = {}
    for cat in CATS:
        r = torch.load(f"artifacts/source_corpus/{cat}.pt",
                       map_location="cpu", weights_only=False)
        per_layer = defaultdict(list)
        for g in r["gates"]:
            per_layer[g["layer"]].append(g)
        # position-ordered rows per layer
        seq = {}
        for L, gs in per_layer.items():
            rows = []
            for g in gs:
                if g["phase"] == "prefill":
                    rows.append((-1, g["idx"][-1]))
                else:
                    rows.append((g["step"],
                                 g["idx"][0] if len(g["idx"]) == 1
                                 else g["idx"][-1]))
            rows.sort()
            seq[L] = [x[1] for x in rows]
        n_pos = len(seq[0])
        stream = []
        for t in range(n_pos):
            for L in range(43):
                for e in seq[L][t]:
                    stream.append((L, e))
        streams[cat] = stream
    return streams


def belady(stream, cap_objects):
    """Returns (hits, misses, per-layer hits/misses)."""
    # next-use precompute
    next_use = [dict() for _ in stream]  # placeholder-free approach:
    last = {}
    next_pos = [0] * len(stream)
    for i in range(len(stream) - 1, -1, -1):
        obj = stream[i]
        next_pos[i] = last.get(obj, 1 << 60)
        last[obj] = i
    cache = {}       # obj -> (position index of current use, next use)
    hits = misses = 0
    per_layer = defaultdict(lambda: [0, 0])
    for i, obj in enumerate(stream):
        L = obj[0]
        if obj in cache:
            hits += 1
            per_layer[L][0] += 1
        else:
            misses += 1
            per_layer[L][1] += 1
            if len(cache) >= cap_objects:
                # evict furthest next use
                victim, nv = None, -1
                for o, (_, u) in cache.items():
                    if u > nv:
                        victim, nv = o, u
                del cache[victim]
        cache[obj] = (i, next_pos[i])
        # refresh next-use for objects equal to this one later
    return hits, misses, dict(per_layer)


def main():
    streams = load_streams()
    total = sum(len(s) for s in streams.values())
    print(f"streams: {[(c, len(s)) for c, s in streams.items()]}")
    print(f"total object accesses: {total} "
          f"({total * EXPERT_MB / 1000:.0f} GB of demand)")

    results = {"note": "cold-window Belady on 32-token genuine traces; "
                       "conservative vs steady conversation",
               "demand_gb_total": round(total * EXPERT_MB / 1000, 1),
               "capacities": {}}
    # reuse-distance distribution
    rd = []
    for cat, stream in streams.items():
        last = {}
        for i, obj in enumerate(stream):
            if obj in last:
                rd.append(i - last[obj])
            last[obj] = i
    rd.sort()
    results["reuse_distance"] = {
        "n": len(rd), "p50": rd[len(rd) // 2], "p90": rd[int(.9 * len(rd))],
        "p95": rd[int(.95 * len(rd))], "p99": rd[int(.99 * len(rd))],
        "max": rd[-1]}
    print(f"reuse distance: p50={rd[len(rd)//2]} p90={rd[int(.9*len(rd))]} "
          f"p99={rd[int(.99*len(rd))]}")

    for cap_gb in CAPS_GB:
        cap_obj = int(cap_gb * 1000 / EXPERT_MB)
        H = M = 0
        for stream in streams.values():
            h, m, _ = belady(stream, cap_obj)
            H += h
            M += m
        hit = H / (H + M)
        results["capacities"][f"{cap_gb}GB"] = {
            "objects": cap_obj, "byte_hit_rate": round(hit, 4),
            "miss_bytes_gb_per_32tok_block": round(
                M * EXPERT_MB / 1000 / len(streams), 1),
            "h2d_s_per_block_at_12.9": round(
                M * EXPERT_MB / 1e9 / 12.9 / len(streams), 3),
        }
        print(f"cap {cap_gb:5.1f}GB ({cap_obj:4d} obj): "
              f"hit={hit:.3f}  miss/block="
              f"{results['capacities'][f'{cap_gb}GB']['miss_bytes_gb_per_32tok_block']:5.1f}GB "
              f"({results['capacities'][f'{cap_gb}GB']['h2d_s_per_block_at_12.9']:.2f}s H2D)")

    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
