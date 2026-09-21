"""Routing coverage: expert concentration statistics from synthetic traces.

Reads ``artifacts/routing_traces.json`` and computes unique/mean/median/max M
and top-N coverage fractions from ``expert_ids``.
"""
from __future__ import annotations

import json
from pathlib import Path

ARTIFACT = Path("artifacts/routing_coverage.json")
COVER_NS = (4, 8, 16, 24, 32, 48, 64)


def _coverage(ids: list[int], n_experts: int = 256) -> dict:
    """Compute coverage fractions for a flat expert-id list."""
    counts: dict[int, int] = {}
    for e in ids:
        counts[e] = counts.get(e, 0) + 1
    total = len(ids)
    unique = len(counts)
    if total == 0:
        return {"unique": 0, "mean_m": 0.0, "median_m": 0.0, "max_m": 0,
                **{f"top{n}_cover_frac": 0.0 for n in COVER_NS}}
    sorted_counts = sorted(counts.values(), reverse=True)
    mean_m = total / unique if unique else 0.0
    median_m = float(sorted_counts[len(sorted_counts) // 2])
    max_m = sorted_counts[0]
    cover = {}
    for n in COVER_NS:
        top_sum = sum(sorted_counts[:n])
        cover[f"top{n}_cover_frac"] = top_sum / total if total else 0.0
    return {"unique": unique, "mean_m": mean_m, "median_m": median_m, "max_m": max_m, **cover}


def run() -> dict:
    traces_path = Path("artifacts/routing_traces.json")
    if not traces_path.exists():
        print(f"routing_traces.json not found at {traces_path}")
        return {}
    data = json.loads(traces_path.read_text())
    rows = []
    for tr in data.get("traces", []):
        flat_ids = [e for row in tr["expert_ids"] for e in row]
        cov = _coverage(flat_ids)
        rows.append({
            "family": tr["family"],
            "source": tr.get("source", "synthetic"),
            "k": tr["k"],
            "top_k": tr["top_k"],
            "assignments": tr["assignments"],
            "unique": cov["unique"],
            "mean_m": cov["mean_m"],
            "median_m": cov["median_m"],
            "max_m": cov["max_m"],
            **{k: v for k, v in cov.items() if k.startswith("top")},
        })
    out = {
        "cover_ns": list(COVER_NS),
        "rows": rows,
    }
    Path("artifacts").mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "n_rows": len(rows),
        "wrote": str(ARTIFACT),
    }, indent=2))
    return out


if __name__ == "__main__":
    run()
