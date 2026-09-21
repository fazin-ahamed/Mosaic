"""Synthetic (or golden) routing traces for Floor B families."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from mosaicvm.bench.floors import K_VALUES, N_EXPERTS, TOP_K


def _entropy(counts: torch.Tensor) -> float:
    p = counts.float()
    p = p[p > 0]
    p = p / p.sum()
    return float((-(p * p.log()).sum()).item())


def _coactivation(expert_ids: torch.Tensor, top_n: int = 32) -> list[list[int]]:
    """Pairwise expert co-counts within a token's top-k, top 32 pairs."""
    pairs: Counter[tuple[int, int]] = Counter()
    rows = expert_ids.detach().cpu().tolist()
    for row in rows:
        s = sorted(int(e) for e in row)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                pairs[(s[i], s[j])] += 1
    top = pairs.most_common(top_n)
    return [[a, b, c] for (a, b), c in top]


def make_trace(
    k: int,
    top_k: int = 6,
    n_unique: int | None = None,
    seed: int = 0,
    n_experts: int = N_EXPERTS,
    routed_scaling_factor: float = 1.5,
) -> dict:
    """Returns expert_ids [k, top_k] int64, weights [k, top_k] fp32 (renormalized, x1.5).

    Each token's top_k experts are distinct. n_unique = max(top_k, desired_unique).
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    assignments = k * top_k
    if n_unique is None:
        n_unique = min(assignments, n_experts)
    n_unique = max(top_k, min(n_unique, assignments, n_experts))

    pool = torch.randperm(n_experts, generator=g)[:n_unique]
    rows = []
    used = torch.zeros(n_unique, dtype=torch.int64)
    for t in range(k):
        perm = torch.randperm(n_unique, generator=g)
        pick = perm[:top_k]
        used[pick] += 1
        rows.append(pool[pick])
    expert_ids = torch.stack(rows, 0).to(torch.int64)
    missing = (used == 0).nonzero(as_tuple=False).view(-1)
    if missing.numel():
        for i, mi in enumerate(missing.tolist()):
            t = i % k
            expert_ids[t, i % top_k] = pool[mi]

    weights = torch.rand(k, top_k, generator=g, dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    weights = weights * routed_scaling_factor

    flat = expert_ids.reshape(-1)
    counts = torch.bincount(flat, minlength=n_experts)
    used = counts[counts > 0]
    mean_m = float(used.float().mean().item()) if used.numel() else 0.0
    median_m = float(used.float().median().item()) if used.numel() else 0.0
    max_m = int(used.max().item()) if used.numel() else 0
    reuse_frac = 1.0 - (n_unique / assignments) if assignments else 0.0

    return {
        "k": k,
        "top_k": top_k,
        "n_unique": n_unique,
        "expert_ids": expert_ids,
        "weights": weights,
        "assignments": assignments,
        "unique": n_unique,
        "mean_m": mean_m,
        "median_m": median_m,
        "max_m": max_m,
        "reuse_frac": reuse_frac,
        "entropy": _entropy(counts),
        "coactivation": _coactivation(expert_ids),
    }


def _family_n_unique(family: str, k: int, top_k: int) -> int:
    if family == "peaked":
        return min(top_k, k)
    if family == "mid":
        return min(k * top_k // 2, k * top_k)
    if family == "worst":
        return min(k * top_k, N_EXPERTS)
    raise ValueError(family)


def _serialize_trace(tr: dict, family: str, source: str) -> dict:
    return {
        "family": family,
        "source": source,
        "k": tr["k"],
        "top_k": tr["top_k"],
        "assignments": tr["assignments"],
        "unique": tr["unique"],
        "mean_m": tr["mean_m"],
        "median_m": tr["median_m"],
        "max_m": tr["max_m"],
        "reuse_frac": tr["reuse_frac"],
        "entropy": tr["entropy"],
        "coactivation": tr["coactivation"],
        "expert_ids": tr["expert_ids"].tolist(),
        "weights": tr["weights"].tolist(),
    }


def run() -> dict:
    families = ("peaked", "mid", "worst")
    traces = []
    golden_path = Path("artifacts/golden_routes.json")
    source = "synthetic"
    if golden_path.exists():
        golden = json.loads(golden_path.read_text())
        source = "golden"
        for row in golden.get("traces") or golden.get("rows") or []:
            traces.append({**row, "source": "golden"})

    fam_seed = {"peaked": 1, "mid": 2, "worst": 3}
    for k in K_VALUES:
        for fam in families:
            n_u = _family_n_unique(fam, k, TOP_K)
            tr = make_trace(k, TOP_K, n_unique=n_u, seed=1000 * k + fam_seed[fam])
            traces.append(_serialize_trace(tr, fam, "synthetic"))

    out = {
        "source": source if golden_path.exists() else "synthetic",
        "top_k": TOP_K,
        "k_values": list(K_VALUES),
        "families": list(families),
        "traces": traces,
    }
    Path("artifacts").mkdir(exist_ok=True)
    path = Path("artifacts/routing_traces.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "n_traces": len(traces),
        "source": out["source"],
        "k_values": list(K_VALUES),
        "wrote": str(path),
    }, indent=2))
    return out


if __name__ == "__main__":
    run()
