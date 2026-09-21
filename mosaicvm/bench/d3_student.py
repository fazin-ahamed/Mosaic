"""CPU-only future-h40 PCA/ridge pilot; exact source tail remains the evaluator.

Fit transforms on train prompts only. Compare open-loop K32 against recursive
K8 chunks without token/tail feedback, at one exact boundary per val prompt.
Sliding training pairs are not independent evaluation cycles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch


def load_captures(directory, manifest, selected=None,
                  horizon=32, require_both=True):
    membership, texts = {}, {}
    for split, cats in manifest["splits"].items():
        for cat in cats:
            if cat in membership:
                raise ValueError(f"Repeated split membership: {cat}")
            membership[cat] = split
    prompts = {p["category"]: p["prompt"] for p in manifest["prompts"]}
    if len(prompts) != len(manifest["prompts"]) or prompts.keys() != membership.keys():
        raise ValueError("Manifest prompt IDs must match disjoint splits")
    for cat, prompt in prompts.items():
        normalized = " ".join(prompt.lower().split())
        if normalized in texts and texts[normalized] != membership[cat]:
            raise ValueError("Duplicate prompt text across splits")
        texts[normalized] = membership[cat]
    paths = sorted(Path(directory).glob("*.pt"))
    if selected is not None:
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate selected prompt")
        missing = set(selected) - {p.stem for p in paths}
        if missing:
            raise ValueError(f"Missing requested captures: {sorted(missing)}")
        paths = [p for p in paths if p.stem in selected]
    captures = []
    for path in paths:
        c = torch.load(path, map_location="cpu", weights_only=True)
        cat = c.get("category")
        if (cat != path.stem or membership.get(cat) not in ("train", "val")
                or c.get("split") != membership[cat]):
            raise ValueError(f"Train/val manifest membership required: {path}")
        if (c.get("teacher_class") != "SOURCE_CLASS_TORCH_SHIM"
                or c.get("prompt") != prompts[cat]
                or c.get("prompt_sha256") != hashlib.sha256(prompts[cat].encode()).hexdigest()
                or not c.get("script_sha256") or not c.get("snapshot_revision")):
            raise ValueError(f"Invalid source capture provenance: {cat}")
        h = c["h40"]
        s = c["S"]
        if (not isinstance(s, int) or s < 1 or h.ndim != 3 or h.shape[1:] != (4, 4096)
                or h.dtype != torch.bfloat16 or not torch.isfinite(h).all()
                or c["h42"].shape != h.shape or c["h42"].dtype != torch.bfloat16
                or not torch.isfinite(c["h42"]).all()
                or len(c["tokens"]) != len(h) + 1 or len(h) < s + horizon
                or c.get("state_position_start") != s - 1
                or c.get("gen") != c["tokens"][s:]
                or not torch.equal(c["states"][40], h[s - 1:])):
            raise ValueError(f"Invalid full-state/root/future/terminal-label alignment: {cat}")
        c["capture_path"] = str(path.resolve())
        c["capture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        captures.append(c)
    if require_both and \
            {c["split"] for c in captures} != {"train", "val"}:
        raise ValueError("At least one train and one "
                         "validation capture required")
    if len({c["snapshot_revision"] for c in captures}) != 1:
        raise ValueError("Mixed source revisions")
    return captures


def fit(train, horizon=32, rank=32, ridge=1.0):
    """Train-only PCA and per-displacement ridge, returning full-state decoder.

    The unmodeled orthogonal root component is held fixed. This is a bounded
    baseline, not a token-conditioned or tail-feedback student.
    """
    if not train or any(c["split"] != "train" for c in train):
        raise ValueError("fit accepts train captures only")
    if horizon < 1 or rank < 1 or ridge <= 0:
        raise ValueError("Positive horizon/rank/ridge required")
    sequences = [c["h40"][c["S"] - 1:].float().flatten(1) for c in train]
    if any(len(x) <= horizon for x in sequences):
        raise ValueError("Every training prompt must cover root + full horizon")
    # Each prompt has equal PCA weight, even with different continuation lengths.
    mean = torch.stack([x.mean(0) for x in sequences]).mean(0)
    centered = torch.cat([(x - mean) / len(x) ** 0.5 for x in sequences])
    effective_rank = min(rank, centered.shape[0] - 1, centered.shape[1])
    _, _, basis = torch.pca_lowrank(centered, q=effective_rank, center=False, niter=3)
    latent = [(x - mean) @ basis for x in sequences]
    scale = torch.cat(latent).square().mean(0).sqrt().clamp_min(1e-6)
    latent = [z / scale for z in latent]
    weights, counts = [], []
    for step in range(1, horizon + 1):
        features, targets = [], []
        for z in latent:
            n = len(z) - step
            features.append(torch.cat([z[:-step], torch.ones(n, 1)], 1) / n ** 0.5)
            targets.append((z[step:] - z[:-step]) / n ** 0.5)
        x, y = torch.cat(features), torch.cat(targets)
        penalty = torch.eye(effective_rank + 1) * ridge
        penalty[-1, -1] = 0  # Fit offset; regularize state-dependent coefficients.
        weights.append(torch.linalg.solve(x.T @ x + penalty, x.T @ y))
        counts.append(sum(len(z) - step for z in latent))
    return {"mean": mean, "basis": basis, "scale": scale,
            "weights": torch.stack(weights), "rank": effective_rank,
            "ridge": ridge, "horizon": horizon, "training_pairs_by_horizon": counts,
            "train_categories": [c["category"] for c in train]}


def predict(model, root, horizon, chunk=None):
    if horizon < 1 or (chunk is None and horizon > model["horizon"]):
        raise ValueError("Unsupported horizon")
    chunk = horizon if chunk is None else chunk
    if chunk < 1 or chunk > model["horizon"]:
        raise ValueError("Unsupported chunk size")
    current = root.float().flatten()
    out = []
    while len(out) < horizon:
        n = min(chunk, horizon - len(out))
        z = (current - model["mean"]) @ model["basis"] / model["scale"]
        feature = torch.cat([z, torch.ones(1)])
        delta = torch.einsum("i,kij->kj", feature, model["weights"][:n])
        states = current + (delta * model["scale"]) @ model["basis"].T
        states = states.reshape(n, *root.shape).to(torch.bfloat16)
        out.extend(states.unbind())
        current = states[-1].float().flatten()
    return torch.stack(out)


def residual_metrics(pred, target):
    p, t = pred.float().flatten(1), target.float().flatten(1)
    e = p - t
    normalized = e / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return {
        "relative_l2_by_horizon": normalized.norm(dim=1).tolist(),
        "cosine_by_horizon": torch.nn.functional.cosine_similarity(p, t, dim=1).tolist(),
        "normalized_mean_bias_l2": normalized.mean(0).norm().item(),
        "residual_lag1_cosine": torch.nn.functional.cosine_similarity(
            e[:-1], e[1:], dim=1).mean().item() if len(e) > 1 else None,
        "selection_metric": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/stage1_prompts.json"))
    parser.add_argument("--categories", help="Exact comma-separated train/val capture IDs")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args()
    if args.horizon < 1 or not 1 <= args.chunk <= args.horizon:
        parser.error("Require 1 <= chunk <= horizon")
    torch.set_num_threads(2)
    torch.manual_seed(104729)
    captures = load_captures(args.data, json.loads(args.manifest.read_text()),
                             args.categories.split(",") if args.categories else None,
                             args.horizon)
    train = [c for c in captures if c["split"] == "train"]
    val = [c for c in captures if c["split"] == "val"]
    start = time.perf_counter()
    model = fit(train, args.horizon, args.rank, args.ridge)
    fit_seconds = time.perf_counter() - start
    args.out.mkdir(parents=True, exist_ok=True)
    model.update({"classification": "D3_TRAIN_ONLY_PCA_RIDGE_ENGINEERING_PILOT",
                  "snapshot_revision": captures[0]["snapshot_revision"],
                  "capture_sha256": {c["category"]: c["capture_sha256"] for c in train},
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    torch.save(model, args.out / "student.pt")
    rows = []
    for c in val:
        s, k = c["S"], args.horizon
        root = c["h40"][s - 1]
        predictions = {"hold": root.expand(k, -1, -1).clone()}
        timings = {}
        for name, chunk in (("open_loop", None), ("recursive_chunk", args.chunk)):
            by_horizon = {}
            for length in sorted({min(8, k), min(16, k), k}):
                predict(model, root, length, chunk)  # One untimed warm-up.
                samples = []
                for _ in range(10):
                    start = time.perf_counter()
                    pred = predict(model, root, length, chunk)
                    samples.append(time.perf_counter() - start)
                by_horizon[str(length)] = sorted(samples)[5]
            predictions[name] = pred
            timings[name] = {"cpu_seconds_p50": by_horizon[str(k)], "repetitions": 10,
                             "cpu_seconds_p50_by_K": by_horizon,
                             "scope": "CPU state prediction + BF16 cast; excludes tail/verifier"}
        artifact = {key: c[key] for key in ("category", "split", "S", "tokens", "prompt",
                    "snapshot_revision", "capture_path", "capture_sha256", "prompt_sha256")}
        artifact.update({"schema_version": 1, "K": k, "verification_width_N": None,
                         "h40_committed": c["h40"][:s].clone(),
                         "predicted_h40": predictions,
                         "h40_target": c["h40"][s:s + k].clone(),
                         "h42_target": c["h42"][:s + k].clone(),
                         "chunk": args.chunk, "chunk_feedback": "predicted h40 only; no tokens or tail",
                         "checkpoint_path": str((args.out / "student.pt").resolve()),
                         "student_cpu_timing": timings})
        torch.save(artifact, args.out / f"{c['category']}.predictions.pt")
        rows.append({"category": c["category"], "split": "val", "K": k,
                     "state_diagnostics": {name: residual_metrics(pred, artifact["h40_target"])
                                           for name, pred in predictions.items()},
                     "student_cpu_timing": timings})
    report = {"classification": "D3_STATE_DIAGNOSTICS_ONLY_SOURCE_TAIL_PENDING",
              "student_pass": False, "production_certified": False,
              "source_tail_acceptance": None, "fit_seconds": fit_seconds,
              "train_prompts": model["train_categories"],
              "val_prompts": [c["category"] for c in val], "rank": model["rank"],
              "training_pairs_by_horizon": model["training_pairs_by_horizon"],
              "K": args.horizon, "chunk": args.chunk, "cpu_threads": 2,
              "platform": platform.platform(), "torch": str(torch.__version__),
              "notes": ["No checkpoint selection by state metrics",
                        "One exact committed boundary per validation prompt",
                        "Recursive chunks use predicted state, no token/tail feedback"],
              "per_prompt": rows}
    (args.out / "diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": str(args.out), "train": len(train), "val": len(val),
                      "fit_seconds": fit_seconds, "source_tail_acceptance": None}))


if __name__ == "__main__":
    main()
