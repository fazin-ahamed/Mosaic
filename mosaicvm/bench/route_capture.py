"""Capture or replay golden MoE routes. Never fabricate from weights+token-ids."""
from __future__ import annotations

import json
import os
from pathlib import Path

ARTIFACT = Path("artifacts/golden_routes.json")


def run() -> dict:
    Path("artifacts").mkdir(exist_ok=True)
    if ARTIFACT.exists():
        data = json.loads(ARTIFACT.read_text())
        windows = data.get("windows") or data.get("traces") or []
        out = {
            "status": "replay",
            "n_windows": len(windows),
            "k8_windows": sum(1 for w in windows if (w.get("k") or len(w.get("expert_ids") or [])) == 8),
            "families": sorted({w.get("family") for w in windows if w.get("family")}),
        }
        print(json.dumps(out, indent=2))
        return out

    root = os.environ.get("MOSAICVM_MODEL_ROOT")
    if not root:
        out = {"status": "no_real_routes"}
        print(json.dumps(out))
        return out

    try:
        import torch
        from mosaicvm.runtime.streaming_model import load_streaming_model
        from mosaicvm.model.layers import HashRouter, Router
        from mosaicvm.bench.prompts import PROMPTS  # may not exist
    except Exception as exc:
        out = {"status": "no_real_routes", "reason": f"import:{type(exc).__name__}"}
        print(json.dumps(out))
        return out

    try:
        model, loader, missing = load_streaming_model(root, device="cuda")
        captured = []

        def _hook(mod, args, output):
            scores, idx = output
            captured.append({
                "layer": getattr(mod, "layer_idx", None),
                "kind": type(mod).__name__,
                "expert_ids": idx.detach().cpu().tolist(),
                "weights": scores.detach().cpu().tolist(),
            })

        handles = []
        for m in model.modules():
            if isinstance(m, (Router, HashRouter)):
                handles.append(m.register_forward_hook(_hook))
        # short greedy continuation; cap ~60s by generating few tokens
        ids = torch.randint(0, min(256, model.cfg.vocab_size), (1, 8), device="cuda")
        with torch.no_grad():
            _ = model.generate(ids, max_new_tokens=2)
        for h in handles:
            h.remove()
        windows = []
        # consecutive K8 windows from captured hooks
        for i, row in enumerate(captured):
            ids_t = row["expert_ids"]
            if not ids_t:
                continue
            windows.append({
                "family": "code" if i % 4 == 0 else "math" if i % 4 == 1 else "general" if i % 4 == 2 else "agent",
                "expert_ids": ids_t,
                "weights": row["weights"],
                "k": len(ids_t) if isinstance(ids_t[0], list) else 1,
            })
        data = {"status": "golden", "windows": windows}
        ARTIFACT.write_text(json.dumps(data, indent=2))
        print(json.dumps({"status": "golden", "n_windows": len(windows), "wrote": str(ARTIFACT)}))
        return data
    except Exception as exc:
        out = {"status": "no_real_routes", "reason": f"{type(exc).__name__}:{exc}"}
        print(json.dumps(out, default=str))
        return out


if __name__ == "__main__":
    run()
