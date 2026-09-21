"""Static top-k quality sweep vs k=6 golden.

Remote H100 run lives in modal_app.topk_quality_sweep. This module holds the
SWEEP constant and a local blocked-stub writer used when Modal is unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path

from mosaicvm.bench.prompts import GOLDEN_PROMPTS

SWEEP = (6, 5, 4, 3, 2)
TURBO_DEFAULT_UNVALIDATED = 4


def write_blocked_stub(path: str | Path = "artifacts/topk_quality.json") -> dict:
    payload = {
        "status": "blocked_no_modal",
        "reason": "Modal token is the leaked Phase-1 credential; remote H100 sweep skipped",
        "sweep": list(SWEEP),
        "prompts": list(GOLDEN_PROMPTS),
        "turbo_default": TURBO_DEFAULT_UNVALIDATED,
        "turbo_default_quality": "unvalidated",
        "fidelity_k": 6,
        "pass_criterion": "mean_token_match >= 0.90 vs k=6 golden",
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    print(json.dumps(write_blocked_stub(), indent=2))
