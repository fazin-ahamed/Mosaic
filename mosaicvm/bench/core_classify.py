"""Phase 2.11B (§13): classify core tensors and compute B_core-resident.

Reads .mosaicvm/core/core.safetensors, categorizes every tensor into the
roadmap's residency classes, and reports the byte cost of each class plus
the implied expert-cache budget.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import struct

CORE_PATH = Path(".mosaicvm/core/core.safetensors")
ARTIFACT = Path("artifacts/core_classification.json")

DT_SIZES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I64": 8}


def classify(name: str) -> str:
    """Map a tensor name to its residency class (roadmap §13)."""
    if name.startswith("mtp."):
        return "MTP_DSPARK_OPTIONAL"
    if name == "head.weight" or name.startswith("head."):
        return "LM_HEAD"
    if name == "embed.weight":
        return "EMBED_HOST"  # RAM, not VRAM (per core_inventory)
    if ".ffn.gate." in name or ".ffn.gate." in name:
        return "ROUTER"  # gate weight + e_score_correction_bias per layer
    if ".ffn.shared_experts." in name:
        return "SHARED_EXPERT"
    if ".attn." in name or ".norm." in name or name.startswith("norm."):
        return "TARGET_ALWAYS_RESIDENT"
    if name.startswith("hc_head"):
        return "TARGET_ALWAYS_RESIDENT"
    # hash router tensors (tid2eid, bias for hash layers)
    if ".ffn.hash" in name or "tid2eid" in name:
        return "ROUTER"
    return "TARGET_ALWAYS_RESIDENT"


def run() -> dict:
    with open(CORE_PATH, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))

    entries = {k: v for k, v in hdr.items() if k != "__metadata__"}

    rows = []
    by_class = defaultdict(lambda: {"count": 0, "bytes": 0, "tensors": []})
    for name, e in entries.items():
        numel = math.prod(e["shape"])
        nbytes = numel * DT_SIZES[e["dtype"]]
        cls = classify(name)
        by_class[cls]["count"] += 1
        by_class[cls]["bytes"] += nbytes
        by_class[cls]["tensors"].append(name)
        rows.append({
            "name": name, "dtype": e["dtype"], "shape": e["shape"],
            "bytes": nbytes, "class": cls,
        })

    total = sum(c["bytes"] for c in by_class.values())
    embed = by_class.get("EMBED_HOST", {"bytes": 0})["bytes"]
    mtp = by_class.get("MTP_DSPARK_OPTIONAL", {"bytes": 0})["bytes"]
    lm_head = by_class.get("LM_HEAD", {"bytes": 0})["bytes"]

    # B_core-resident: everything that must be in VRAM for the target path
    # (excludes EMBED_HOST which lives in RAM, and MTP/DSPARK which is
    # optional for the target verifier)
    b_resident = total - embed - mtp

    # with MTP/DSPark (for speculation later)
    b_resident_with_mtp = b_resident + mtp

    # the real expert-cache budget (12 GB VRAM, minus core, minus KV/workspace
    # estimates from floors.py: KV at 4K context ≈ 43 layers × 1 head × 4096
    # ctx × 512 dim × 2 (K+V) × 2 bytes = ~700 MB; workspace ~500 MB;
    # safety margin ~500 MB)
    KV_4K = 43 * 1 * 4096 * 512 * 2 * 2  # bytes
    WORKSPACE = 500 * 1024 * 1024
    SAFETY = 500 * 1024 * 1024
    VRAM = 12 * 1024**3
    expert_budget = VRAM - b_resident - KV_4K - WORKSPACE - SAFETY

    out = {
        "phase": "2.11B/§13",
        "core_path": str(CORE_PATH),
        "total_tensors": len(entries),
        "total_bytes": total,
        "by_class": {
            cls: {"count": v["count"], "bytes": v["bytes"],
                  "gb": v["bytes"] / 1e9,
                  "example_tensors": v["tensors"][:5]}
            for cls, v in sorted(by_class.items(), key=lambda x: -x[1]["bytes"])
        },
        "b_core_total_gb": total / 1e9,
        "b_core_resident_gb": b_resident / 1e9,
        "b_core_resident_with_mtp_gb": b_resident_with_mtp / 1e9,
        "embed_host_gb": embed / 1e9,
        "mtp_dspark_gb": mtp / 1e9,
        "lm_head_gb": lm_head / 1e9,
        "expert_cache_budget": {
            "vram_gb": VRAM / 1e9,
            "core_resident_gb": b_resident / 1e9,
            "kv_4k_gb": KV_4K / 1e9,
            "workspace_gb": WORKSPACE / 1e9,
            "safety_gb": SAFETY / 1e9,
            "expert_cache_gb": expert_budget / 1e9,
            "note": "Expert cache = VRAM - core - KV(4K) - workspace - safety. "
                    "KV grows with context; workspace/safety are provisional.",
        },
        "rows": rows,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "total_gb": round(total / 1e9, 2),
        "b_core_resident_gb": round(b_resident / 1e9, 2),
        "b_with_mtp_gb": round(b_resident_with_mtp / 1e9, 2),
        "embed_host_gb": round(embed / 1e9, 2),
        "mtp_gb": round(mtp / 1e9, 2),
        "lm_head_gb": round(lm_head / 1e9, 2),
        "expert_cache_gb": round(expert_budget / 1e9, 2),
        "classes": {cls: round(v["bytes"] / 1e9, 2)
                    for cls, v in sorted(by_class.items(), key=lambda x: -x[1]["bytes"])},
    }, indent=2))
    return out


if __name__ == "__main__":
    run()
