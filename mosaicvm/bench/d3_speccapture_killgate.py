"""SpecCapture-Exact kill gate (shadow, not SCALE1000).

Replay 20 already-captured SOURCE_CLASS trajectories with a
draft-K verify loop. Drafter never writes labels.

Kill criteria (every prompt, every K in {2,4,8}):
  tokens_new == tokens_old
  torch.equal(h40_new, h40_old)
  torch.equal(h42_new, h42_old)

Fail-closed GPU rule (permanent while a recorder is live):
  - never import stage1_record
  - never inherit torch.set_default_device("cuda")
  - never init CUDA if any stage1_record.py process exists
  --run-gpu checks the recorder BEFORE importing torch
  --prepare-gold-only forces CUDA_VISIBLE_DEVICES="" then
    loads tensors on CPU only

Do not enable for SCALE1000 capture until this gate PASSes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

GOLD_DIR = Path("artifacts/d3_scale700_data")
GOLD_MANIFEST = Path("artifacts/d3_scale700/prompts_scale700.json")
OUT = Path("artifacts/d3_speccapture/killgate.json")
K_GRID = (2, 4, 8)
N_PROMPTS = 20
SNAP_NAME = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"


def recorder_alive() -> bool:
    """Generic: any stage1_record.py, not a hardcoded PID."""
    r = subprocess.run(["pgrep", "-f", "stage1_record.py"],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def recorder_pids() -> list[str]:
    r = subprocess.run(["pgrep", "-f", "stage1_record.py"],
                       capture_output=True, text=True)
    return [p for p in r.stdout.split() if p]


def gold_hash(torch, c) -> dict:
    h40 = c["h40"].contiguous().cpu()
    h42 = c["h42"].contiguous().cpu()
    return {
        "category": c["category"],
        "split": c["split"],
        "S": int(c["S"]),
        "n_tokens": len(c["tokens"]),
        "tokens_sha256": hashlib.sha256(
            json.dumps(c["tokens"]).encode()).hexdigest(),
        "h40_sha256": hashlib.sha256(
            h40.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "h42_sha256": hashlib.sha256(
            h42.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "h40_shape": list(h40.shape),
        "h42_shape": list(h42.shape),
        "snapshot_revision": c["snapshot_revision"],
        "teacher_class": c["teacher_class"],
        "capture_sha256": hashlib.sha256(
            Path(c["capture_path"]).read_bytes()).hexdigest()
        if "capture_path" in c else None,
    }


def prepare_gold(n=N_PROMPTS):
    """CPU-only. Hash n existing SCALE700 captures. No CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, ".")
    sys.path.insert(0, "mosaicvm/bench")
    import torch
    torch.set_default_device("cpu")
    spec = json.loads(GOLD_MANIFEST.read_text())
    from mosaicvm.bench.d3_student import load_captures
    from mosaicvm.bench.d3_v1 import HORIZON
    caps = load_captures(str(GOLD_DIR), spec, None, HORIZON,
                         require_both=False)
    caps = sorted(caps, key=lambda c: (c["split"] != "val",
                                       c["category"]))
    chosen = caps[:n]
    rows = [gold_hash(torch, c) for c in chosen]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classification": "SPECCAPTURE_GOLD_HASHES",
        "n": len(rows),
        "source_dir": str(GOLD_DIR),
        "k_grid": list(K_GRID),
        "gate": "tokens + h40 + h42 bitwise vs these hashes",
        "gpu_run": "BLOCKED_UNTIL_RECORDER_IDLE",
        "prompts": rows,
    }
    path = OUT.parent / "gold_hashes.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path, payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-gold-only", action="store_true")
    p.add_argument("--run-gpu", action="store_true",
                   help="Load source model and replay. Refuses "
                        "BEFORE torch import if a recorder lives.")
    a = p.parse_args()

    if a.run_gpu:
        pids = recorder_pids()
        if pids:
            print("REFUSE: stage1_record.py alive "
                  f"pids={pids}. No torch import. "
                  "SpecCapture GPU replay would steal the "
                  "3060 from SCALE1000 capture.")
            raise SystemExit(2)
        print("GPU replay not implemented in this drop. "
              "Gold hashes + refuse-if-busy gate only. "
              "Implement SpecCapture loop after SCALE1000 "
              "releases the GPU, then require bitwise "
              "equality against gold_hashes.json.")
        raise SystemExit(3)

    path, payload = prepare_gold()
    print(f"wrote {path}: {payload['n']} gold hashes")
    print("GPU replay not started (--run-gpu required).")
    pids = recorder_pids()
    if pids:
        print(f"recorder ALIVE pids={pids} — GPU gate would refuse")
    else:
        print("recorder idle — GPU gate would allow")


if __name__ == "__main__":
    main()
