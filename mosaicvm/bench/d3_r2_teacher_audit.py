"""R2-1 teacher-dataset audit: canonical, hash-bound, fail-closed.

Canonical requirements (roadmap S26):
  - FULL SHA-256 (no prefixes) for every record, capture, the A2
    checkpoint, the generator script (d3_r2_teacher.py), the
    auditor, and the source snapshot identity
  - dataset-global aggregate hash + total byte size
  - n_records must equal 84; positions 2688
  - records_passed accounts for h42 AND greedy failures
  - any drift from a previous audit fails closed
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

TEACHER = Path("artifacts/d3_r2_teacher")
CAPS = Path("artifacts/d3_data_105")
AUDIT = Path("artifacts/d3_r2_teacher_audit.json")
A2 = Path("artifacts/d3_v1_A2/student.pt")
GENERATOR = Path("mosaicvm/bench/d3_r2_teacher.py")
SNAPSHOT = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--"
    "DeepSeek-V4-Flash-0731/snapshots/"
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062")
EXPECT_N = 84
EXPECT_POS = 2688


def sha_full(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    files = sorted(TEACHER.glob("*.pt"))
    prev = json.loads(AUDIT.read_text()) if AUDIT.exists() else None
    prev_hashes = prev.get("record_hashes_full", {}) if prev else {}

    rows, drift, h42_fail, greedy_fail = [], [], [], []
    greedy_hits_total, total_pos = 0, 0
    total_bytes = 0
    rec_hashes = {}

    for f in files:
        rec_h = sha_full(f)
        total_bytes += f.stat().st_size
        rec_hashes[f.stem] = rec_h
        if f.stem in prev_hashes and prev_hashes[f.stem] != rec_h:
            drift.append(f.stem)
        cap_h = sha_full(CAPS / f.name)
        rec = torch.load(f, map_location="cpu", weights_only=True)
        cap = torch.load(CAPS / f.name, map_location="cpu",
                         weights_only=True)
        S, K = rec["S"], rec["K"]
        ok_h42 = torch.equal(rec["h42_gold_future"].view(torch.int16),
                             cap["h42"][S:S + K].view(torch.int16))
        if not ok_h42:
            h42_fail.append(f.stem)
        g_ok = 0
        for row in rec["decisions"]:
            total_pos += 1
            if row["gold_greedy"] == row["true_token"]:
                g_ok += 1
                greedy_hits_total += 1
            else:
                greedy_fail.append((f.stem, row["t"]))
        rows.append({"category": f.stem, "positions": K,
                     "capture_sha256": cap_h,
                     "teacher_sha256": rec_h,
                     "h42_bit_equal": bool(ok_h42),
                     "greedy_correct": f"{g_ok}/{K}"})

    # dataset-global aggregate: hash of the sorted name:hash lines
    agg = hashlib.sha256("\n".join(
        f"{k}:{v}" for k, v in sorted(rec_hashes.items())
    ).encode()).hexdigest()
    # source snapshot identity: hash the config + index (weight
    # shards are the pinned hub snapshot; identity = revision dir
    # contents manifest via safetensors index + config hashes)
    snap_id = hashlib.sha256(
        (SNAPSHOT / "model.safetensors.index.json").read_bytes()
        + (SNAPSHOT / "config.json").read_bytes()
    ).hexdigest()

    n = len(rows)
    passed = (n - len(set(drift)) - len(set(h42_fail))
              - len({g for g, _ in greedy_fail}))
    passed_ok = (n == EXPECT_N and total_pos == EXPECT_POS
                 and not drift and not h42_fail
                 and not greedy_fail
                 and greedy_hits_total == total_pos)
    audit = {
        "classification": "D3_R2_TEACHER_AUDIT_CANONICAL",
        "verdict": "PASS" if passed_ok else "FAIL",
        "fail_closed": "any hash drift, check failure, or count "
                       "mismatch fails the audit",
        "n_records": n, "expected_records": EXPECT_N,
        "records_passed": f"{passed}/{n}",
        "positions": total_pos, "expected_positions": EXPECT_POS,
        "greedy_correct": f"{greedy_hits_total}/{total_pos}",
        "dataset_total_bytes": total_bytes,
        "dataset_aggregate_sha256": agg,
        "source_snapshot_identity_sha256": snap_id,
        "source_snapshot_name": SNAPSHOT.name,
        "a2_checkpoint_sha256": sha_full(A2),
        "generator_sha256": sha_full(GENERATOR),
        "auditor_sha256": sha_full(Path(__file__)),
        "hash_drift": drift,
        "h42_failures": h42_fail,
        "greedy_failures": greedy_fail[:20],
        "record_hashes_full": rec_hashes,
    }
    AUDIT.write_text(json.dumps(audit, indent=1))
    summary = {k: v for k, v in audit.items()
               if k != "record_hashes_full"}
    print(json.dumps(summary, indent=1))
    sys.exit(0 if passed_ok else 1)


if __name__ == "__main__":
    main()
