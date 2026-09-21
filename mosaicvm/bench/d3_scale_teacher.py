"""SCALE210 teacher generator: gold-only SourceTail replay for
the 126 new training prompts (roadmap S19, S23-S24).

Produces records with EXACTLY the fields the HIST/WS trainers
consume (build_hist_examples interface):
    {category, S, K, tokens, decisions[{t, true_token,
     gold{top_idx, top_val, true_logit, true_rank,
     margin_vs_top_comp}, gold_greedy}]}

Provenance: replay is deterministic — verified bitwise against
the existing d3_r2_teacher h42_gold_future (see
session evidence). os/chain variants are NOT recorded (not
consumed by HIST/WS training).

Also writes a hash-bound audit: per-record SHA-256, prompt
text SHA, capture SHA, overlap checks.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.source_tail import SourceTail
from mosaicvm.bench.d3_v1 import HORIZON

MANIFEST = Path("artifacts/d3_scale/prompts_scale.json")
CAP_DIR = Path("artifacts/d3_scale_data")
OUT_DIR = Path("artifacts/d3_scale_teacher")


def topk_pack(logits, true_tok, k=16):
    v, i = logits.topk(k)
    tl = float(logits[true_tok])
    rank = int((logits > tl).sum())
    top_comp = int(i[0]) if int(i[0]) != true_tok \
        else int(i[1])
    margin = tl - float(logits[top_comp])
    return {"top_idx": i.tolist(),
            "top_val": [round(float(x), 4) for x in v],
            "true_logit": round(tl, 4), "true_rank": rank,
            "margin_vs_top_comp": round(margin, 4)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    spec = json.loads(MANIFEST.read_text())
    prompts = {p["category"]: p["prompt"]
               for p in spec["prompts"]}
    train_cats = spec["splits"]["train"]

    # overlap audit inputs
    old = json.loads(
        Path("artifacts/stage1_prompts.json").read_text())
    old_ids = {p["category"] for p in old["prompts"]}
    legacy_val = set(old["splits"]["val"])
    scale_val = set(spec["splits"]["val"])

    missing = [c for c in train_cats
               if not (CAP_DIR / f"{c}.pt").exists()]
    if missing:
        print(f"MISSING CAPTURES ({len(missing)}): "
              f"{missing[:5]}...")
        raise SystemExit(1)

    tail = SourceTail(device="cuda", backend="reference",
                      max_seq=64, cache_cap=8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    audit = {"classification": "D3_SCALE_TEACHER",
             "n": len(train_cats), "records": {},
             "overlap_checks": {
                 "train_vs_old_manifest":
                     sorted(set(train_cats) & old_ids),
                 "train_vs_legacy_val":
                     sorted(set(train_cats) & legacy_val),
                 "train_vs_scale_val":
                     sorted(set(train_cats) & scale_val),
                 "all_empty_required": True}}
    for cat in train_cats:
        c = torch.load(CAP_DIR / f"{cat}.pt",
                       map_location="cpu",
                       weights_only=True)
        assert c["category"] == cat \
            and c["split"] == "train"
        S = c["S"]
        K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]
        h42 = tail.replay(c["h40"], toks, S)
        rows = []
        for j in range(K):
            t = S + j
            true_tok = toks[t + 1]
            lg = tail.logits(h42[t:t + 1])[0]
            rows.append({"t": t, "true_token": true_tok,
                         "gold": topk_pack(lg, true_tok),
                         "gold_greedy": int(lg.argmax())})
        rec = {"category": cat, "S": S, "K": K,
               "tokens": toks[:S + K + 1], "decisions": rows}
        torch.save(rec, OUT_DIR / f"{cat}.pt")
        h = hashlib.sha256()
        h.update(cat.encode())
        h.update((OUT_DIR / f"{cat}.pt").read_bytes())
        audit["records"][cat] = {
            "record_sha256": h.hexdigest(),
            "capture_sha256": sha(CAP_DIR / f"{cat}.pt"),
            "prompt_sha256": hashlib.sha256(
                prompts[cat].encode()).hexdigest(),
            "S": S, "K": K}
        print(f"{cat}: S={S} K={K} "
              f"({time.time()-t0:.0f}s)", flush=True)

    for k, v in audit["overlap_checks"].items():
        if k != "all_empty_required" and v:
            audit["overlap_checks"][
                "all_empty_required"] = False
    audit["manifest_sha256"] = sha(MANIFEST)
    (OUT_DIR / "audit.json").write_text(
        json.dumps(audit, indent=1))
    print(f"\noverlap checks empty: "
          f"{audit['overlap_checks']['all_empty_required']}")
    print(f"wrote {OUT_DIR}/audit.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
