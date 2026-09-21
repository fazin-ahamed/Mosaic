"""SCALE1000 teacher: gold-only SourceTail replay for the
301 new training prompts.

Identical contract to d3_scale700_teacher.py with SCALE1000
paths. Overlap audit includes SCALE-DEV3.
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

MANIFEST = Path(
    "artifacts/d3_scale1000/prompts_scale1000.json")
CAP_DIR = Path("artifacts/d3_scale1000_data")
OUT_DIR = Path("artifacts/d3_scale1000_teacher")
DEV3 = Path("artifacts/d3_scale1000/prompts_scale_dev3.json")
DEV3_SHA = (
    "41c5b929560e4605cb66ce139fef761b136d4dad"
    "7be3c0fbddac4ac18718eacf"
)


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
    assert sha(DEV3) == DEV3_SHA, "SCALE-DEV3 mutated"
    spec = json.loads(MANIFEST.read_text())
    prompts = {p["category"]: p["prompt"]
               for p in spec["prompts"]}
    train_cats = spec["splits"]["train"]
    val_cats = set(spec["splits"]["val"])
    assert len(train_cats) == 301
    assert len(val_cats) == 70

    prior_ids, prior_texts = set(), set()
    for mf in ("artifacts/stage1_prompts.json",
               "artifacts/stage05_prompts.json",
               "artifacts/d3_scale/prompts_scale.json",
               "artifacts/d3_scale700/prompts_scale700.json",
               "artifacts/d3_scale1000/prompts_scale_dev3.json"):
        d = json.loads(Path(mf).read_text())
        for p in d["prompts"]:
            prior_ids.add(p["category"])
            prior_texts.add(" ".join(
                p["prompt"].lower().split()))

    missing = [c for c in train_cats
               if not (CAP_DIR / f"{c}.pt").exists()]
    if missing:
        print(f"MISSING CAPTURES ({len(missing)}): "
              f"{missing[:5]}...")
        raise SystemExit(1)

    tail = SourceTail(device="cuda", backend="reference",
                      max_seq=128, cache_cap=8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    audit = {"classification": "D3_SCALE1000_TEACHER",
             "n": len(train_cats), "records": {},
             "overlap_checks": {
                 "train_vs_prior_manifests":
                     sorted(set(train_cats) & prior_ids),
                 "train_texts_vs_prior":
                     sorted({" ".join(prompts[c].lower()
                                      .split())
                             for c in train_cats}
                            & prior_texts),
                 "train_vs_scale_dev3":
                     sorted(set(train_cats) & val_cats),
                 "all_empty_required": True}}
    for cat in train_cats:
        out_file = OUT_DIR / f"{cat}.pt"
        if out_file.exists():
            rec = torch.load(out_file, map_location="cpu", weights_only=True)
            S, K = rec["S"], rec["K"]
        else:
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
            torch.save(rec, out_file)
        h = hashlib.sha256()
        h.update(cat.encode())
        h.update((OUT_DIR / f"{cat}.pt").read_bytes())
        audit["records"][cat] = {
            "record_sha256": h.hexdigest(),
            "capture_sha256": sha(CAP_DIR / f"{cat}.pt"),
            "prompt_sha256": hashlib.sha256(
                prompts[cat].encode()).hexdigest(),
            "S": S, "K": K}
        if len(audit["records"]) % 25 == 0:
            print(f"{len(audit['records'])}/{len(train_cats)}"
                  f" ({time.time()-t0:.0f}s)", flush=True)

    for k, v in audit["overlap_checks"].items():
        if k != "all_empty_required" and v:
            audit["overlap_checks"][
                "all_empty_required"] = False
    audit["manifest_sha256"] = sha(MANIFEST)
    audit["dev3_sha256"] = sha(DEV3)
    (OUT_DIR / "audit.json").write_text(
        json.dumps(audit, indent=1))
    print(f"\noverlap checks empty: "
          f"{audit['overlap_checks']['all_empty_required']}")
    print(f"wrote {OUT_DIR}/audit.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
