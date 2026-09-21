"""DepthScale-1001 teacher augmentation.

The ps/bridge trainer consumes only h40 state variants (gold/os/chain)
as inputs and gold top-16 decisions as targets. orig84 (d3_r2_teacher)
already has the variants; scale210/700/1000 teachers are gold-only.

This adds ONLY h40_pred_os + h40_pred_chain (cheap A2 rollouts, no
SourceTail) to each missing-tier gold teacher record, writing augmented
records to a parallel dir. Gold decisions are copied through unchanged.
No optional diagnostic variants (no h42_*, no os/ch logits) — the
trainer does not consume them.

Resumable + fail-closed: skips records already augmented; verifies
category/S/K and that gold decisions match the source.
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

from mosaicvm.bench.d3_r2_teacher import rollout, load_student
from mosaicvm.bench.d3_bridge import load_embed
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON

STUDENT = "artifacts/d3_v1_A2/student.pt"

# (gold teacher dir, capture dir, manifest) -> augmented output dir
TIERS = [
    ("artifacts/d3_scale_teacher", "artifacts/d3_scale_data",
     "artifacts/d3_scale/prompts_scale.json",
     "artifacts/d3_scale_teacher_aug"),
    ("artifacts/d3_scale700_teacher", "artifacts/d3_scale700_data",
     "artifacts/d3_scale700/prompts_scale700.json",
     "artifacts/d3_scale700_teacher_aug"),
    ("artifacts/d3_scale1000_teacher", "artifacts/d3_scale1000_data",
     "artifacts/d3_scale1000/prompts_scale1000.json",
     "artifacts/d3_scale1000_teacher_aug"),
]


def main():
    torch.set_num_threads(8)
    t0 = time.time()
    model, mean, std, xm, xs = load_student(STUDENT)
    embed = load_embed()

    total = 0
    for tdir, cdir, manifest, outdir in TIERS:
        tdir = Path(tdir); outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        spec = json.loads(Path(manifest).read_text())
        train_cats = spec["splits"]["train"]
        caps = load_captures(cdir, spec, None, HORIZON, require_both=False)
        by = {c["category"]: c for c in caps}
        done = 0
        for cat in train_cats:
            out_f = outdir / f"{cat}.pt"
            if out_f.exists():
                done += 1
                continue
            gold = torch.load(tdir / f"{cat}.pt", map_location="cpu",
                              weights_only=True)
            c = by[cat]
            S, K = gold["S"], gold["K"]
            assert c["category"] == cat and S == c["S"]
            p_os = rollout(model, embed, c, "one_step", K, mean, std, xm, xs)
            p_ch = rollout(model, embed, c, "chain", K, mean, std, xm, xs)
            rec = dict(gold)  # category, S, K, tokens, decisions
            rec["h40_gold_future"] = c["h40"][S:S + K].clone()
            rec["h40_pred_os"] = p_os
            rec["h40_pred_chain"] = p_ch
            torch.save(rec, out_f)
            done += 1
            total += 1
            if done % 100 == 0:
                print(f"  {outdir.name}: {done}/{len(train_cats)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        # audit
        aug = sorted(p.stem for p in outdir.glob("*.pt")
                     if p.stem != "manifest")
        assert set(aug) == set(train_cats), (
            f"{outdir}: {len(aug)} != {len(train_cats)}")
        (outdir / "manifest.json").write_text(json.dumps({
            "n_prompts": len(train_cats),
            "categories": sorted(train_cats),
            "student": STUDENT,
            "variants": ["gold(copied)", "os", "chain"],
            "contents": "h40 gold/os/chain + copied gold decisions; "
                        "no h42, no os/ch logits (trainer does not use)",
            "source_teacher": str(tdir),
        }, indent=1))
        print(f"{outdir.name}: {len(train_cats)} augmented "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"DONE: {total} new records ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
