"""R2-1 (roadmap S31): downstream teacher dataset generation.

For every TRAIN prompt, generate and store:
  - gold h40 trajectory (committed + future window)
  - A2 one-step predicted h40 (gold prev state + true driver)
  - A2 chained predicted h40 (its own trajectory, true tokens)
  - gold h42 (from the capture)
  - exact-tail h42 of BOTH predicted trajectories (GPU replay)
  - teacher FP32 logits (gold + both predicted), compressed to
    top-16 (idx, vals) + true-token logit + rank + margin
  - tokens, position, domain

This is the canonical offline teacher/diagnostic dataset for the
Decision Bridge (S32-S35), Tail-Jacobian VJPs (S28-S30), and the
gradient-conflict diagnostics (S27). No training happens here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_v1 import (build_transitions, load_embed,
                                  make_feat, DeltaNet, D_STATE,
                                  HORIZON)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.source_tail import SourceTail

OUT_DIR = Path("artifacts/d3_r2_teacher")
STUDENT = Path("artifacts/d3_v1_A2/student.pt")


def load_student(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = DeltaNet(ck["hidden"], ck["depth"], ck["rank"],
                     aux_out=ck.get("h42_aux_weight", 0) > 0
                     or "aux.weight" in ck["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    return (model, ck["target_mean"], ck["target_std"],
            ck["x_mean"], ck["x_std"])


@torch.no_grad()
def rollout(model, embed, c, mode, K, mean, std, x_mean, x_std):
    """mode 'one_step' (gold prev) or 'chain' (own trajectory);
    true-token driver conditioning throughout."""
    S = c["S"]
    toks = c["tokens"]
    h = c["h40"].float()
    state = h[S - 1].flatten()
    prev = h[S - 2].flatten()
    out = []
    for j in range(K):
        t = S + j
        if mode == "one_step":
            state_in = h[t - 1].flatten()
            delta_in = h[t - 1].flatten() - h[t - 2].flatten()
            cur_in = toks[t - 1]
        else:
            state_in = state
            delta_in = state - prev
            cur_in = toks[t - 1]
        feat = make_feat(embed, state_in, delta_in, toks[t],
                         cur_in, t)
        feat = (feat - x_mean) / x_std
        pred = model(feat.unsqueeze(0))[0] * std + mean
        new_state = state_in + pred
        out.append(new_state)
        prev = state_in
        state = new_state
    return (torch.stack(out).view(K, 4, 4096)
            .to(torch.bfloat16))                    # [K, 4, 4096]


def topk_pack(logits, true_tok, k=16):
    v, i = logits.topk(k)
    tl = float(logits[true_tok])
    rank = int((logits > tl).sum())
    top_comp = int(i[0]) if int(i[0]) != true_tok else int(i[1])
    margin = tl - float(logits[top_comp])
    return {"top_idx": i.tolist(), "top_val":
            [round(float(x), 4) for x in v],
            "true_logit": round(tl, 4), "true_rank": rank,
            "margin_vs_top_comp": round(margin, 4)}


def main():
    t0 = time.time()
    model, mean, std, x_mean, x_std = load_student(STUDENT)
    embed = load_embed()
    captures = load_captures("artifacts/d3_data_105",
                             json.loads(Path(
                                 "artifacts/stage1_prompts.json"
                             ).read_text()), None, HORIZON)
    train = [c for c in captures if c["split"] == "train"]
    print(f"{len(train)} train prompts", flush=True)

    tail = SourceTail(device="cuda", backend="reference",
                      max_seq=64, cache_cap=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in train:
        S = c["S"]
        K = min(HORIZON, c["h40"].shape[0] - S)
        toks = c["tokens"]
        h_gold = c["h40"]                      # bf16 [S+K, 4, 4096]

        # A2 predicted trajectories (CPU)
        p_os = rollout(model, embed, c, "one_step", K, mean, std,
                       x_mean, x_std)
        p_ch = rollout(model, embed, c, "chain", K, mean, std,
                       x_mean, x_std)

        # exact-tail replays (GPU): gold + both predicted
        def replay(pred):
            exact = torch.cat([h_gold[:S], pred])
            return tail.replay(exact, toks, S)

        h42_gold_tr = replay(h_gold[S:S + K])   # == capture h42
        h42_os = replay(p_os)
        h42_ch = replay(p_ch)

        # teacher logits per position per variant
        rows = []
        for j in range(K):
            t = S + j
            true_tok = toks[t + 1]
            row = {"t": t, "true_token": true_tok}
            for tag, h42_tr in (("gold", h42_gold_tr),
                                ("os", h42_os), ("ch", h42_ch)):
                lg = tail.logits(h42_tr[S + j:S + j + 1])[0]
                row[tag] = topk_pack(lg, true_tok)
                row[tag + "_greedy"] = int(lg.argmax())
            rows.append(row)

        rec = {
            "category": c["category"], "S": S, "K": K,
            "tokens": toks[:S + K + 1],
            "h40_gold_future": h_gold[S:S + K].clone(),
            "h40_pred_os": p_os, "h40_pred_chain": p_ch,
            "h42_gold_future": h42_gold_tr[S:].clone(),
            "h42_tail_os": h42_os[S:].clone(),
            "h42_tail_chain": h42_ch[S:].clone(),
            "decisions": rows,
        }
        torch.save(rec, OUT_DIR / f"{c['category']}.pt")
        manifest.append(c["category"])
        print(f"{c['category']} ({time.time()-t0:.0f}s)",
              flush=True)

    (OUT_DIR / "manifest.json").write_text(json.dumps(
        {"n_prompts": len(manifest), "categories": manifest,
         "student": str(STUDENT),
         "variants": ["gold", "os(one-step)", "ch(chain)"],
         "contents": "h40 gold/pred(os,chain), h42 gold + "
                     "tail-of-predicted, top-16 logits, true "
                     "rank/margin per position"}, indent=1))
    print(f"wrote {len(manifest)} teacher records "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
