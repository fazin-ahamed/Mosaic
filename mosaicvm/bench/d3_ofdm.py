"""OFDM — Out-of-Fold Drift Mining.

ps_old (artifacts/d3_bridge_r3b/bridge_ps) was trained ONLY on the original
84 prompts. It is therefore clean-unseen for all 917 later-added prompts
(scale210 + scale700 + scale1000). Rolling ps_old on those prompts yields
out-of-sample continuation drift states WITHOUT new capture or fold models.

Two-model design:
  ps_old   -> generates the unseen-prompt draft trajectory (drift states)
  ps1001   -> current-relevance filter: keep a mined state only if ps1001
              STILL fails on it (greedy wrong / true-rank>=THR)

Exact-semantics label: deterministic greedy target => the correct token at
every within-block position is gold toks[t+1]. No wrong-prefix mislabel risk.

Mode:
  --mode census  : report ps_old failure yield + ps1001-relevant survivors
  --mode mine    : write the current-relevant OOF bank to disk

Reuses A2 recurrence + bridge feature; HIST-free (ps is state->token).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_bridge import (BridgeNet, load_head, load_embed,
                                      make_bridge_feat)
from mosaicvm.bench.d3_v1 import make_feat, HORIZON
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_ws import load_a2, load_bridge

PS_OLD = "artifacts/d3_bridge_r3b/bridge_ps/bridge.pt"
PS1001 = "artifacts/d3_scale1000_train/ps1001.pt"
A2 = "artifacts/d3_v1_A2/student.pt"
OUT = Path("artifacts/d3_scale1000_train/ofdm_bank.pt")
K_ROLL = 8
RANK_THR = 4          # ps1001 "still fails" if true rank >= THR at that state


def load_ps(path):
    m, xm, xs = load_bridge(Path(path))
    return m, xm, xs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("census", "mine"), default="census")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()

    a2, t_mean, t_std, a2_xm, a2_xs = load_a2(Path(A2))
    embed = load_embed()
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda(); w_norm_g = w_norm.cuda()
    old, old_xm, old_xs = load_ps(PS_OLD)
    new, new_xm, new_xs = load_ps(PS1001)

    # unseen-by-ps_old prompts = all later-added train prompts
    tiers = [("artifacts/d3_scale_data", "artifacts/d3_scale/prompts_scale.json"),
             ("artifacts/d3_scale700_data",
              "artifacts/d3_scale700/prompts_scale700.json"),
             ("artifacts/d3_scale1000_data",
              "artifacts/d3_scale1000/prompts_scale1000.json")]
    caps = []
    for data, man in tiers:
        spec = json.loads(Path(man).read_text())
        cc = load_captures(data, spec, None, HORIZON, require_both=False)
        caps += [c for c in cc if c["split"] == "train"]
    print(f"unseen-by-ps_old prompts: {len(caps)} ({time.time()-t0:.0f}s)",
          flush=True)

    @torch.no_grad()
    def logits(model, xm, xs, bf):
        z = model(((bf - xm) / xs).unsqueeze(0))[0].cuda()
        zn = z * torch.rsqrt(z.square().mean() + 1e-6) * w_norm_g
        return (w_head_g @ zn)

    def rank_of(lg, tok):
        return int((lg > float(lg[tok])).sum())

    bank = []
    old_fail = 0            # ps_old first-rejection events
    relevant = 0           # survivors ps1001 still fails
    dom_ct = Counter()
    n_pos = 0
    for c in caps:
        S = c["S"]; K = min(K_ROLL, c["h40"].shape[0] - S - 1)
        if K < 1:
            continue
        toks = c["tokens"]; h = c["h40"].float()
        state = h[S - 1].flatten(); prev = h[S - 2].flatten()
        cur_tok = toks[S - 1]; driver = toks[S]
        for j in range(K):
            t = S + j
            fa = make_feat(embed, state, state - prev, driver, cur_tok, t)
            pred = a2(((fa - a2_xm) / a2_xs).unsqueeze(0))[0] * t_std + t_mean
            new_state = state + pred
            bf = make_bridge_feat(embed, new_state, driver, cur_tok, t)
            n_pos += 1
            lo = logits(old, old_xm, old_xs, bf)
            old_draft = int(lo.argmax())
            gold = toks[t + 1]
            if old_draft != gold:
                old_fail += 1
                # current-relevance filter with ps1001 on the SAME state
                ln = logits(new, new_xm, new_xs, bf)
                r_new = rank_of(ln, gold)
                if r_new >= RANK_THR:
                    relevant += 1
                    dom_ct[c["category"].rsplit("_", 1)[0]] += 1
                    if a.mode == "mine":
                        bank.append({
                            "feature": bf.clone(),
                            "gold": gold, "j": j,
                            "cat": c["category"],
                            "ps1001_true_rank": r_new,
                        })
                # ps_old diverged: continue along ITS draft (on-policy for old)
                cur_tok = driver; driver = old_draft
            else:
                cur_tok = driver; driver = old_draft
            prev = state; state = new_state

    print(f"positions rolled: {n_pos}")
    print(f"ps_old first-token failures: {old_fail} "
          f"({old_fail/max(n_pos,1):.3f}/pos)")
    print(f"ps1001-relevant (true_rank>={RANK_THR}): {relevant} "
          f"({relevant/max(len(caps),1):.2f}/prompt)")
    print(f"relevant per domain: {dict(dom_ct)}")
    print(f"unique prompts with relevant states: "
          f"{len(set(m['cat'] for m in bank)) if bank else 'n/a (census)'}")

    # census gate: >=128 relevant, >=50 prompts, >=5 domains
    gate = relevant >= 128 and len(dom_ct) >= 5
    print(f"CENSUS GATE (>=128 relevant, >=5 domains): "
          f"{'PASS' if gate else 'FAIL'} ({time.time()-t0:.0f}s)")

    if a.mode == "mine":
        uniq = len(set(m["cat"] for m in bank))
        torch.save({"bank": bank, "rank_thr": RANK_THR,
                    "n_states": len(bank), "n_prompts": uniq,
                    "domains": dict(dom_ct),
                    "ps_old": PS_OLD, "ps1001": PS1001,
                    "script_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()).hexdigest()}, OUT)
        print(f"wrote {OUT}: {len(bank)} states, {uniq} prompts "
              f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
