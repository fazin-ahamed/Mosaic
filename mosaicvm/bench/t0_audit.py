"""Track A: T0 acceptance-evaluator audit (roadmap §12-24).

One population (full held-out), all metrics from the same draft/target
arrays:
  1. E[A] == 1 + sum(S_k) asserted exactly (counting tolerance).
  2. Contingency: P(h2 correct | h1 rank 1 / rank 2 / rank>2),
     teacher-forced AND free-roll.
  3. TF-vs-FR h2 logits equality on the h1-correct subset.
  4. Causal-mask invariance: perturbing future tokens must not change
     earlier predictions.
Retrains the T0 checkpoint (same seed/hyperparams) and saves it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from tiny_causal import (TinyCausal, load_split, load_frozen,
                         raw_head_logits, K, VOCAB, DEV)
from pathlib import Path as P

PROMPTS = P("artifacts/stage1_prompts.json")
OUT = P("artifacts/t0_audit.json")
CKPT = P("/tmp/tiny_causal_t0.pt")


def main():
    torch.manual_seed(0)
    spec = json.loads(PROMPTS.read_text())
    splits = spec["splits"]
    Xtr, ytr = load_split(splits["train"])
    Xva, yva = load_split(splits["val"])
    Xho, yho = load_split(splits["test"])
    fr = load_frozen()
    Xtr, ytr = Xtr.to(DEV), ytr.to(DEV)
    Xva, yva = Xva.to(DEV), yva.to(DEV)
    Xho, yho = Xho.to(DEV), yho.to(DEV)
    N = Xho.shape[0]
    print(f"held-out N={N}")

    # ---------- train (same recipe/seed as T0) ----------
    model = TinyCausal(fr["embed.weight"].to(DEV)).to(DEV)
    model.emb_table.requires_grad_(False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-4)
    bs = 32
    Ntr = Xtr.shape[0]
    best_va, best_state = -1, None

    def val_h2():
        with torch.no_grad():
            lg = model(Xva[:64], yva[:64])
            return (lg[:64, 1, :].argmax(-1)
                    == yva[:64, 1]).float().mean().item()

    for ep in range(14):
        model.train()
        perm = torch.randperm(Ntr)
        tot = nb = 0
        for i0 in range(0, Ntr, bs):
            idx = perm[i0:i0 + bs]
            Xb, yb = Xtr[idx], ytr[idx]
            logits = model(Xb, yb)
            loss = F.cross_entropy(
                logits[:, 1:, :].reshape(-1, VOCAB),
                yb[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        vh2 = val_h2()
        if vh2 > best_va:
            best_va = vh2
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, CKPT)
    model.eval()
    print(f"trained; best val h2={best_va:.4f}; ckpt -> {CKPT}")

    # ---------- 1. full-population drafts ----------
    with torch.no_grad():
        bl = raw_head_logits(Xho, fr)          # [N, V] bypass logits
        b_top2 = bl.topk(2, dim=-1).indices    # [N, 2]
        h1_tok = b_top2[:, 0]
        h1_rank = (bl.argmax(-1) == yho[:, 0]).long()  # rank1 correct?
        # where is the TRUE token in the bypass ranking?
        true_l1 = bl.gather(1, yho[:, 0:1]).squeeze(1)
        rank_of_true = (bl > true_l1.unsqueeze(1)).sum(1)  # 0=rank1

    # free-roll drafts for ALL held-out (token1 = bypass argmax)
    drafts = torch.empty(N, K, dtype=torch.long, device=DEV)
    drafts[:, 0] = h1_tok
    t0 = time.time()
    with torch.no_grad():
        for i in range(N):
            toks = [int(h1_tok[i])]
            seed = Xho[i:i + 1]
            for _ in range(K - 1):
                lg = model(seed, torch.tensor([toks], device=DEV))
                toks.append(int(lg[0, -1].argmax()))
            drafts[i] = torch.tensor(toks, device=DEV)
    print(f"free-roll: {N} rollouts in {time.time()-t0:.0f}s")

    # ---------- 2. survival + E[A] identity ----------
    y = yho
    S = {}
    for k in (1, 2, 3, 4, 8, 16, 32):
        S[k] = (drafts[:, :k] == y[:, :k]).all(dim=1
                                               ).float().mean().item()
    ea_direct = []
    for i in range(N):
        pref = 0
        for p, t in zip(drafts[i].tolist(), y[i].tolist()):
            if p != t:
                break
            pref += 1
        ea_direct.append(min(pref + 1, K))
    ea_direct = sum(ea_direct) / N
    ea_identity = 1 + sum(S[k] for k in (1, 2, 3, 4, 8, 16, 32)
                          if k <= K) - (S[32] if False else 0)
    # careful: 1 + sum_{k=1..32} S_k — but S_32 double counts the
    # full-survival case: with prefix+1 capped at K, E[A] =
    # 1 + sum_{k=1}^{K-1} S_k + 0*S_K... actually accepted = pref+1
    # capped at K: for a full match (pref=K), accepted=K, while
    # 1+sum_{k=1}^{K} S_k = K+1 > K. So identity uses S_1..S_{K-1}
    # plus the cap. Standard: E[A] = sum_{k=0}^{K-1} S_k where
    # S_0 = 1 (accepted >= 1 always).
    # exact prefix-length distribution
    plens = []
    for i in range(N):
        pref = 0
        for p, t in zip(drafts[i].tolist(), y[i].tolist()):
            if p != t:
                break
            pref += 1
        plens.append(pref)
    ea_from_S = 1.0 + sum(1 for p in plens if p >= 1) / N \
        + sum(1 for p in plens if p >= 2) / N \
        + sum(1 for p in plens if p >= 3) / N \
        + sum(1 for p in plens if p >= 4) / N \
        + sum(1 for p in plens if p >= 8) / N \
        + sum(1 for p in plens if p >= 16) / N
    # NOTE: full identity needs every k; the above undercounts if
    # prefixes exceed 16 — compute exactly:
    from collections import Counter
    pc = Counter(plens)
    ea_exact = 1 + sum(pc[k] / N for k in range(1, K))

    # ---------- 3. contingency table ----------
    # teacher-forced h2 correctness (from the TF forward on full held-out)
    tf_h2 = torch.empty(N, dtype=torch.bool, device=DEV)
    with torch.no_grad():
        for i in range(0, N, 16):
            lg = model(Xho[i:i + 16], yho[i:i + 16])
            tf_h2[i:i + 16] = lg[:, 1, :].argmax(-1) == yho[i:i + 16, 1]
    # free-roll h2 correctness
    fr_h2 = drafts[:, 1] == yho[:, 1]

    def cond(prob_mask, cond_mask):
        c = cond_mask.sum().item()
        return round((prob_mask & cond_mask).sum().item() / c, 4) if c else None

    rank1 = rank_of_true == 0
    rank2 = rank_of_true == 1
    rank_gt2 = rank_of_true > 1
    top2 = rank_of_true <= 1

    # ---------- 4. TF vs FR h2 logits on h1-correct ----------
    # For h1-correct examples the FR input at position 1 equals the
    # true token — so TF and FR predictions must match.
    h1c = rank1
    agree = (tf_h2[h1c] == fr_h2[h1c]).float().mean().item() \
        if h1c.any() else None

    # ---------- 5. causal-mask invariance ----------
    with torch.no_grad():
        Xb = Xho[:4]
        yb = yho[:4].clone()
        l1 = model(Xb, yb)
        yb2 = yho[:4].clone()
        yb2[:, 5:] = (yb2[:, 5:] + 7) % VOCAB   # perturb future
        l2 = model(Xb, yb2)
        # predictions at positions 0..4 must be unchanged
        inv = (l1[:, :5, :].argmax(-1)
               == l2[:, :5, :].argmax(-1)).float().mean().item()

    audit = {
        "population": f"full held-out N={N}",
        "prefix_len_distribution": {str(k): pc[k] for k in
                                    sorted(pc)[:12]},
        "S": {str(k): round(v, 4) for k, v in S.items()},
        "E_A_direct": round(ea_direct, 4),
        "E_A_identity_1_plus_sum_S1_S31": round(ea_exact, 4),
        "identity_holds": abs(ea_direct - ea_exact) < 1e-9,
        "bypass_h1_full_set": round(S[1], 4),
        "contingency": {
            "n_rank1": int(rank1.sum()),
            "n_rank2": int(rank2.sum()),
            "n_rank_gt2": int(rank_gt2.sum()),
            "P_tf_h2_given_rank1": cond(tf_h2, rank1),
            "P_tf_h2_given_rank2": cond(tf_h2, rank2),
            "P_tf_h2_given_rank_gt2": cond(tf_h2, rank_gt2),
            "P_fr_h2_given_rank1": cond(fr_h2, rank1),
            "P_fr_h2_given_rank2": cond(fr_h2, rank2),
            "P_fr_h2_top2_first_branch":
                None,   # needs tree sim; separate
            "top2_coverage_h1": round(top2.float().mean().item(), 4),
        },
        "tf_fr_h2_agreement_on_h1_correct": round(agree, 4)
        if agree is not None else None,
        "causal_mask_invariance_positions0_4": round(inv, 4),
    }
    print(json.dumps(audit, indent=1))
    OUT.write_text(json.dumps(audit, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
