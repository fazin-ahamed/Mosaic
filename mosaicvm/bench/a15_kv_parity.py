"""Track A1.5: E[A] invariant fix + KV rollout parity certification.

1. E[A] must satisfy THREE independent computations:
   direct accepted-count, 1+sum(S_k), prefix histogram — all equal.
   (The prior bug: summed exact-k counts instead of cumulative >=k.)

2. Clean single-pass KV rollout: seed at position 0, then each
   drafted token at position len(toks); caches grow per block.
   Must be token-for-token identical to the corrected naive rollout
   across full K=32 on every held-out example.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from tiny_causal import (TinyCausal, load_split, load_frozen,
                         raw_head_logits, K, DEV)

PROMPTS = Path("artifacts/stage1_prompts.json")
CKPT = Path("/tmp/tiny_causal_t0.pt")
OUT = Path("artifacts/a15_kv_parity.json")


def fixed_rollout(model, seed, first_tok, n_steps):
    """Corrected naive rollout (semantic reference)."""
    with torch.no_grad():
        toks = [int(first_tok)]
        for _ in range(n_steps - 1):
            t_in = torch.tensor([toks + [0]], device=seed.device)
            lg = model(seed, t_in)
            toks.append(int(lg[0, -1].argmax()))
        return toks


def kv_rollout(model, seed, first_tok, n_steps):
    """Clean single-pass KV-cached rollout. Processes the seed at
    position 0, then each drafted token at position len(toks);
    per-block K/V caches grow incrementally."""
    with torch.no_grad():
        toks = [int(first_tok)]
        # --- seed at position 0 ---
        x = model.seed_proj(seed.flatten(1)).unsqueeze(1) \
            + model.pos[0].view(1, 1, -1)
        caches = []
        for b in model.blocks:
            h = b.ln1(x)
            E = F.linear(h, b.attn.in_proj_weight,
                         b.attn.in_proj_bias)
            _, k, v = E.chunk(3, dim=-1)
            # single position attends to itself
            a = F.linear(v, b.attn.out_proj.weight,
                         b.attn.out_proj.bias)
            caches.append((k, v))
            x = x + a
            x = x + b.mlp(b.ln2(x))
        # --- each drafted token at position len(toks) ---
        for _ in range(n_steps - 1):
            e = model.embed_tokens(
                torch.tensor([[toks[-1]]], device=seed.device))
            x = e + model.pos[len(toks)].view(1, 1, -1)
            new_caches = []
            for bi, b in enumerate(model.blocks):
                h = b.ln1(x)
                E = F.linear(h, b.attn.in_proj_weight,
                             b.attn.in_proj_bias)
                q, k, v = E.chunk(3, dim=-1)
                k_prev, v_prev = caches[bi]
                k_all = torch.cat([k_prev, k], dim=1)
                v_all = torch.cat([v_prev, v], dim=1)
                new_caches.append((k_all, v_all))
                n_head = b.attn.num_heads
                d_head = k_all.shape[-1] // n_head
                q_h = q.view(1, 1, n_head, d_head).transpose(1, 2)
                k_h = k_all.view(1, -1, n_head,
                                 d_head).transpose(1, 2)
                v_h = v_all.view(1, -1, n_head,
                                 d_head).transpose(1, 2)
                a = F.scaled_dot_product_attention(q_h, k_h, v_h)
                a = a.transpose(1, 2).reshape(1, 1, -1)
                a = F.linear(a, b.attn.out_proj.weight,
                             b.attn.out_proj.bias)
                x = x + a
                x = x + b.mlp(b.ln2(x))
            caches = new_caches
            h_last = model.ln_f(x)[:, -1]
            lg = model.head_out(F.gelu(model.head_down(h_last)))
            toks.append(int(lg.argmax()))
        return toks


def main():
    spec = json.loads(PROMPTS.read_text())
    Xho, yho = load_split(spec["splits"]["test"])
    fr = load_frozen()
    Xho, yho = Xho.to(DEV), yho.to(DEV)
    N = Xho.shape[0]

    model = TinyCausal(fr["embed.weight"].to(DEV)).to(DEV)
    model.load_state_dict(torch.load(CKPT, map_location=DEV))
    model.eval()

    with torch.no_grad():
        bl = raw_head_logits(Xho, fr)
    h1_pred = bl.argmax(-1)

    # ---------- 1. full-population corrected rollout (naive ref) ----------
    drafts = []
    t0 = time.time()
    for i in range(N):
        drafts.append(fixed_rollout(model, Xho[i:i + 1],
                                    h1_pred[i], K))
    naive_time = time.time() - t0

    # three-way E[A] invariant
    plens = []
    for i in range(N):
        pref = 0
        for p, t in zip(drafts[i], yho[i].tolist()):
            if p != t:
                break
            pref += 1
        plens.append(pref)
    pc = Counter(plens)
    # (a) direct accepted count
    ea_direct = sum(min(p + 1, K) for p in plens) / N
    # (b) 1 + sum S_k  (S_k = P(prefix >= k) — CUMULATIVE)
    ea_identity = 1.0 + sum(
        sum(1 for p in plens if p >= k) / N
        for k in range(1, K))
    # (c) prefix histogram
    ea_hist = sum((pc[j] / N) * min(j + 1, K)
                  for j in range(K)) \
        if max(plens) < K else None
    # histogram form: sum over exact prefix length j of (j+1)/N
    ea_hist = sum(pc[j] * min(j + 1, K) for j in pc) / N

    S = {k: round(sum(1 for p in plens if p >= k) / N, 4)
         for k in (1, 2, 3, 4, 8, 16, 32)}

    # ---------- 2. KV parity: token-for-token on every example ----------
    kv_full_match = 0
    kv_tok_match = 0
    kv_tok_total = 0
    for i in range(N):
        kv = kv_rollout(model, Xho[i:i + 1], h1_pred[i], K)
        nv = drafts[i]
        if kv == nv:
            kv_full_match += 1
        for a, b in zip(kv, nv):
            kv_tok_total += 1
            if a == b:
                kv_tok_match += 1

    # ---------- 3. KV latency ----------
    x1 = Xho[0:1]
    t1 = h1_pred[0]
    with torch.no_grad():
        for _ in range(3):
            kv_rollout(model, x1, t1, K)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            kv_rollout(model, x1, t1, K)
        torch.cuda.synchronize()
        kv_lat = (time.perf_counter() - t0) / 10 * 1e3
        t0 = time.perf_counter()
        for _ in range(10):
            fixed_rollout(model, x1, t1, K)
        torch.cuda.synchronize()
        naive_lat = (time.perf_counter() - t0) / 10 * 1e3

    result = {
        "classification": "A15_EA_INVARIANT_AND_KV_PARITY",
        "population": f"full held-out N={N}",
        "E_A_three_way": {
            "direct": round(ea_direct, 6),
            "identity_1_plus_sum_Sk": round(ea_identity, 6),
            "prefix_histogram": round(ea_hist, 6),
            "invariant_holds": (
                abs(ea_direct - ea_identity) < 1e-6 and
                abs(ea_direct - ea_hist) < 1e-6),
        },
        "S": S,
        "prefix_distribution": {str(k): pc[k] for k in
                                sorted(pc)[:10]},
        "kv_parity": {
            "full_K32_sequence_match": f"{kv_full_match}/{N}",
            "full_match_rate": round(kv_full_match / N, 4),
            "per_token_agreement": round(kv_tok_match / kv_tok_total, 6),
        },
        "latency_ms_K32": {
            "naive_corrected": round(naive_lat, 1),
            "kv_cached": round(kv_lat, 1),
        },
        "verdict": None,
    }
    inv_ok = result["E_A_three_way"]["invariant_holds"]
    kv_ok = kv_full_match == N
    if inv_ok and kv_ok:
        result["verdict"] = ("A1.5 PASS — E[A] invariant holds "
                            "three-way; KV rollout token-identical "
                            f"on all {N} examples; KV latency "
                            f"{kv_lat:.1f}ms certified")
    else:
        result["verdict"] = (f"A1.5 PARTIAL — invariant={inv_ok}, "
                            f"KV full match {kv_full_match}/{N}")
    print(json.dumps(result, indent=1))
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
