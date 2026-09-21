"""Track T1: temporal multi-state causal proposer.

Conditions on h40,t-3:t (4 observed target states) + explicit state
deltas, on top of the validated T0 pieces (raw h1 bypass, tiny causal
transformer, low-rank draft head, certified KV rollout).

Architecture deltas from T0:
  seed = seed_proj(h40[t])                          # same as T0
        + context_proj(cat(state_proj(h[t-i])))     # 4-state context
        + motion_proj(cat(deltas))                  # explicit Δh

Primary metric: P(h2 | h1 correct). Baseline: 0.121 (T0).
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

from tiny_causal import (load_frozen, raw_head_logits, K, VOCAB,
                         DEV, D_MODEL, N_LAYER, N_HEAD, LRANK)
from tiny_causal import Block

DATA = Path("/tmp/stage1_data")
PROMPTS = Path("artifacts/stage1_prompts.json")
OUT = Path("artifacts/t1_multistate_result.json")
HC, HIDDEN = 4, 4096
CTX = 4          # number of states in the temporal context


def load_split_ms(names):
    """Returns (X_multi [N, CTX, 4, 4096], X_cur [N, 4, 4096],
    y [N, K]). Position p must have p >= CTX-1 for full context."""
    xs_m, xs_c, ys = [], [], []
    for cat in names:
        f = DATA / f"{cat}.pt"
        if not f.exists():
            continue
        d = torch.load(f, weights_only=False)
        g = d["gen"]
        st = d["states"][40].float().view(-1, 4, 4096)
        pos = min(len(g), st.shape[0])
        for p in range(CTX - 1, pos):
            if p + K - 1 < len(g):
                xs_m.append(st[p - CTX + 1:p + 1])   # [CTX, 4, 4096]
                xs_c.append(st[p])                    # [4, 4096]
                ys.append(g[p:p + K])
    if not xs_m:
        return None, None, None
    return (torch.stack(xs_m), torch.stack(xs_c),
            torch.tensor(ys, dtype=torch.long))


class TinyCausalMS(torch.nn.Module):
    """Multi-state tiny causal transformer."""

    def __init__(self, emb_table):
        super().__init__()
        self.emb_proj = torch.nn.Linear(HIDDEN, D_MODEL, bias=False)
        self.emb_table = emb_table
        # T0 seed (current state)
        self.seed_proj = torch.nn.Linear(HC * HIDDEN, D_MODEL)
        # multi-state context: per-state projections + combine
        self.state_proj = torch.nn.ModuleList([
            torch.nn.Linear(HC * HIDDEN, 128) for _ in range(CTX)])
        self.ctx_combine = torch.nn.Linear(CTX * 128, D_MODEL)
        # explicit motion (deltas)
        self.motion_proj = torch.nn.Linear(
            (CTX - 1) * HC * HIDDEN, 128)
        self.motion_out = torch.nn.Linear(128, D_MODEL)
        self.pos = torch.nn.Parameter(
            torch.randn(K, D_MODEL) * 0.02)
        self.blocks = torch.nn.ModuleList(
            [Block(D_MODEL, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = torch.nn.LayerNorm(D_MODEL)
        self.head_down = torch.nn.Linear(D_MODEL, LRANK)
        self.head_out = torch.nn.Linear(LRANK, VOCAB)
        torch.nn.init.zeros_(self.head_out.weight)
        torch.nn.init.zeros_(self.head_out.bias)

    def compute_seed(self, X_cur, X_multi):
        """X_cur [N, 4, 4096]; X_multi [N, CTX, 4, 4096]."""
        N = X_cur.shape[0]
        s = self.seed_proj(X_cur.flatten(1))            # [N, d]
        # per-state projections
        sp = [self.state_proj[i](X_multi[:, i].flatten(1))
              for i in range(CTX)]                       # each [N, 128]
        ctx = self.ctx_combine(torch.cat(sp, dim=-1))    # [N, d]
        # deltas: Δh[i] = h[i+1] - h[i] for i in 0..CTX-2
        deltas = (X_multi[:, 1:] - X_multi[:, :-1]).reshape(
            N, -1)                                       # [N, 3*16384]
        mot = self.motion_out(F.gelu(self.motion_proj(deltas)))
        return s + ctx + mot                             # [N, d]

    def embed_tokens(self, toks):
        with torch.no_grad():
            e = self.emb_table[toks]
        return self.emb_proj(e.float())

    def forward(self, X_cur, X_multi, toks):
        """toks [N, L] input tokens. Returns [N, L, V]."""
        N = X_cur.shape[0]
        s = self.compute_seed(X_cur, X_multi).unsqueeze(1)  # [N,1,d]
        t = self.embed_tokens(toks)
        x = torch.cat([s, t[:, :-1]], dim=1)
        T = x.shape[1]
        x = x + self.pos[:T].unsqueeze(0)
        mask = torch.triu(torch.ones(T, T, device=x.device,
                                     dtype=torch.bool), diagonal=1)
        for b in self.blocks:
            x = b(x, mask)
        x = self.ln_f(x)
        return self.head_out(F.gelu(self.head_down(x)))

    def kv_rollout(self, X_cur, X_multi, first_tok, n_steps):
        """Certified single-pass KV rollout (multi-state seed)."""
        with torch.no_grad():
            toks = [int(first_tok)]
            x = self.compute_seed(X_cur, X_multi).unsqueeze(1) \
                + self.pos[0].view(1, 1, -1)
            caches = []
            for b in self.blocks:
                h = b.ln1(x)
                E = F.linear(h, b.attn.in_proj_weight,
                             b.attn.in_proj_bias)
                _, k, v = E.chunk(3, dim=-1)
                a = F.linear(v, b.attn.out_proj.weight,
                             b.attn.out_proj.bias)
                caches.append((k, v))
                x = x + a
                x = x + b.mlp(b.ln2(x))
            for _ in range(n_steps - 1):
                e = self.embed_tokens(
                    torch.tensor([[toks[-1]]], device=x.device))
                x = e + self.pos[len(toks)].view(1, 1, -1)
                new_caches = []
                for bi, b in enumerate(self.blocks):
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
                h_last = self.ln_f(x)[:, -1]
                lg = self.head_out(F.gelu(self.head_down(h_last)))
                toks.append(int(lg.argmax()))
            return toks


def main():
    torch.manual_seed(0)
    spec = json.loads(PROMPTS.read_text())
    splits = spec["splits"]
    Xm_tr, Xc_tr, y_tr = load_split_ms(splits["train"])
    Xm_va, Xc_va, y_va = load_split_ms(splits["val"])
    Xm_ho, Xc_ho, y_ho = load_split_ms(splits["test"])
    print(f"windows: train {Xc_tr.shape[0]} val {Xc_va.shape[0]} "
          f"test {Xc_ho.shape[0]}")
    fr = load_frozen()
    model = TinyCausalMS(fr["embed.weight"].to(DEV)).to(DEV)
    model.emb_table.requires_grad_(False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-4)
    Xc_tr, Xm_tr, y_tr = (Xc_tr.to(DEV), Xm_tr.to(DEV),
                           y_tr.to(DEV))
    Xc_va, Xm_va, y_va = (Xc_va.to(DEV), Xm_va.to(DEV),
                           y_va.to(DEV))
    Xc_ho, Xm_ho, y_ho = (Xc_ho.to(DEV), Xm_ho.to(DEV),
                           y_ho.to(DEV))
    N = Xc_tr.shape[0]
    bs = 32
    best_va, best_state = -1, None

    def val_h2():
        with torch.no_grad():
            lg = model(Xc_va[:64], Xm_va[:64], y_va[:64])
            return (lg[:64, 1, :].argmax(-1)
                    == y_va[:64, 1]).float().mean().item()

    t0 = time.time()
    CKPT_T1 = Path("/tmp/tiny_causal_t1.pt")
    if CKPT_T1.exists():
        model.load_state_dict(torch.load(CKPT_T1, map_location=DEV))
        model.eval()
        print("loaded existing T1 checkpoint; skipping training")
        best_va = -1
    else:
        for ep in range(14):
            model.train()
            perm = torch.randperm(N)
            tot = nb = 0
            for i0 in range(0, N, bs):
                idx = perm[i0:i0 + bs]
                logits = model(Xc_tr[idx], Xm_tr[idx], y_tr[idx])
                loss = F.cross_entropy(
                    logits[:, 1:, :].reshape(-1, VOCAB),
                    y_tr[idx][:, 1:].reshape(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                opt.step()
                tot += loss.item(); nb += 1
            vh2 = val_h2()
            if vh2 > best_va:
                best_va = vh2
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            print(f"ep{ep}: loss={tot/nb:.3f} val_h2={vh2:.4f}",
                  flush=True)
        model.load_state_dict(best_state)
        model.eval()
        torch.save(best_state, "/tmp/tiny_causal_t1.pt")

    # ---------- evaluation (certified KV rollout) ----------
    with torch.no_grad():
        bl = raw_head_logits(Xc_ho, fr)
    h1_pred = bl.argmax(-1)
    N_ho = Xc_ho.shape[0]
    drafts = []
    for i in range(N_ho):
        d = model.kv_rollout(Xc_ho[i:i + 1], Xm_ho[i:i + 1],
                             h1_pred[i], K)
        drafts.append(d)
    plens = []
    for i in range(N_ho):
        pref = 0
        for p, t in zip(drafts[i], y_ho[i].tolist()):
            if p != t:
                break
            pref += 1
        plens.append(pref)
    pc = Counter(plens)
    ea = sum(pc[j] * min(j + 1, K) for j in pc) / N_ho
    ea_id = 1.0 + sum(sum(1 for p in plens if p >= k) / N_ho
                      for k in range(1, K))
    S = {k: round(sum(1 for p in plens if p >= k) / N_ho, 4)
         for k in (1, 2, 3, 4, 8, 16, 32)}
    # conditionals
    h1c = h1_pred == y_ho[:, 0]
    fr_h2 = torch.tensor([d[1] for d in drafts],
                         device=y_ho.device) == y_ho[:, 1]
    p_h2_h1 = ((fr_h2 & h1c).sum().item()
               / max(h1c.sum().item(), 1))
    conds = {}
    for k in (2, 3, 4, 8):
        prev = torch.tensor(
            [all(d[j] == t[j] for j in range(k - 1))
             for d, t in zip(drafts, y_ho.tolist())],
            device=y_ho.device)
        cur = torch.tensor(
            [all(d[j] == t[j] for j in range(k))
             for d, t in zip(drafts, y_ho.tolist())],
            device=y_ho.device)
        n_prev = prev.sum().item()
        conds[f"P(h{k}|prefix{k-1})"] = round(
            (cur & prev).sum().item() / n_prev, 4) if n_prev else None
    # latency
    with torch.no_grad():
        for _ in range(3):
            model.kv_rollout(Xc_ho[:1], Xm_ho[:1], h1_pred[0], K)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(10):
            model.kv_rollout(Xc_ho[:1], Xm_ho[:1], h1_pred[0], K)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t1) / 10 * 1e3
    result = {
        "classification": "T1_TEMPORAL_MULTISTATE",
        "design": f"4-state context + deltas + T0 pieces "
                  f"(bypass, causal, low-rank head, certified KV)",
        "params_millions": round(sum(
            p.numel() for p in model.parameters()
            if p.requires_grad) / 1e6, 2),
        "val_h2_best": round(best_va, 4),
        "S": S,
        "prefix_distribution": {str(k): pc[k] for k in
                                sorted(pc)[:10]},
        "E_A_direct": round(ea, 6),
        "E_A_identity": round(ea_id, 6),
        "invariant_holds": abs(ea - ea_id) < 1e-6,
        "P_h2_given_h1": round(p_h2_h1, 4),
        "conditional_chain": conds,
        "kv_latency_ms_K32": round(lat, 1),
        "train_seconds": round(time.time() - t0),
    }
    print(json.dumps(result, indent=1))
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
