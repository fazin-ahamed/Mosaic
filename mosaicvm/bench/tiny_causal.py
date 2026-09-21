"""Track T: seed-preserving tiny causal transformer (T0 screen).

Hard design rules (roadmap §26-35):
  1. Token 1 BYPASSES the learned model: logits_1 = raw L40 head
     (source hc_head mixing + frozen lm_head), zero-init residual.
     The temporal model starts at horizon 2.
  2. No full-vocab head per step: low-rank draft head (d->128->V).
     Input embeddings = FROZEN target embedding x learned projection
     (zero vocabulary-sized parameters).
  3. Causal attention over [seed, committed/proposed tokens] — the
     model can revisit the full trajectory at every horizon.

T0 gate: h1 >= ~0.44 (from bypass), h2 >> 0.084 (target 0.20-0.25),
S2 > 0.10, S4 >> 0.004, E[A] free-roll > 2, rollout < 30ms class.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

DATA = Path("/tmp/stage1_data")
PROMPTS = Path("artifacts/stage1_prompts.json")
OUT = Path("artifacts/tiny_causal_result.json")
K = 32
HC, HIDDEN = 4, 4096
VOCAB = 129280
DEV = "cuda"
D_MODEL = 512
N_LAYER = 3
N_HEAD = 8
LRANK = 128


def load_split(names):
    xs, ys = [], []
    for cat in names:
        f = DATA / f"{cat}.pt"
        if not f.exists():
            continue
        d = torch.load(f, weights_only=False)
        g = d["gen"]
        st = d["states"][40].float().view(-1, 4, 4096)
        pos = min(len(g), st.shape[0])
        for p in range(pos):
            if p + K - 1 < len(g):
                xs.append(st[p])
                ys.append(g[p:p + K])
    if not xs:
        return None, None
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def load_frozen():
    import glob, os
    from safetensors.torch import safe_open
    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--"
        "DeepSeek-V4-Flash-0731/snapshots/*"))[0]
    idx = json.load(open(os.path.join(
        snap, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for nm in ("head.weight", "hc_head_fn", "hc_head_base",
               "hc_head_scale", "embed.weight"):
        with safe_open(os.path.join(snap, idx[nm]), framework="pt") as f:
            out[nm] = f.get_tensor(nm)
    return out


class Block(torch.nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d)
        self.attn = torch.nn.MultiheadAttention(
            d, n_head, batch_first=True)
        self.ln2 = torch.nn.LayerNorm(d)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, 4 * d), torch.nn.GELU(),
            torch.nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyCausal(torch.nn.Module):
    def __init__(self, emb_table):
        super().__init__()
        self.emb_proj = torch.nn.Linear(HIDDEN, D_MODEL, bias=False)
        self.emb_table = emb_table              # frozen [V, 4096] bf16
        self.seed_proj = torch.nn.Linear(HC * HIDDEN, D_MODEL)
        self.pos = torch.nn.Parameter(
            torch.randn(K, D_MODEL) * 0.02)
        self.blocks = torch.nn.ModuleList(
            [Block(D_MODEL, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = torch.nn.LayerNorm(D_MODEL)
        # low-rank draft head
        self.head_down = torch.nn.Linear(D_MODEL, LRANK)
        self.head_out = torch.nn.Linear(LRANK, VOCAB)
        torch.nn.init.zeros_(self.head_out.weight)
        torch.nn.init.zeros_(self.head_out.bias)

    def embed_tokens(self, toks):
        with torch.no_grad():
            e = self.emb_table[toks]            # [N, K, 4096] bf16
        return self.emb_proj(e.float())

    def forward(self, seed, toks):
        """seed [N, 4, 4096]; toks [N, K] input tokens (positions
        0..K-1 receive seed at 0 and toks[:, :-1] at 1..K-1).
        Returns draft logits [N, K, V]."""
        N = seed.shape[0]
        s = self.seed_proj(seed.flatten(1)).unsqueeze(1)
        t = self.embed_tokens(toks)             # [N, K, d]
        x = torch.cat([s, t[:, :-1]], dim=1)    # [N, T, d]
        T = x.shape[1]
        x = x + self.pos[:T].unsqueeze(0)
        mask = torch.triu(torch.ones(T, T, device=x.device,
                                     dtype=torch.bool), diagonal=1)
        for b in self.blocks:
            x = b(x, mask)
        x = self.ln_f(x)
        return self.head_out(F.gelu(self.head_down(x)))


@torch.no_grad()
def raw_head_logits(seed, fr):
    """Bypass: source hc_head mixing + frozen lm_head."""
    xf = seed.flatten(1).float()
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
    mixes = F.linear(xf, fr["hc_head_fn"].float().to(DEV)) * rs
    pre = torch.sigmoid(
        mixes * fr["hc_head_scale"].float().to(DEV)
        + fr["hc_head_base"].float().to(DEV)) + 1e-6
    y = torch.sum(pre.unsqueeze(-1) * seed, dim=1)
    return F.linear(y.to(torch.bfloat16),
                    fr["head.weight"].to(DEV)).float()   # [N, V]


def chunked_ce(logits_fn, targets, chunk=4096, vocab_chunk=None):
    """CE over [N*K, V] without materializing all logits."""
    total = 0.0
    n = targets.numel()
    for i in range(0, n, chunk):
        lg = logits_fn(i, min(chunk, n - i))
        t = targets.reshape(-1)[i:i + chunk]
        total = total + F.cross_entropy(lg, t, reduction="sum")
    return total / n


@torch.no_grad()
def evaluate(model, X, y, fr):
    model.eval()
    N = X.shape[0]
    # bypass h1
    bl = raw_head_logits(X, fr)
    h1 = (bl.argmax(-1) == y[:, 0]).float().mean().item()
    # teacher-forced draft logits (chunked over batch)
    pred = torch.empty(N, K, dtype=torch.long, device=DEV)
    for i in range(0, N, 8):
        lg = model(X[i:i + 8], y[i:i + 8])
        pred[i:i + 8] = lg.argmax(-1)
    # draft h-k (k>=2) — positions 1..K-1
    top1 = {"1": round(h1, 4)}
    for k in (2, 4, 8, 16, 32):
        top1[str(k)] = round(
            (pred[:, k - 1] == y[:, k - 1]).float().mean().item(), 4)
    # free-roll E[A]: token1 = bypass argmax; tokens 2.. = model
    # rollout with the model conditioned on its own trajectory
    model.eval()
    eas = []
    for i in range(min(96, N)):
        toks = [int(bl[i].argmax())]
        seed = X[i:i + 1]
        for step in range(K - 1):
            t_in = torch.tensor([toks], device=DEV)
            lg = model(seed, t_in)              # [1, len, V]
            nxt = int(lg[0, -1].argmax())
            toks.append(nxt)
        truth = y[i].tolist()
        pref = 0
        for p, t in zip(toks, truth):
            if p != t:
                break
            pref += 1
        eas.append(min(pref + 1, K))
    ea = sum(eas) / len(eas)
    # survival from the free-roll subset
    surv = {}
    roll = []
    for i in range(min(96, N)):
        toks = [int(bl[i].argmax())]
        seed = X[i:i + 1]
        for step in range(K - 1):
            t_in = torch.tensor([toks], device=DEV)
            lg = model(seed, t_in)
            toks.append(int(lg[0, -1].argmax()))
        roll.append(toks)
    for k in (2, 4, 8):
        surv[str(k)] = round(sum(
            1 for i, toks in enumerate(roll)
            if toks[:k] == y[i].tolist()[:k]) / len(roll), 4)
    return {"top1_by_h": top1, "survival": surv,
            "E_A_free_roll": round(ea, 3)}


def main():
    spec = json.loads(PROMPTS.read_text())
    splits = spec["splits"]
    Xtr, ytr = load_split(splits["train"])
    Xva, yva = load_split(splits["val"])
    Xho, yho = load_split(splits["test"])
    print(f"windows: {Xtr.shape[0]}/{Xva.shape[0]}/{Xho.shape[0]}")
    fr = load_frozen()
    Xtr, ytr = Xtr.to(DEV), ytr.to(DEV)
    Xva, yva = Xva.to(DEV), yva.to(DEV)
    Xho, yho = Xho.to(DEV), yho.to(DEV)

    # bypass sanity on held-out
    with torch.no_grad():
        bl = raw_head_logits(Xho, fr)
        h1_raw = (bl.argmax(-1) == yho[:, 0]).float().mean().item()
    print(f"bypass raw h1 (held-out): {h1_raw:.4f}")

    model = TinyCausal(fr["embed.weight"].to(DEV)).to(DEV)
    # freeze the embedding table
    model.emb_table.requires_grad_(False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-4)
    N = Xtr.shape[0]
    bs = 32
    best_va, best_state = -1, None

    def val_h2():
        with torch.no_grad():
            # bypass token1; model token2 on val
            lg = model(Xva[:64], yva[:64])
            return (lg[:64, 1, :].argmax(-1)
                    == yva[:64, 1]).float().mean().item()

    for ep in range(14):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        nb = 0
        for i0 in range(0, N, bs):
            idx = perm[i0:i0 + bs]
            Xb, yb = Xtr[idx], ytr[idx]
            # loss ONLY on horizons 2..32 (positions 1..K-1):
            # horizon 1 belongs to the frozen bypass
            logits = model(Xb, yb)             # [bs, K, V]
            lg = logits[:, 1:, :]              # horizons 2..K
            loss = F.cross_entropy(
                lg.reshape(-1, VOCAB), yb[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        vh2 = val_h2()
        if vh2 > best_va:
            best_va = vh2
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        print(f"ep{ep}: loss={tot/nb:.3f} val_h2={vh2:.4f}",
              flush=True)
    if best_state:
        model.load_state_dict(best_state)

    ev = evaluate(model, Xho, yho, fr)
    # rollout latency (free-roll, K=32, batch 1)
    model.eval()
    x1 = Xho[:1]
    toks0 = [0]
    with torch.no_grad():
        for _ in range(2):
            model(x1, torch.tensor([toks0 * 32], device=DEV))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            toks = [0]
            for _ in range(K - 1):
                lg = model(x1, torch.tensor([toks], device=DEV))
                toks.append(int(lg[0, -1].argmax()))
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / 5 * 1e3
    ev["rollout_latency_ms_K32"] = round(lat, 1)
    ev["params_millions_trainable"] = round(sum(
        p.numel() for p in model.parameters()
        if p.requires_grad) / 1e6, 2)
    ev["bypass_h1_heldout"] = round(h1_raw, 4)
    print(json.dumps(ev, indent=1))
    OUT.write_text(json.dumps({
        "classification": "TINY_CAUSAL_SEED_PRESERVING_T0",
        "design": f"{N_LAYER}-layer causal, d={D_MODEL}, low-rank "
                  f"{LRANK} draft head, frozen target embeddings, "
                  "raw-L40 h1 bypass",
        "results": ev}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
