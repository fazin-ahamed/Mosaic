"""Track P: recurrent target-feature drafter (R0/R1/R2).

Design: z_0 = tanh(seed(h40)); z_{i+1} = GRU(z_i, embed(y_i));
logits_i = lm_head(out(z_i)). The head (frozen) is applied BATCHED
over all K states after the recurrence — training is fully
teacher-forced/vectorized; free-running eval decodes stepwise but the
head GEMM runs on [1, 4096] rows (5ms class per block, amortized).

Latency model for production: recurrence (32 x cheap GRU) + one
[N=1, K=32, 4096] head GEMM ≈ 5-8ms — well inside the ~30ms budget.

Stages: R0 teacher-forced CE; R1 scheduled sampling (p decays);
R2 free-running fine-tune (rollout CE on self-predicted tokens).
Metric: held-out greedy E[A] (prefix + 1 semantics), path survival.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")

DATA = Path("/tmp/stage1_data")
PROMPTS = Path("artifacts/stage1_prompts.json")
OUT = Path("artifacts/recurrent_race_result.json")
K = 32
HC, HIDDEN = 4, 4096
VOCAB = 129280
DEV = "cuda"


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


def load_head():
    import glob, os
    from safetensors.torch import safe_open
    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--"
        "DeepSeek-V4-Flash-0731/snapshots/*"))[0]
    idx = json.load(open(os.path.join(
        snap, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(snap, idx["head.weight"]),
                   framework="pt") as f:
        return f.get_tensor("head.weight").to(DEV)   # bf16


class RecurrentDrafter(torch.nn.Module):
    def __init__(self, d=2048, emb=512):
        super().__init__()
        self.seed = torch.nn.Linear(HC * HIDDEN, d)
        self.embed = torch.nn.Embedding(VOCAB, emb)
        self.gate_i = torch.nn.Linear(emb + d, d)
        self.cand = torch.nn.Linear(emb + d, d)
        self.out = torch.nn.Linear(d, HIDDEN, bias=False)
        self.d = d

    def forward(self, X, toks):
        """Teacher-forced. X [N, 4, 4096]; toks [N, K] ground truth.
        Returns head inputs [N, K, 4096] (out projection included)."""
        N = X.shape[0]
        z = torch.tanh(self.seed(X.flatten(1)))
        outs = [self.out(z)]
        # input token for step i is toks[:, i-1]
        e_all = self.embed(toks)             # [N, K, emb]
        for i in range(1, K):
            e = e_all[:, i - 1]
            gi = torch.sigmoid(self.gate_i(torch.cat([z, e], -1)))
            cand = torch.tanh(self.cand(torch.cat([z, e], -1)))
            z = (1 - gi) * z + gi * cand
            outs.append(self.out(z))
        return torch.stack(outs, dim=1)      # [N, K, 4096]

    def rollout(self, x1, head_w, K_roll=K, tok0=None):
        """Free-running greedy rollout from one state. x1 [1, 4, 4096].
        Returns [K_roll] predicted token ids."""
        z = torch.tanh(self.seed(x1.flatten(1)))
        preds = []
        tok = tok0
        for i in range(K_roll):
            h = self.out(z)                  # [1, 4096]
            logits = torch.nn.functional.linear(
                h.to(torch.bfloat16), head_w).float()
            nxt = int(logits.argmax(-1))
            preds.append(nxt)
            e = self.embed(torch.tensor([nxt], device=x1.device))
            gi = torch.sigmoid(self.gate_i(torch.cat([z, e], -1)))
            cand = torch.tanh(self.cand(torch.cat([z, e], -1)))
            z = (1 - gi) * z + gi * cand
        return preds


def head_logits(H, head_w, rows=256):
    """H [N, K, 4096] -> logits [N, K, V] via frozen head, chunked
    over flattened (N*K) rows; peak logits = rows x 129280 fp32."""
    N, Kk, D = H.shape
    flat = H.reshape(-1, D)
    outs = []
    for i in range(0, flat.shape[0], rows):
        lg = torch.nn.functional.linear(
            flat[i:i + rows].to(torch.bfloat16),
            head_w).float()
        outs.append(lg)
    return torch.cat(outs).reshape(N, Kk, -1)


@torch.no_grad()
def evaluate(model, X, y, head_w):
    """Held-out: teacher-forced argmax accuracy per horizon + E[A] via
    the greedy-rollout equivalence check (rollout on subset)."""
    model.eval()
    Nn = X.shape[0]
    pred = torch.empty(Nn, K, dtype=torch.long, device=DEV)
    for i in range(0, Nn, 8):
        H = model(X[i:i + 8], y[i:i + 8])
        for h0 in range(0, K, 4):
            lg = head_logits(H[:, h0:h0 + 4], head_w)
            pred[i:i + 8, h0:h0 + 4] = lg.argmax(-1)
    top1 = {k: round((pred[:, k - 1] == y[:, k - 1])
                     .float().mean().item(), 4)
            for k in (1, 2, 4, 8, 16, 32)}
    # E[A] (greedy-block semantics): prefix match of argmax chain
    match = (pred == y)
    acc = torch.zeros(X.shape[0], device=DEV)
    m = match.clone()
    for jj in range(K):
        ok = m[:, jj]
        acc += ok.float()
        m = m & ok.unsqueeze(1)
    ea = (acc + 1).clamp(max=K).mean().item()
    # survival
    surv = {k: round(torch.all(match[:, :k], dim=1)
                     .float().mean().item(), 4)
            for k in (2, 4, 8, 16, 32)}
    # free-roll check on a subset (the real deployment decode)
    model.eval()
    n_roll = min(64, X.shape[0])
    roll_ea = []
    for i in range(n_roll):
        preds = model.rollout(X[i:i + 1], head_w)
        truth = y[i].tolist()
        pref = 0
        for p, t in zip(preds, truth):
            if p != t:
                break
            pref += 1
        roll_ea.append(min(pref + 1, K))
    return {"top1_by_h": top1, "survival": surv,
            "E_A_teacher_forced_argmax": round(ea, 3),
            "E_A_free_roll_subset": round(
                sum(roll_ea) / len(roll_ea), 3)}


def main():
    spec = json.loads(PROMPTS.read_text())
    splits = spec["splits"]
    Xtr, ytr = load_split(splits["train"])
    Xva, yva = load_split(splits["val"])
    Xho, yho = load_split(splits["test"])
    print(f"windows: train {Xtr.shape[0]} val {Xva.shape[0]} "
          f"test {Xho.shape[0]}")
    head_w = load_head()
    Xtr, ytr = Xtr.to(DEV), ytr.to(DEV)
    Xva, yva = Xva.to(DEV), yva.to(DEV)
    Xho = Xho.to(DEV)
    yho_l = yho.tolist()
    yho = yho.to(DEV)

    model = RecurrentDrafter().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=1e-4)
    N = Xtr.shape[0]
    bs = 64
    best_va, best_state = -1, None

    def train_epochs(epochs, p_teacher=1.0, tag=""):
        nonlocal best_va, best_state
        for ep in range(epochs):
            perm = torch.randperm(N)
            for i0 in range(0, N, bs):
                idx = perm[i0:i0 + bs]
                Xb, yb = Xtr[idx], ytr[idx]
                if p_teacher < 1.0:
                    # scheduled sampling: with prob (1-p) replace the
                    # teacher input token by the model's own previous
                    # argmax (computed in this forward's no-grad pre-pass)
                    with torch.no_grad():
                        H = model(Xb, yb)
                        own = head_logits(H, head_w).argmax(-1)
                    mix = torch.rand(yb.shape, device=DEV) < p_teacher
                    yb_in = torch.where(mix, yb, own)
                else:
                    yb_in = yb
                loss = 0.0
                for h0 in range(0, K, 8):
                    Hc = model(Xb, yb_in)[:, h0:h0 + 8]
                    lg = head_logits(Hc, head_w)
                    loss = loss + torch.nn.functional.cross_entropy(
                        lg.reshape(-1, VOCAB),
                        yb_in[:, h0:h0 + 8].reshape(-1))
                loss = loss / (K // 8)
                opt.zero_grad()
                loss.backward()
                opt.step()
            # val: horizon-1 top1 only (single-horizon head GEMM)
            with torch.no_grad():
                Hv = model(Xva, yva)[:, :1]     # [N, 1, 4096]
                lv = head_logits(Hv, head_w)
                acc1 = (lv[:, 0, :].argmax(-1) == yva[:, 0]
                        ).float().mean().item()
            if acc1 > best_va:
                best_va = acc1
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            print(f"  {tag} ep{ep}: loss={loss.item():.3f} "
                  f"val_h1={acc1:.4f}", flush=True)

    t0 = time.time()
    train_epochs(12, 1.0, "R0")
    train_epochs(6, 0.7, "R1")
    train_epochs(4, 0.4, "R2")
    if best_state is not None:
        model.load_state_dict(best_state)

    ev = evaluate(model, Xho, yho, head_w)
    ev["val_h1_best"] = round(best_va, 4)
    ev["train_seconds"] = round(time.time() - t0)
    ev["params_millions"] = round(sum(
        p.numel() for p in model.parameters()) / 1e6, 2)
    # rollout latency
    x1 = Xho[:1]
    for _ in range(3):
        model.rollout(x1, head_w, K_roll=8)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(10):
        model.rollout(x1, head_w, K_roll=32)
    torch.cuda.synchronize()
    ev["rollout_latency_ms_K32"] = round(
        (time.perf_counter() - t1) / 10 * 1e3, 1)
    print(json.dumps(ev, indent=1))
    OUT.write_text(json.dumps({
        "classification": "RECURRENT_TARGET_FEATURE_DRAFTER_R0R2",
        "design": "z0=tanh(seed(h40)); GRU recurrence on token "
                  "embeddings; frozen lm_head batched over K",
        "results": ev}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
