"""Stage-1: one-state multi-token proposer race (roadmap §31-40).

Architectures (all read ONE L40 state [4, 4096] and emit proposals for
horizons k=1..K through the FROZEN lm_head, batched into a single head
GEMM):

  A  independent horizon mixings (hc_head form, 65K x K)
  B  shared trunk transform + per-horizon bias
  C  state-chain recurrent decoder (parallelizable chain, no token
     feedback — token-feedback variants are latency-doomed by the
     per-step head GEMM)

Training: CE on next-token at each horizon, summed over k.
Evaluation (held-out): top-k accuracy by horizon, path survival,
greedy-block E[A] (accepted = common prefix of proposals vs actual
continuation + 1 bonus), proposal latency (one state -> all horizons).

Success metric (roadmap §32): E[A] / (T_proposal + T_verify).
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
OUT = Path("artifacts/stage1_race_result.json")
K_HORIZONS = 32
HC, HIDDEN = 4, 4096
DEV = "cuda"


def load_split(names):
    xs, ys = [], []
    for cat in names:
        f = DATA / f"{cat}.pt"
        if not f.exists():
            continue
        d = torch.load(f, weights_only=False)
        g = d["gen"]                      # list of tokens
        st = d["states"][40].float()      # [pos, 4, 4096] L40
        pos = min(len(g), st.shape[0])
        for p in range(pos):
            # horizon k target = g[p + k - 1]
            if p + K_HORIZONS - 1 < len(g):
                xs.append(st[p])
                ys.append(g[p:p + K_HORIZONS])
    if not xs:
        return None, None
    return (torch.stack(xs), torch.tensor(ys, dtype=torch.long))


class ArchA(torch.nn.Module):
    """Independent per-horizon hc_head-form mixings."""

    def __init__(self, src_fn, src_base, src_scale, K=K_HORIZONS):
        super().__init__()
        self.fn = torch.nn.Parameter(
            src_fn.detach().clone().repeat(K, 1, 1))   # [K, 4, 16384]
        self.base = torch.nn.Parameter(
            src_base.detach().clone().repeat(K, 1))
        self.scale = torch.nn.Parameter(
            src_scale.detach().clone().repeat(K))

    def forward(self, X):                 # X [N, 4, 4096] -> [N, K, 4096]
        N = X.shape[0]
        xf2 = X.flatten(1)                # [N, 16384]
        rs2 = torch.rsqrt(xf2.square().mean(-1, keepdim=True) + 1e-6)
        # per-horizon mixes: fn [K, 4, 16384] x xf [N, 16384]
        mixes = torch.einsum("kcd,nd->nkc", self.fn, xf2) \
            * rs2.unsqueeze(-1)
        pre = torch.sigmoid(mixes * self.scale.unsqueeze(-1)
                            + self.base) + 1e-6
        return torch.einsum("nkc,ncd->nkd", pre, X)


class ArchB(torch.nn.Module):
    """Shared trunk transform + per-horizon bias."""

    def __init__(self, K=K_HORIZONS):
        super().__init__()
        self.trunk = torch.nn.Linear(HC * HIDDEN, HIDDEN, bias=False)
        self.emb = torch.nn.Parameter(torch.zeros(K, HIDDEN))
        torch.nn.init.normal_(self.emb, std=0.02)

    def forward(self, X):
        N = X.shape[0]
        y = self.trunk(X.flatten(1))      # [N, 4096]
        return y.unsqueeze(1) + self.emb.unsqueeze(0)


class ArchC(torch.nn.Module):
    """State-chain decoder (no token feedback; parallel in k)."""

    def __init__(self, d=2048, K=K_HORIZONS):
        super().__init__()
        self.h0 = torch.nn.Linear(HC * HIDDEN, d, bias=False)
        self.cell = torch.nn.Linear(d, d, bias=True)
        self.out = torch.nn.Linear(d, HIDDEN, bias=False)
        self.d = d

    def forward(self, X):
        h = torch.tanh(self.h0(X.flatten(1)))
        outs = []
        for _ in range(K_HORIZONS):
            outs.append(self.out(h))
            h = torch.tanh(h + self.cell(h))
        return torch.stack(outs, dim=1)   # [N, K, 4096]


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
        return f.get_tensor("head.weight").to(DEV)   # bf16 [V, 4096]


def get_src_hc():
    import glob, os
    from safetensors.torch import safe_open
    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--"
        "DeepSeek-V4-Flash-0731/snapshots/*"))[0]
    idx = json.load(open(os.path.join(
        snap, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for nm in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        with safe_open(os.path.join(snap, idx[nm]), framework="pt") as f:
            out[nm] = f.get_tensor(nm).float().to(DEV)
    return out


def train_arch(arch, Xtr, ytr, Xva, yva, head_w, steps=1200, bs=64):
    arch = arch.to(DEV)
    opt = torch.optim.AdamW(arch.parameters(), lr=1e-3, weight_decay=1e-4)
    N = Xtr.shape[0]
    best_va, best_state = -1, None
    HC_CHUNK = 8

    def chunked_step(Xb, yb, backward=True):
        # per-horizon-chunk forward+backward; peak logits ~ [bs*8, V]
        opt.zero_grad()
        total = 0.0
        for kc in range(0, K_HORIZONS, HC_CHUNK):
            Y = arch(Xb)                   # recompute (arch is tiny)
            Yc = Y[:, kc:kc + HC_CHUNK]
            logits = torch.nn.functional.linear(
                Yc.reshape(-1, HIDDEN).to(torch.bfloat16),
                head_w).float()
            loss = torch.nn.functional.cross_entropy(
                logits, yb[:, kc:kc + HC_CHUNK].reshape(-1))
            (loss / (K_HORIZONS // HC_CHUNK)).backward()
            total += loss.item()
            del Y, Yc, logits, loss
        if backward:
            opt.step()
        return total / (K_HORIZONS // HC_CHUNK)

    for step in range(steps):
        idx = torch.randint(0, N, (bs,), device=DEV)
        Xb, yb = Xtr[idx], ytr[idx]        # yb [bs, K]
        chunked_step(Xb, yb)
        if step % 50 == 49:
            with torch.no_grad():
                va = quick_top1(arch, Xva, yva, head_w)
            if va > best_va:
                best_va = va
                best_state = {k: v.detach().clone()
                              for k, v in arch.state_dict().items()}
    if best_state is not None:
        arch.load_state_dict(best_state)
    return arch, best_va


@torch.no_grad()
def quick_top1(arch, X, y, head_w, bs=64):
    hits = tot = 0
    for i in range(0, X.shape[0], bs):
        Xb = X[i:i + bs].to(DEV)
        Y = arch(Xb)[:, :1]                # horizon 1 only
        logits = torch.nn.functional.linear(
            Y.reshape(-1, HIDDEN).to(torch.bfloat16),
            head_w).float()
        pred = logits.argmax(-1)
        hits += (pred == y[i:i + bs, 0].to(DEV)).sum().item()
        tot += Xb.shape[0]
    return hits / tot


@torch.no_grad()
def full_eval(arch, X, y, head_w, name):
    """Per-horizon top-k, path survival, greedy-block E[A], latency.
    Chunked: accumulates only [N, K] argmax + top-4 membership."""
    X = X.to(DEV)
    yt = y.to(DEV)
    Nn = X.shape[0]
    pred_all = torch.empty(Nn, K_HORIZONS, dtype=torch.long, device=DEV)
    top4_all = torch.empty(Nn, K_HORIZONS, 4, dtype=torch.long,
                           device=DEV)
    B = 64
    for i in range(0, Nn, B):
        Y = arch(X[i:i + B])
        for kc in range(0, K_HORIZONS, 8):
            Yc = Y[:, kc:kc + 8]
            lg = torch.nn.functional.linear(
                Yc.reshape(-1, HIDDEN).to(torch.bfloat16),
                head_w).float()
            lg = lg.reshape(-1, 8, lg.shape[-1])
            tv, ti = lg.topk(4, dim=-1)
            pred_all[i:i + B, kc:kc + 8] = ti[:, :, 0]
            top4_all[i:i + B, kc:kc + 8] = ti
    # rescore with a logits-free view: top1 + top4 membership
    res = {"top1_by_k": {}, "top4_by_k": {}, "survival_by_k": {}}
    for k in (1, 2, 4, 8, 16, 24, 32):
        j = k - 1
        res["top1_by_k"][k] = round(
            (pred_all[:, j] == yt[:, j]).float().mean().item(), 4)
        res["top4_by_k"][k] = round(
            (top4_all[:, j] == yt[:, j].unsqueeze(1)).any(-1)
            .float().mean().item(), 4)
        surv = torch.ones(Nn, dtype=torch.bool, device=DEV)
        for jj in range(k):
            surv &= pred_all[:, jj] == yt[:, jj]
        res["survival_by_k"][k] = round(
            surv.float().mean().item(), 4)
    # greedy-block E[A]: accepted = common prefix + 1
    proposals = pred_all                  # [N, K]
    match = (proposals == yt)
    acc = torch.zeros(X.shape[0], device=DEV)
    for jj in range(K_HORIZONS):
        ok = match[:, jj]
        acc += ok.float()
        if not bool(ok.all()):
            # positions that failed stop accumulating
            match = match & ok.unsqueeze(1)
    ea = (acc + 1).clamp(max=K_HORIZONS).mean().item()
    res["E_A_greedy_block"] = round(ea, 3)
    # latency: one state -> all horizons
    arch.eval()
    x1 = X[:1].to(DEV)
    for _ in range(5):
        Y1 = arch(x1)
        _ = torch.nn.functional.linear(
            Y1.reshape(-1, HIDDEN).to(torch.bfloat16), head_w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        Y1 = arch(x1)
        lg = torch.nn.functional.linear(
            Y1.reshape(-1, HIDDEN).to(torch.bfloat16), head_w)
        _ = lg.argmax(-1)
    torch.cuda.synchronize()
    res["proposal_latency_ms"] = round(
        (time.perf_counter() - t0) / 20 * 1e3, 1)
    res["params_millions"] = round(sum(
        p.numel() for p in arch.parameters()) / 1e6, 2)
    return res


def main():
    spec = json.loads(PROMPTS.read_text())
    splits = spec["splits"]
    Xtr, ytr = load_split(splits["train"])
    Xva, yva = load_split(splits["val"])
    Xho, yho = load_split(splits["test"])
    if Xtr is None or Xho is None:
        print("insufficient data recorded yet")
        return
    print(f"train {Xtr.shape[0]} / val {Xva.shape[0]} / "
          f"test {Xho.shape[0]} state windows")
    head_w = load_head()
    src = get_src_hc()
    Xtr, ytr = Xtr.to(DEV), ytr.to(DEV)
    Xva, yva = Xva.to(DEV), yva.to(DEV)
    Xho, yho = Xho.to(DEV), yho.to(DEV)

    archs = {
        "A_horizon_mixings": ArchA(src["hc_head_fn"], src["hc_head_base"],
                                   src["hc_head_scale"]),
        "B_shared_trunk": ArchB(),
        "C_state_chain": ArchC(),
    }
    results = {}
    for name, arch in archs.items():
        t0 = time.time()
        arch, va = train_arch(arch, Xtr, ytr, Xva, yva, head_w)
        ev = full_eval(arch, Xho, yho, head_w, name)
        ev["val_top1_h1"] = round(va, 4)
        ev["train_seconds"] = round(time.time() - t0)
        results[name] = ev
        t1 = ev["top1_by_k"].get(1, ev["top1_by_k"].get("1"))
        t32 = ev["top1_by_k"].get(32, ev["top1_by_k"].get("32"))
        s16 = ev["survival_by_k"].get(16, ev["survival_by_k"].get("16"))
        print(f"[{name}] E[A]={ev['E_A_greedy_block']} "
              f"top1@1={t1} top1@32={t32} "
              f"surv@16={s16} "
              f"lat={ev['proposal_latency_ms']}ms "
              f"params={ev['params_millions']}M", flush=True)
    out = {
        "classification": "STAGE1_ONE_STATE_MULTITOKEN_RACE",
        "dataset": "stage1_prompts.json 126 prompts x 40 tokens, "
                   "L40 state, K=32 horizons, frozen lm_head",
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
