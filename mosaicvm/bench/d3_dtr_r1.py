"""DTR-R1: shared nonlinear candidate reranker (roadmap
S16-S24). Frozen HIST-1 trunk + frozen WS-1 Q256 candidate
generator; trained: a SMALL shared token scorer.

Per candidate t (WS-1 top-M, M=16; true token force-included
during TRAINING only, marked — never counted as production
recall):

    r = P_r(q)          q = Q(z_rms)      [256 -> 128]
    e = P_e(E_t)        E_t = basis row   [256 -> 128]
    f_t = [r, e, r*e, |r-e|, l_base_t]    [513]
    Δ_t = MLP(f_t)                        zero-init final
    s_t = l_base_t + Δ_t

Step-0 (Δ=0) reproduces WS-1 exactly (fail-closed).
Objective: candidate-set CE + 0.5 * candidate KL (teacher
top16 restricted to candidates, renormalized). No hinge, no
trunk unfreeze, no per-token free parameters.

Recipe frozen: AdamW 1e-3/wd 1e-4, cosine 600, CHUNK 512,
clip 1.0, seed 104729.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_dd1_hist import (DDHistNet,
                                        build_hist_examples)
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import load_head, load_embed
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON
from mosaicvm.bench.d3_ws import (eval_telemetry, splice_eval,
                                  load_a2, load_bridge)

HIST1 = Path("artifacts/d3_dd1_hist/dd1_hist_prev/dd1_hist.pt")
BASIS = Path("artifacts/d3_lrdl0/basis_r256.pt")
WS1 = Path("artifacts/d3_ws/ws1.pt")
WS1_EVAL = Path("artifacts/d3_ws/ws1_eval.json")
OUTDIR = Path("artifacts/d3_dtr_r1")
M_CAND = 16


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Reranker(nn.Module):
    def __init__(self, d_root=256, d_tok=256, d=128,
                 hidden=256):
        super().__init__()
        self.p_r = nn.Linear(d_root, d)
        self.p_e = nn.Linear(d_tok, d)
        self.mlp = nn.Sequential(
            nn.Linear(d * 4 + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        # zero-init final layer: Δ = 0 at step 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, q, e_cand, l_base):
        """q [B,256], e_cand [B,C,256], l_base [B,C]
        -> scores [B,C] = l_base + Δ."""
        r = self.p_r(q).unsqueeze(1)          # [B,1,d]
        e = self.p_e(e_cand)                  # [B,C,d]
        f = torch.cat([r.expand(-1, e.shape[1], -1),
                       e, r * e, (r - e).abs(),
                       l_base.unsqueeze(-1)], -1)
        return l_base + self.mlp(f).squeeze(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(104729)
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- frozen stack ----
    ck = torch.load(HIST1, map_location="cpu",
                    weights_only=True)
    hist = DDHistNet(ck["hidden"], ck["depth"], ck["aux_dim"])
    hist.load_state_dict(ck["model"])
    hist.eval()
    for p_ in hist.parameters():
        p_.requires_grad_(False)
    xm_mean, xm_std = ck["xm_mean"], ck["xm_std"]
    xa_mean, xa_std = ck["xa_mean"], ck["xa_std"]

    ws1 = torch.load(WS1, map_location="cpu",
                     weights_only=True)
    assert ws1["hist1_ckpt_sha256"] == sha(HIST1)
    assert ws1["basis_sha256"] == sha(BASIS)
    Q = ws1["Q"].cuda()
    bk = torch.load(BASIS, map_location="cpu",
                    weights_only=True)
    E_r = bk["E_r"]
    E_g = E_r.cuda()

    w_head, w_norm = load_head()
    w_head_g = w_head.cuda()
    w_norm_g = w_norm.cuda()
    embed = load_embed()

    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    print(f"{len(val)} DEV prompts ({time.time()-t0:.0f}s)",
          flush=True)

    eval_telemetry.xm_mean = xm_mean
    eval_telemetry.xm_std = xm_std
    eval_telemetry.xa_mean = xa_mean
    eval_telemetry.xa_std = xa_std
    splice_eval.xm_mean = xm_mean
    splice_eval.xm_std = xm_std
    splice_eval.xa_mean = xa_mean
    splice_eval.xa_std = xa_std

    @torch.inference_mode()
    def z_rms_of(mfn, afn):
        z = hist(mfn.unsqueeze(0), afn.unsqueeze(0))
        zg = z.cuda()
        return zg * torch.rsqrt(
            zg.square().mean(-1, keepdim=True) + 1e-6)

    @torch.inference_mode()
    def ws1_base(mfn, afn):
        zr = z_rms_of(mfn, afn)
        return (zr @ Q.t()) @ E_g.t()      # [1, V]

    # ---- step-0 equivalence: Δ=0 ⇒ rerank = WS-1 ----
    model = Reranker().cuda()
    ws1_ev = json.loads(WS1_EVAL.read_text())
    ref_pp = ws1_ev["per_prompt"]
    c0 = val[0]; S0 = c0["S"]
    mf0 = make_dd_feat(embed,
                       c0["h40"][S0 - 1].float().flatten(),
                       c0["tokens"][S0], c0["tokens"][S0 - 1],
                       S0)
    mfn0 = (mf0 - xm_mean) / xm_std
    afn0 = ((c0["h40"][S0 - 2].float().flatten() - xa_mean)
            / xa_std)
    with torch.inference_mode():
        lg0 = ws1_base(mfn0, afn0)[0]
        top16 = lg0.topk(M_CAND)             # 1D: [16]
        q0 = (z_rms_of(mfn0, afn0) @ Q.t())   # [1,256]
        s0 = model(q0, E_g[top16.indices].unsqueeze(0),
                   top16.values.unsqueeze(0))
        bitwise0 = torch.equal(s0[0], top16.values)

    hits0 = 0; all_hits = True; max_md = 0.0
    with torch.inference_mode():
        for c in val:
            S = c["S"]; toks = c["tokens"]
            mf = make_dd_feat(embed,
                              c["h40"][S - 1].float().flatten(),
                              toks[S], toks[S - 1], S)
            mfn = (mf - xm_mean) / xm_std
            afn = ((c["h40"][S - 2].float().flatten()
                    - xa_mean) / xa_std)
            lg = ws1_base(mfn, afn)[0]
            t16 = lg.topk(M_CAND)      # 1D
            q = z_rms_of(mfn, afn) @ Q.t()
            s = model(q, E_g[t16.indices].unsqueeze(0),
                      t16.values.unsqueeze(0))
            # top-1 of candidates = global top-1 (top-1 is
            # always in top-M); compare against WS-1 decision
            pred = int(t16.indices[int(s[0].argmax())])
            true = toks[S + 1]
            hit = pred == true
            hits0 += hit
            if hit != ref_pp[c["category"]]["hit"]:
                all_hits = False
    step0_pass = bitwise0 and hits0 == 10 and all_hits
    print(f"step-0: bitwise={bitwise0} S1={hits0}/21 "
          f"{'PASS' if step0_pass else 'FAIL'}", flush=True)
    if not step0_pass:
        raise RuntimeError("step-0 equivalence failed")

    # ---- training data: candidates + forced true token ----
    xm, xa, labels, _, _ = build_hist_examples("h40", "prev")
    n = xm.shape[0]
    xmn = (xm - xm_mean) / xm_std
    xan = (xa - xa_mean) / xa_std
    y_idx = torch.tensor([m["y"] for m in labels])
    top_idx_t = torch.tensor([m["top_idx"] for m in labels])
    p_t = torch.stack([
        F.softmax(torch.tensor(m["top_val"]), dim=0)
        for m in labels])

    print(f"caching q + candidates for {n} rows...",
          flush=True)
    CMAX = M_CAND + 1
    cand_idx = torch.zeros(n, CMAX, dtype=torch.long)
    cand_val = torch.zeros(n, CMAX)
    cand_mask = torch.zeros(n, CMAX)
    y_pos = torch.zeros(n, dtype=torch.long)
    naturally = torch.zeros(n, dtype=torch.bool)
    q_cache = torch.zeros(n, 256)
    teach_p = torch.zeros(n, CMAX)   # renormalized teacher
    with torch.inference_mode():
        for i in range(0, n, 256):
            sl = slice(i, min(i + 256, n))
            z = hist(xmn[sl], xan[sl]).cuda()
            zr = z * torch.rsqrt(
                z.square().mean(-1, keepdim=True) + 1e-6)
            qc = zr @ Q.t()                    # [B,256]
            q_cache[sl] = qc.cpu()
            lg = qc @ E_g.t()                  # [B, V]
            tv, ti = lg.topk(M_CAND, -1)       # [B,16]
            for b in range(sl.stop - sl.start):
                row = sl.start + b
                y = int(y_idx[row])
                idxs = ti[b].tolist()
                vals = tv[b].tolist()
                if y in idxs:
                    naturally[row] = True
                    pos = idxs.index(y)
                    c_idx = idxs
                    c_val = vals
                else:
                    c_idx = idxs + [y]
                    c_val = vals + [float(lg[b, y])]
                    pos = len(c_idx) - 1
                L = len(c_idx)
                cand_idx[row, :L] = torch.tensor(c_idx)
                cand_val[row, :L] = torch.tensor(c_val)
                cand_mask[row, :L] = 1.0
                y_pos[row] = pos
                # teacher KL: renormalize teacher top16 over
                # the candidate intersection
                tset = {t: j for j, t in enumerate(
                    top_idx_t[row].tolist())}
                pr = p_t[row]
                inter = [(j, t) for t, j in tset.items()
                         if t in c_idx]
                if inter:
                    tot = sum(float(pr[j]) for j, _ in inter)
                    if tot > 1e-9:
                        for j, t in inter:
                            teach_p[row, c_idx.index(t)
                                    ] = float(pr[j]) / tot
    print(f"cached ({time.time()-t0:.0f}s); naturally "
          f"retrieved: {int(naturally.sum())}/{n}",
          flush=True)

    cand_idx_g = cand_idx.cuda()
    cand_val_g = cand_val.cuda()
    cand_mask_g = cand_mask.cuda()
    y_pos_g = y_pos.cuda()
    teach_g = teach_p.cuda()
    q_cache_g = q_cache.cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=600, eta_min=1e-3 * 0.02)
    LAM_KL = 0.5
    CHUNK = 512
    telem = []
    for step in range(1, 601):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, n, CHUNK):
            sl = slice(i, min(i + CHUNK, n))
            b = sl.stop - sl.start
            e_c = E_g[cand_idx_g[sl]]           # [B,C,256]
            s = model(q_cache_g[sl], e_c,
                      cand_val_g[sl])
            # mask invalid slots
            s = s + (cand_mask_g[sl] - 1) * 1e9
            ce = F.cross_entropy(
                s, y_pos_g[sl], reduction="none")
            logp = F.log_softmax(s, -1)
            kl = (teach_g[sl] * (teach_g[sl].clamp_min(
                1e-9).log() - logp)).sum(-1)
            row = ce + LAM_KL * kl
            part = row.mean() * (b / n)
            part.backward()
            total += part.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 600:
            with torch.no_grad():
                accs = []
                for i in range(0, n, 1024):
                    sl = slice(i, min(i + 1024, n))
                    e_c = E_g[cand_idx_g[sl]]
                    s = model(q_cache_g[sl], e_c,
                              cand_val_g[sl])
                    s = s + (cand_mask_g[sl] - 1) * 1e9
                    accs.append((s.argmax(-1) == y_pos_g[sl])
                                .float().mean().item())
            telem.append({"step": step,
                          "loss": round(total, 4),
                          "train_cand_acc": round(
                              sum(accs) / len(accs), 4)})
            print(f"step {step}: L {total:.4f} cand-acc "
                  f"{telem[-1]['train_cand_acc']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- end-to-end eval: REAL retrieval + rerank ----
    model.eval()
    per_prompt = {}
    wall_ms = []
    rerank_correct = 0; retrieved = 0
    with torch.inference_mode():
        for c in val:
            S = c["S"]; toks = c["tokens"]
            mf = make_dd_feat(embed,
                              c["h40"][S - 1].float().flatten(),
                              toks[S], toks[S - 1], S)
            mfn = (mf - xm_mean) / xm_std
            afn = ((c["h40"][S - 2].float().flatten()
                    - xa_mean) / xa_std)
            tw = time.perf_counter()
            lg = ws1_base(mfn, afn)[0]
            t16 = lg.topk(M_CAND)      # 1D
            q = z_rms_of(mfn, afn) @ Q.t()
            s = model(q, E_g[t16.indices].unsqueeze(0),
                      t16.values.unsqueeze(0))
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - tw)
                           * 1000)
            pred = int(t16.indices[int(s[0].argmax())])
            true = toks[S + 1]
            tgt_in = bool(true in t16.indices.tolist())
            retrieved += tgt_in
            rerank_correct += (pred == true and tgt_in)
            lgm = s[0].clone()
            lgm[int(s[0].argmax())] = float("-inf")
            margin = float(s[0].max() - lgm.max())
            per_prompt[c["category"]] = {
                "hit": pred == true,
                "pred": pred, "true": true,
                "retrieved": tgt_in,
                "base_rank_of_true": int(
                    (lg > lg[true]).sum().item()) + 1,
                "margin": round(margin, 4),
                "naturally_reranked":
                    pred != int(t16.indices[0]),
            }
    n_p = len(per_prompt)
    s1 = sum(v["hit"] for v in per_prompt.values())
    agg = {
        "S1": s1 / n_p,
        "recall_at_16": retrieved / n_p,
        "rerank_acc_given_retrieved":
            rerank_correct / max(retrieved, 1),
        "wall_ms_mean": round(sum(wall_ms) / len(wall_ms),
                              3),
        "naturally_reranked_top1": sum(
            v["naturally_reranked"]
            for v in per_prompt.values()),
    }
    dom = {}
    for cat, v in per_prompt.items():
        dom.setdefault(cat.rsplit("_", 1)[0],
                       []).append(v["hit"])
    agg["per_domain_S1"] = {
        d: round(sum(vs) / len(vs), 4)
        for d, vs in sorted(dom.items())}
    print(f"\n[DTR-R1] end-to-end: S1={s1}/{n_p} "
          f"recall@16={agg['recall_at_16']:.3f} "
          f"rerank|retr={agg['rerank_acc_given_retrieved']:.3f}"
          f" wall={agg['wall_ms_mean']}ms", flush=True)
    hits = sorted(c for c, v in per_prompt.items()
                  if v["hit"])
    print(f"[DTR-R1] hits: {hits}", flush=True)

    # ---- splice if S1 >= 10 ----
    @torch.inference_mode()
    def dtr_logits(mfn, afn):
        lg = ws1_base(mfn, afn)[0]
        t16 = lg.topk(M_CAND)      # 1D
        q = z_rms_of(mfn, afn) @ Q.t()
        s = model(q, E_g[t16.indices].unsqueeze(0),
                  t16.values.unsqueeze(0))
        # full-vocab-shaped logits for telemetry helpers:
        # reranked candidates, others -inf
        out = torch.full_like(lg, float("-inf"))
        out[t16.indices] = s[0]
        return out

    if s1 >= 10:
        a2, t_mean, t_std, a2_xm, a2_xs = load_a2(
            Path("artifacts/d3_v1_A2/student.pt"))
        bridge, b_xm, b_xs = load_bridge(
            Path("artifacts/d3_bridge_r3b/"
                 "bridge_ps/bridge.pt"))
        dd1_hits, cs, cond = splice_eval(
            val, embed, dtr_logits, a2, t_mean, t_std,
            a2_xm, a2_xs, bridge, b_xm, b_xs, w_head_g,
            w_norm_g, t0)
        print(f"[DTR-R1] splice: mean={cs['mean']} "
              f"S1={dd1_hits}/{len(val)} S4={cs['S4']} "
              f"deep={cs['deepest']}", flush=True)
    else:
        dd1_hits, cs, cond = 0, {}, {}
        print("[DTR-R1] S1 < 10/21 — splice skipped",
              flush=True)

    result = {
        "classification": "D3_DTR_R1",
        "M": M_CAND,
        "step0_equivalence": {
            "bitwise": bool(bitwise0),
            "S1": f"{hits0}/21", "pass": step0_pass},
        "train_telemetry": telem,
        "natural_retrieval_train":
            f"{int(naturally.sum())}/{n}",
        "aggregate": agg, "per_prompt": per_prompt,
        "dd1_splice_S1_count":
            f"{dd1_hits}/{len(val)}" if cs else None,
        "canonical_stats": cs,
        "conditional_survival": cond,
        "gates": {
            "promote_S1_ge_11": s1 >= 11,
            "strong_S1_ge_12": s1 >= 12,
            "regress_S1_le_9": s1 <= 9},
        "frozen_refs": {
            "hist1": sha(HIST1), "basis": sha(BASIS),
            "ws1": sha(WS1)},
    }
    torch.save({"model": model.state_dict(), "M": M_CAND,
                "frozen_refs": result["frozen_refs"]},
               OUTDIR / "dtr_r1.pt")
    (OUTDIR / "dtr_r1_eval.json").write_text(
        json.dumps(result, indent=1))
    print(f"\ngates: {json.dumps(result['gates'])}")
    print(f"wrote {OUTDIR}/dtr_r1.pt, dtr_r1_eval.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
