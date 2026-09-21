"""B4-O: fixed-width shallow-tree oracle from valid B3-S-RFIX traces.

Replays the EXACT chain drafter semantics along the true greedy path
(from the mirrored 7x210 dataset) WITHOUT the target model:
  - rings committed via write_committed_kv (same projection + RoPE)
  - per-level draft_with_mx with the same speculative-slot semantics
  - state ladder: hold (anchor mx) / rank64 (per-level delta=1 advance)

At every level j along the true path, records the drafter Top-16
candidate set + scores and the rank of the TRUE next token. This gives
the exact conditional tree quantity:
    c_j(W) = P(true in Top-W at level j | path correct through j-1)
and therefore E[A]_oracle(D, W) = sum_{d<=D} prod_{j<=d} c_j(W).

Anchors are restricted to dataset indices >= 128 so every ring slot the
drafter can attend was written by the true-path walk (prompt KVs are
outside the 128-window => no prompt prefill needed, no target needed).

Validation anchors:
  - W=1 ladder should approximate B3-S hold/rank64 D1 E[A]
    (1.679 / 1.643; different anchor set + fp16 dataset states)
  - level-1 coverage should approximate B3-R isolated D1 tables
    (hold Top-1 26.8%, Top-16 66.1%; rank64 24.1%, 70.5%)

Classification: BLACKWELL_SANDBOX_ONLY + DRAFTER_NUMERICS_DEVIATION_
RECORDED (dataset states are fp16-rounded; block-depth>=2 topk are
survivor-conditioned diagnostics only).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/marimo/MosaicVM")

import torch

from mosaicvm.bench.dspark_wire import (
    BLOCK_SIZE, N_STAGES, PROMPTS, TARGET_LAYERS, WINDOW,
    DSparkStage, FP4_TABLE_VALUES, load_mtp_tensors,
    precompute_freqs_cis, _rope_last)

OUT = "/marimo/MosaicVM/mosaicvm/bench/rfix_out"
L_LEVELS = 30                  # levels per anchor ladder (= MAX_DEPTH)
# phase 1: deep anchors, zeroed rings + true-path walk only (no prompt
# needed since every attended slot is walk-written). phase 2: prompt
# prefill rings + early anchors (B3-S-matched validation + full
# position-resolved picture). Analysis merges both.
PHASE = int(os.environ.get("B4O_PHASE", "1"))
if PHASE == 1:
    ANCHOR_LO, ANCHOR_HI = 128, 178
    REC_PREFIX = "b4o_records"
else:
    ANCHOR_LO, ANCHOR_HI = 1, 127
    REC_PREFIX = "b4o_records2"
TRAIN_DOMAINS = {"structured_json", "long_context_retrieval",
                 "long_generation", "agent_tool_use", "general_conversation"}
PREDICTORS = ("hold", "rank64")
TOPK_REC = 16


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _save_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    os.sync()


def main():
    t00 = time.time()
    os.makedirs(OUT, exist_ok=True)

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from safetensors import safe_open

    model_dir = snapshot_download("deepseek-ai/DeepSeek-V4-Flash-0731")
    revision = os.path.basename(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    idx = json.load(open(os.path.join(
        model_dir, "model.safetensors.index.json")))

    def ckpt(key):
        with safe_open(os.path.join(model_dir, idx["weight_map"][key]),
                       framework="pt", device="cpu") as f:
            return f.get_tensor(key).to("cuda")

    head_w = ckpt("head.weight")
    dev = head_w.device
    # embed from the index (same trick as B3-S: try model, else shard)
    embed_key = "model.embed_tokens.weight"
    if embed_key in idx["weight_map"]:
        embed_weight = ckpt(embed_key)
    else:
        embed_weight = ckpt("embed.weight")
    embed_f = embed_weight.to(torch.float32)

    fp4 = torch.tensor(FP4_TABLE_VALUES, dtype=torch.float32, device=dev)
    raw = load_mtp_tensors(model_dir, dev)
    fc = precompute_freqs_cis(64, 8192, 65536, 10000, 16, 32, 1).to(dev)
    stages = [DSparkStage(s, raw[s], fc, dev, fp4) for s in range(N_STAGES)]
    log(f"drafter loaded ({time.time()-t00:.0f}s) — no target model needed")

    # ---- dataset (true greedy path + states) ----
    dataset = torch.load(f"{OUT}/nanospark_states_210.pt")
    N_CAP = dataset[PROMPTS[0][0]]["mx"].shape[0]
    log(f"dataset: {len(dataset)} domains x {N_CAP} positions")

    # ---- rank64 predictors (same training as B3-S) ----
    predictors = {}
    for delta in (1,):
        S_tr, TOK_tr, T_tr = [], [], []
        for d in TRAIN_DOMAINS:
            v = dataset[d]
            mx = v["mx"].float().cuda()
            for i in range(0, N_CAP - delta, delta):
                S_tr.append(mx[i])
                TOK_tr.append(v["tokens"][i])
                T_tr.append(mx[i + delta])
        S_tr = torch.stack(S_tr)
        T_tr = torch.stack(T_tr)
        D_tr = T_tr - S_tr
        X_tr = torch.stack([
            torch.cat([s, embed_f[t]]) for s, t in zip(S_tr, TOK_tr)])
        Xc = X_tr - X_tr.mean(0)
        _, sig, Vt = torch.linalg.svd(Xc, full_matrices=False)
        r = 64
        V_r = Vt[:r].T
        Z = (X_tr - X_tr.mean(0)) @ V_r
        Z_std = Z.std(0).clamp_min(1e-6)
        Zn = (Z - Z.mean(0)) / Z_std
        A = Zn.T @ Zn + 1.0 * torch.eye(r, device=dev)
        U_coef = torch.linalg.solve(A, Zn.T @ (D_tr - D_tr.mean(0)))
        predictors[delta] = {
            "V": V_r, "U": U_coef,
            "Z_raw_mean": Z.mean(0), "D_mean": D_tr.mean(0),
            "x_mean": X_tr.mean(0), "Z_std": Z_std,
        }
    log("rank64 predictor trained (delta=1 ladder)")

    def rank64_pred(s, tok, delta):
        p = predictors[delta]
        x = torch.cat([s, embed_f[tok]])
        z = (x - p["x_mean"]) @ p["V"]
        z = (z - p["Z_raw_mean"]) / p["Z_std"]
        return s + z @ p["U"] + p["D_mean"]

    def hold_pred(s, tok, delta):
        return s

    # ---- drafter helpers (kit-exact, from B3-S) ----
    def draft_with_mx(input_ids, main_x, start_pos):
        ids = input_ids.reshape(-1)
        draft_ids = ids.new_full((ids.size(0), BLOCK_SIZE), 128799)
        draft_ids[:, 0] = ids
        x = torch.nn.functional.embedding(
            draft_ids, embed_weight.to(torch.bfloat16))
        x = x.unsqueeze(2).repeat(1, 1, 4, 1)
        h = x
        for st in stages:
            h = st(h, start_pos, main_x)
        return stages[-1].forward_head(h, input_ids, head_w)

    def write_committed_kv(mh_single, pos):
        """Same projection + per-stage ring write as B3-S-RFIX."""
        with torch.no_grad():
            mx = stages[0].main_norm(
                stages[0].main_proj(mh_single.to(torch.bfloat16)))
            mx_flat = mx[0, 0] if mx.dim() == 3 else mx[0]
            for st in stages:
                main_kv = st.attn.kv_norm(
                    st.attn.wkv(mx_flat[None, None, :]))
                main_kv = _rope_last(
                    main_kv, st.attn.freqs_cis[pos:pos + 1])
                st.attn.kv_cache[pos % WINDOW] = main_kv[0, 0]

    # ---- drafter latency (for T_proposal estimates) ----
    s_test = torch.randn(4096, device=dev, dtype=torch.bfloat16)
    tok_test = torch.tensor([[100]], device=dev)
    for _ in range(5):
        draft_with_mx(tok_test, s_test[None, None, :], 200)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        draft_with_mx(tok_test, s_test[None, None, :], 200)
    torch.cuda.synchronize()
    t_draft_ms = (time.perf_counter() - t0) / 50 * 1e3
    log(f"drafter latency: {t_draft_ms:.2f} ms/call")

    # ---- replay: per domain, per predictor ----
    for cat, _ in PROMPTS:
        v = dataset[cat]
        mh_d = v["mh"].cuda()          # [N, 1, 3*dim] fp16
        mx_d = v["mx"]                 # [N, 4096] fp16 (cpu)
        toks = v["tokens"]             # token at position base-1+i
        n_prompt = v["prompt_len"]
        base = n_prompt - 1            # absolute pos of dataset index 0

        # phase 2: prompt prefill rings (kit semantics, as in B3-S)
        ps_path = f"{OUT}/b4o_prompt_states_{cat}.pt"
        prompt_states = None
        if PHASE == 2:
            prompt_states = torch.load(ps_path)
            mh_full = prompt_states["mh_full"].cuda()  # [1,n,3d] fp16
            with torch.no_grad():
                mx_full = stages[0].main_norm(
                    stages[0].main_proj(mh_full.to(torch.bfloat16)))
            # mx_full: [1, n_prompt, 4096]

        for pred_name in PREDICTORS:
            pred_fn = rank64_pred if pred_name == "rank64" else hold_pred
            rec_path = f"{OUT}/{REC_PREFIX}_{pred_name}_{cat}.pt"
            if os.path.exists(rec_path):
                log(f"  [{cat}/{pred_name}] already captured, skipping")
                continue

            anchors = list(range(ANCHOR_LO, ANCHOR_HI + 1))
            ranks = []       # [n_anchor, L] rank of true token (0-based; -1 miss)
            top_ids = []     # [n_anchor, L, 16]
            top_scores = []  # [n_anchor, L, 16]
            blk_ids = []     # [n_anchor, L, 5, 16] native-block diag
            blk_scores = []

            for i_a in anchors:
                if PHASE == 2:
                    # prefill rings from full-prompt projection
                    # (start_pos=0 fills all prompt slots), then commit
                    # the prompt-end position like the chain
                    with torch.no_grad():
                        draft_with_mx(
                            torch.tensor([[prompt_states["last_tok"]]],
                                         device=dev),
                            mx_full, 0)
                    write_committed_kv(mh_full[0, -1:], base)
                else:
                    # phase 1: zeroed rings; the true-path walk rewrites
                    # every slot the drafter can attend (anchor >= 128)
                    for st in stages:
                        st.attn.kv_cache.zero_()
                # commit true KVs for positions base+1 .. base+i_a
                # (dataset indices 1..i_a)
                for i in range(1, i_a + 1):
                    write_committed_kv(mh_d[i], base + i)

                s = mx_d[i_a].to(dev).to(torch.bfloat16)
                a_ranks, a_ids, a_sc, a_bids, a_bsc = [], [], [], [], []
                for j in range(L_LEVELS):
                    p = base + i_a + j
                    with torch.no_grad():
                        out = draft_with_mx(
                            torch.tensor([[toks[i_a + j]]], device=dev),
                            s[None, None, :], p)
                    gt = out["greedy_topk"]      # list of (ids, scores)
                    lv_ids, lv_sc = gt[0]
                    true_tok = toks[i_a + j + 1]
                    if true_tok in lv_ids:
                        a_ranks.append(lv_ids.index(true_tok))
                    else:
                        a_ranks.append(-1)
                    a_ids.append(lv_ids)
                    a_sc.append(lv_sc)
                    # native-block diagnostics (survivor-conditioned
                    # at depths >= 2): all 5 block depths' topk
                    b_ids, b_sc = [], []
                    for d in range(BLOCK_SIZE):
                        b_ids.append(gt[d][0])
                        b_sc.append(gt[d][1])
                    a_bids.append(b_ids)
                    a_bsc.append(b_sc)
                    # commit true KV at p+1 (matches chain semantics)
                    write_committed_kv(mh_d[i_a + j + 1], p + 1)
                    # advance state (hold: no-op)
                    s = pred_fn(s, toks[i_a + j], 1)

                ranks.append(a_ranks)
                top_ids.append(a_ids)
                top_scores.append(a_sc)
                blk_ids.append(a_bids)
                blk_scores.append(a_bsc)

            rec = {
                "ranks": ranks, "top_ids": top_ids,
                "top_scores": top_scores,
                "block_ids": blk_ids, "block_scores": blk_scores,
                "anchors": anchors, "n_prompt": n_prompt,
                "levels": L_LEVELS,
            }
            _save_atomic(rec, rec_path)

            # quick coverage echo + W=1 ladder validation vs B3-S D1
            lv1_top1 = sum(1 for r in ranks if r[0] == 0) / len(ranks)
            lv1_top16 = sum(1 for r in ranks if 0 <= r[0] < 16) / len(ranks)
            ea1, surv = 0.0, 1.0
            n_a = len(ranks)
            alive = [True] * n_a
            for j in range(L_LEVELS):
                n_alive = sum(alive)
                if n_alive == 0:
                    break
                hits = sum(1 for k in range(n_a)
                           if alive[k] and 0 <= ranks[k][j] < 1)
                surv *= hits / n_alive
                ea1 += surv
                alive = [alive[k] and 0 <= ranks[k][j] < 1
                         for k in range(n_a)]
            log(f"  [{cat}/{pred_name}] n={n_a} "
                f"lv1 Top-1={lv1_top1:.3f} Top-16={lv1_top16:.3f} "
                f"E[A](W=1)={ea1:.3f}")

    log("=== B4-O capture DONE in "
        f"{(time.time()-t00)/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
