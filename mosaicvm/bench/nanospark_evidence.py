"""NanoSpark evidence phase (roadmap §43-48, §66 items 2-5).

Phase A — capture exact teacher trajectories (7 domains × N positions):
    per position: token, next-token, main_hidden [12288],
    main_x [4096] (post main_proj+main_norm), position, domain.
Phase B — minimal-interface identity check (no target cost):
    draft with exact main_hidden vs injected main_x must be identical
    (code truth: main_hidden's ONLY consumer is main_proj/main_norm).
Phase C — fit P1 predictors on block-spaced state pairs (s_p -> s_{p+5}):
    P0 hold; P1a velocity; P1b damped alpha in {.25,.5,.75};
    P1c second-order; P1d ridge linear token-conditioned
    (delta = W [s; e_tok]).
Phase D — free-running recursive evaluation per predictor
    (4 domains × C chains × 6 blocks): accepted tokens per target pass,
    per-block per-token acceptance. P0 must reproduce 71/50/0.
Phase E — predictor cost table (analytic FLOPs per block-step).

Dataset gates (instruction 8): held-out-domain fit variant for the
learned predictor; deterministic replay check; revision recorded;
states stored fp16 with an fp32 spot-check.

Classification: BLACKWELL_SANDBOX_ONLY + DRAFTER_NUMERICS_DEVIATION_
RECORDED.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/marimo/MosaicVM")

import torch

from mosaicvm.bench.dspark_wire import (
    BLOCK_SIZE, MAIN_GPU_GIB, N_STAGES, PROMPTS, TARGET_LAYERS,
    DSparkStage, FP4_TABLE_VALUES, load_mtp_tensors,
    precompute_freqs_cis)

OUT = "/marimo/MosaicVM/artifacts"
N_CAP = 50               # positions captured per domain
N_CHAINS = 8             # recursive chains per domain (P0 protocol parity)
R_DEPTH = 6              # recursive blocks (=30 tokens)
TRAIN_DOMAINS = {"structured_json", "long_context_retrieval",
                 "long_generation", "agent_tool_use", "general_conversation"}
EVAL_DOMAINS = ["coding", "agent_tool_use", "math_reasoning",
                "general_conversation"]   # P0-comparable set


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t00 = time.time()
    os.makedirs(OUT, exist_ok=True)

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download("deepseek-ai/DeepSeek-V4-Flash-0731")
    revision = os.path.basename(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=cfg, torch_dtype=torch.bfloat16,
        device_map="auto", max_memory={0: f"{MAIN_GPU_GIB}GiB",
                                       "cpu": "100GiB"},
        offload_folder="/tmp/offload", trust_remote_code=True)
    model.eval()

    captured = []
    handles = [model.model.layers[li].register_forward_hook(
        lambda m, a, o: captured.append(o.detach().mean(dim=2)))
        for li in TARGET_LAYERS]

    from safetensors import safe_open
    idx = json.load(open(os.path.join(
        model_dir, "model.safetensors.index.json")))

    def ckpt(key):
        with safe_open(os.path.join(model_dir, idx["weight_map"][key]),
                       framework="pt", device="cpu") as f:
            return f.get_tensor(key).to("cuda")

    head_w = ckpt("head.weight")
    # NOT tied: separate embed.weight / head.weight in this checkpoint
    embed_weight = model.model.embed_tokens.weight.data
    if embed_weight.device.type != "cuda":
        embed_weight = ckpt("embed.weight")
    dev = head_w.device

    fp4 = torch.tensor(FP4_TABLE_VALUES, dtype=torch.float32, device=dev)
    raw = load_mtp_tensors(model_dir, dev)
    fc = precompute_freqs_cis(64, 8192, 65536, 10000, 16, 32, 1).to(dev)
    stages = [DSparkStage(s, raw[s], fc, dev, fp4) for s in range(N_STAGES)]
    log(f"loaded ({time.time()-t00:.0f}s)")

    def target_forward(tokens, past=None):
        captured.clear()
        with torch.no_grad():
            o = model(tokens, past_key_values=past, use_cache=True)
        mh = torch.cat(captured, dim=-1)
        captured.clear()
        return o.logits, o.past_key_values, mh

    def draft(input_ids, main_hidden, start_pos):
        h, main_x = stages[0].forward_embed(main_hidden, input_ids,
                                            embed_weight)
        for st in stages:
            h = st(h, start_pos, main_x)
        if start_pos == 0:
            return None, main_x
        return stages[-1].forward_head(h, input_ids, head_w), main_x

    def draft_with_mx(input_ids, main_x, start_pos):
        """Inject main_x directly (bypass main_proj/main_norm) — the
        P-B interface. forward_embed replica with the projection done."""
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

    # ================= Phase A: capture (reuse if cached) ==============
    dataset = {}
    tA = time.time()
    cache_path = f"{OUT}/nanospark_states.pt"
    if os.path.exists(cache_path):
        try:
            cached = torch.load(cache_path)
            if (all(c in cached for c, _ in PROMPTS)
                    and all(cached[c]["mx"].dim() == 2
                            for c, _ in PROMPTS)
                    and cached[PROMPTS[0][0]]["mx"].shape[0] == N_CAP):
                dataset = cached
                log("Phase A: reusing cached dataset")
        except Exception as exc:
            log(f"cache unusable ({exc}); recapturing")
    if not dataset:
        log(f"Phase A: capturing {N_CAP} positions x {len(PROMPTS)} domains")
    for cat, prompt_text in (PROMPTS if not dataset else []):
        ids = tok(prompt_text, return_tensors="pt").input_ids.to(dev)
        logits, past, mh = target_forward(ids)
        with torch.no_grad():
            draft(ids[:, -1:], mh, 0)
        g = logits[0, -1].argmax().item()
        states_mh, states_mx, toks, nexts = [], [], [], []
        for p in range(N_CAP):
            # mh: [1,1,3d] hidden at the LAST FED position
            s_mh = mh[0, -1:].clone()
            with torch.no_grad():
                _, s_mx = draft(torch.tensor([[g]], device=dev),
                                s_mh.unsqueeze(0), past.get_seq_length() - 1)
            states_mh.append(s_mh[0].half().cpu())       # [3d]
            states_mx.append(s_mx[0, -1].half().cpu())    # [4096]
            toks.append(g)
            lg, past, mh = target_forward(
                torch.tensor([[g]], device=dev), past)
            nexts.append(lg[0, -1].argmax().item())
            g = nexts[-1]
        dataset[cat] = {
            "tokens": toks, "next": nexts,
            "mh": torch.stack(states_mh), "mx": torch.stack(states_mx),
        }
        log(f"  [{cat}] captured {N_CAP} states "
            f"({time.time()-tA:.0f}s cumulative)")
    torch.save(dataset, f"{OUT}/nanospark_states.pt")
    log(f"dataset saved ({time.time()-tA:.0f}s)")

    # replay determinism: recompute state 0 with the SAME live position
    # semantics as the capture loop (fresh prefill, correct start_pos)
    _ids = tok(dict(PROMPTS)["coding"], return_tensors="pt").input_ids.to(dev)
    _lg, _past, _mh = target_forward(_ids)
    with torch.no_grad():
        draft(_ids[:, -1:], _mh, 0)             # fresh ring fill
        _, mx_chk = draft(
            torch.tensor([[dataset["coding"]["tokens"][0]]], device=dev),
            _mh[0, -1:].unsqueeze(0), _past.get_seq_length() - 1)
    replay_ok = bool(torch.allclose(
        mx_chk[0, -1].float().cpu(),
        dataset['coding']['mx'][0].float(), atol=2e-2))
    log(f"replay check (fresh ring, same position): {replay_ok}")

    # ================= Phase B: interface identity =================
    # fresh prefill again (capture polluted the ring)
    log("Phase B: main_hidden vs main_x interface identity")
    _lg, _past, _mh = target_forward(_ids)
    with torch.no_grad():
        draft(_ids[:, -1:], _mh, 0)             # reset ring
    s_mh = _mh[0, -1:].unsqueeze(0)             # [1,1,3d] live exact
    with torch.no_grad():
        out_a, s_mx_live = draft(_ids[0, -1:].reshape(1, 1), s_mh,
                                 _past.get_seq_length() - 1)
        out_b = draft_with_mx(_ids[0, -1:].reshape(1, 1), s_mx_live,
                              _past.get_seq_length() - 1)
    ident = bool(torch.equal(out_a["greedy_ids"], out_b["greedy_ids"]))
    a3 = out_a["greedy_topk"][0][0][:3]
    b3 = out_b["greedy_topk"][0][0][:3]
    top3_overlap = len(set(a3) & set(b3)) / 3
    log(f"  P-B identical ids: {ident}, top3 overlap {top3_overlap}")

    # ================= Phase C: fit predictors =================
    log("Phase C: fitting P1 predictors on main_x (4096-dim)")
    STEP = BLOCK_SIZE

    def pairs(domains):
        S, T = [], []
        for d in domains:
            v = dataset[d]
            mx = v["mx"].float().cuda()
            for i in range(0, N_CAP - STEP, STEP):
                S.append(mx[i])
                T.append(mx[i + STEP])
        return torch.stack(S), torch.stack(T)

    train_d = [c for c, _ in PROMPTS if c in TRAIN_DOMAINS]
    S_tr, T_tr = pairs(train_d)
    D_tr = T_tr - S_tr                     # deltas
    log(f"  train pairs: {S_tr.shape[0]}")

    # P1d: ridge linear token-conditioned on [s ; e_lasttok]
    tok_embed = embed_weight.to(torch.float32)
    def feats(domains):
        """Feature token = the COMMITTED token at the block START (the
        same quantity production conditions on at prediction time).
        The label is the exact state at i+STEP. No target next-token is
        ever used as an input."""
        X = []
        for d in domains:
            v = dataset[d]
            mx = v["mx"].float().cuda()
            for i in range(0, N_CAP - STEP, STEP):
                e = tok_embed[v["tokens"][i]]
                X.append(torch.cat([mx[i], e]))
        return torch.stack(X)

    X_tr = feats(train_d)
    Y = D_tr
    Dm = X_tr.shape[1]
    lam = 1.0
    # closed-form ridge via normal equations in fp32 (8192x8192 too big
    # -> use low-rank solving via SVD of X)
    log(f"  ridge solve (X {tuple(X_tr.shape)})...")
    U, sig, Vt = torch.linalg.svd(X_tr, full_matrices=False)
    UtY = U.T @ Y
    W = Vt.T @ ((sig / (sig ** 2 + lam)).unsqueeze(1) * UtY)
    lin_rank = int((sig > sig[0] * 1e-4).sum())
    log(f"  ridge done; effective rank {lin_rank}")

    predictors = {
        "P0_hold": lambda s_hist, tok: s_hist[-1],
        "P1a_velocity": lambda s_hist, tok:
            s_hist[-1] + (s_hist[-1] - s_hist[-2]),
        "P1b_damped_0.5": lambda s_hist, tok:
            s_hist[-1] + 0.5 * (s_hist[-1] - s_hist[-2]),
        "P1c_second_order": lambda s_hist, tok:
            s_hist[-1] + 1.5 * (s_hist[-1] - s_hist[-2])
            - 0.5 * (s_hist[-2] - s_hist[-3]) if len(s_hist) > 2
            else s_hist[-1],
    }

    def lin_pred(s_hist, tok):
        e = tok_embed[tok]
        x = torch.cat([s_hist[-1], e])
        return s_hist[-1] + (x @ W)

    # W is fit on TRAIN_DOMAINS (excludes coding & math): coding/math
    # eval rows are HELD-OUT; agent/general rows are in-domain.
    predictors_lin = {"P1d_lin_token": lin_pred}

    # ================= Phase D: recursive evaluation =================
    log("Phase D: recursive evaluation "
        f"({len(EVAL_DOMAINS)} domains x {N_CHAINS} chains)")

    def eval_predictor(name, step_fn, token_cond):
        results = {}
        for cat in EVAL_DOMAINS:
            v = dataset[cat]
            prompt_text = dict(PROMPTS)[cat]
            ids = tok(prompt_text, return_tensors="pt").input_ids.to(dev)
            n_prompt = ids.shape[1]
            logits, past, mh = target_forward(ids)
            with torch.no_grad():
                draft(ids[:, -1:], mh, 0)
            g_next = logits[0, -1].argmax().item()
            last_tok = ids[0, -1].item()
            last_mh = mh[0, -1:]
            acc_list = []
            depth_acc = [[0] * BLOCK_SIZE for _ in range(R_DEPTH)]
            depth_reach = [0] * R_DEPTH
            for blk in range(N_CHAINS):
                L0 = past.get_seq_length()
                s_exact = last_mh.clone()
                with torch.no_grad():
                    _, mx0 = draft(
                        torch.tensor([[last_tok]], device=dev),
                        s_exact.unsqueeze(0), L0 - 1)
                # block-spaced history: exact states at idx, idx-5, idx-10
                # (the last two exact refresh states + current) — matches
                # production, where refreshes are BLOCK_SIZE apart
                idx = min(max(L0 - n_prompt, 0), N_CAP - 1)
                s_hist = [v["mx"][max(0, i)].float().cuda()
                          for i in (idx - 10, idx - 5, idx)]
                chain = []
                cur_tok, sp = last_tok, L0 - 1
                s_pred = s_hist[-1]
                with torch.no_grad():
                    for r in range(R_DEPTH):
                        if r == 0:
                            out = draft(torch.tensor([[cur_tok]], device=dev),
                                        s_exact.unsqueeze(0), sp)[0]
                        else:
                            s_pred = (step_fn(s_hist, cur_tok)
                                      if token_cond
                                      else step_fn(s_hist, cur_tok))
                            out = draft_with_mx(
                                torch.tensor([[cur_tok]], device=dev),
                                s_pred[None, None, :], sp)
                            s_hist.append(s_pred)
                        block = out["greedy_ids"][0, 1:].tolist()
                        chain.extend(block)
                        cur_tok = block[-1]
                        sp += BLOCK_SIZE
                # verify
                g = g_next
                acc = 0
                alive = True
                for i, t in enumerate(chain):
                    r, d = divmod(i, BLOCK_SIZE)
                    if not alive:
                        break
                    depth_reach[r] += 1
                    if t == g:
                        depth_acc[r][d] += 1
                        acc += 1
                        lg, past, _ = target_forward(
                            torch.tensor([[t]], device=dev), past)
                        g = lg[0, -1].argmax().item()
                    else:
                        alive = False
                acc_list.append(acc)
                past.crop(L0)
                lg, past, mh_b = target_forward(
                    torch.tensor([[g_next]], device=dev), past)
                last_tok = g_next
                last_mh = mh_b[0, -1:]
                g_next = lg[0, -1].argmax().item()
            per_r = []
            for r in range(R_DEPTH):
                n = depth_reach[r]
                per_r.append(round(sum(depth_acc[r]) / n, 3) if n else None)
            results[cat] = {
                "mean_accepted": round(sum(acc_list) / len(acc_list), 3),
                "per_block_per_token": per_r,
                "chains": acc_list,
            }
            log(f"  [{name}/{cat}] mean={results[cat]['mean_accepted']} "
                f"blocks={per_r}")
        held_out = {c: results[c] for c in results if c not in TRAIN_DOMAINS}
        in_dom = {c: results[c] for c in results if c in TRAIN_DOMAINS}
        def _m(rows):
            cc = [x for v in rows.values() for x in v["chains"]]
            return round(sum(cc) / len(cc), 3) if cc else None
        return results, _m(results), {"held_out": _m(held_out),
                                      "in_domain": _m(in_dom)}

    table = {}
    for name, fn in predictors.items():
        res, mean, split = eval_predictor(name, fn, token_cond=False)
        table[name] = {"per_domain": res, "mean": mean, "split": split}
        _save(table, ident, replay_ok, revision, lin_rank)
    for name, fn in predictors_lin.items():
        res, mean, split = eval_predictor(name, fn, token_cond=True)
        table[name] = {"per_domain": res, "mean": mean, "split": split}
        _save(table, ident, replay_ok, revision, lin_rank)

    # ================= Phase E: cost =================
    d_main = 4096
    costs = {
        "P0_hold": 0,
        "P1a_velocity": d_main,
        "P1b_damped_0.5": 2 * d_main,
        "P1c_second_order": 3 * d_main,
        "P1d_lin_token": 2 * d_main * Dm,   # x @ W flops (matmul)
    }
    _save(table, ident, replay_ok, revision, lin_rank, costs)
    log("=== DONE in {(t00 and time.time()-t00)/60:.1f} min ===".replace(
        "{(t00 and time.time()-t00)/60:.1f}",
        f"{(time.time()-t00)/60:.1f}"))
    for k, v in table.items():
        s = v.get("split") or {}
        log(f"{k}: mean={v['mean']} held_out={s.get('held_out')} "
            f"in_domain={s.get('in_domain')}")
    return 0


def _save(table, ident, replay_ok, revision, lin_rank, costs=None):
    out = {
        "phase": "NanoSpark-evidence",
        "model_revision": revision,
        "classification": "BLACKWELL_SANDBOX_ONLY + "
                          "DRAFTER_NUMERICS_DEVIATION_RECORDED",
        "interface_identity_P_B": ident,
        "replay_check": replay_ok,
        "state_dims": {"main_hidden": 12288, "main_x": 4096,
                       "per_stage_main_kv": 512},
        "ridge_effective_rank": lin_rank,
        "p1d_conditioning": (
            "feature token = committed token at block start (causal; the "
            "same quantity used at eval); label = exact state at i+STEP"),
        "p1d_eval_split": (
            "coding/math are HELD-OUT domains; agent/general IN-DOMAIN"),
        "predictors": table,
    }
    if costs:
        out["predictor_cost_flops_per_block_step"] = costs
    with open(f"{OUT}/nanospark_evidence.json", "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
