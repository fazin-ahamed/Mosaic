"""NanoSpark B1-B4 ladder (roadmap §76 B1-B5, evidence ladder).

B1: expand dataset to 210 positions/domain (batched capture, all 7
    prompts in one greedy decode loop with hooks on layers 40/41/42).
B2: transition diagnostics (cosine, norm drift, abs/rel error by
    domain) + predictor latency measurement.
B3: alpha sweep {0.25, 0.5, 0.75} for damped-velocity.
B4: low-rank residual transition (PCA projection V_r + ridge U) at
    rank 16/32/64, trained on 5 domains, coding+math held out.
B5: block-3 revival gate: held-out block-3 acceptance > 0.

Batched capture design: all 7 prompts left-padded to the same length,
greedy decode one token at a time (batch 7), hooks capture main_hidden
[7,1,3*dim] per step. mx computed offline via main_proj/main_norm.
This is ~7x faster than sequential per-domain capture.

Classification: BLACKWELL_SANDBOX_ONLY + DRAFTER_NUMERICS_DEVIATION_
RECORDED.
"""
from __future__ import annotations

import hashlib
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
N_CAP = 210              # positions per domain (B1: 200+)
N_CHAINS = 4             # recursive chains per domain per predictor
R_DEPTH = 6              # recursive blocks (30 tokens)
TRAIN_DOMAINS = {"structured_json", "long_context_retrieval",
                 "long_generation", "agent_tool_use", "general_conversation"}
EVAL_DOMAINS = ["coding", "math_reasoning", "agent_tool_use"]
ALPHAS = (0.25, 0.5, 0.75)
RANKS = (16, 32, 64)
STEP = BLOCK_SIZE        # block-spaced state spacing


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t00 = time.time()
    os.makedirs(OUT, exist_ok=True)

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download("deepseek-ai/DeepSeek-V4-Flash-0731")
    revision = os.path.basename(model_dir)
    tok = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=cfg, torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: f"{MAIN_GPU_GIB}GiB", "cpu": "100GiB"},
        offload_folder="/tmp/offload", trust_remote_code=True)
    model.eval()

    # hooks on layers 40/41/42 -> captured mean-over-hc states
    captured = []          # list of [7, 1, 3*dim] per step

    def hook(mod, args, output):
        captured.append(output.detach().mean(dim=2))

    handles = [model.model.layers[li].register_forward_hook(hook)
               for li in TARGET_LAYERS]

    from safetensors import safe_open
    idx = json.load(open(os.path.join(
        model_dir, "model.safetensors.index.json")))

    def ckpt(key):
        with safe_open(os.path.join(model_dir, idx["weight_map"][key]),
                       framework="pt", device="cpu") as f:
            return f.get_tensor(key).to("cuda")

    head_w = ckpt("head.weight")
    embed_weight = model.model.embed_tokens.weight.data
    if embed_weight.device.type != "cuda":
        embed_weight = ckpt("embed.weight")
    dev = head_w.device

    fp4 = torch.tensor(FP4_TABLE_VALUES, dtype=torch.float32, device=dev)
    raw = load_mtp_tensors(model_dir, dev)
    fc = precompute_freqs_cis(64, 8192, 65536, 10000, 16, 32, 1).to(dev)
    stages = [DSparkStage(s, raw[s], fc, dev, fp4) for s in range(N_STAGES)]
    log(f"loaded ({time.time()-t00:.0f}s)")

    # ================= B1: batched capture =================
    cache_path = f"{OUT}/nanospark_states_210.pt"
    dataset = None
    if os.path.exists(cache_path):
        try:
            dataset = torch.load(cache_path)
            if all(c in dataset for c, _ in PROMPTS) and \
                    dataset[PROMPTS[0][0]]["mx"].shape[0] == N_CAP:
                log("B1: reusing cached 210-position dataset")
            else:
                dataset = None
        except Exception:
            dataset = None

    if dataset is None:
        log(f"B1: batched capture {N_CAP} positions x {len(PROMPTS)} domains")
        texts = [p for _, p in PROMPTS]
        cats = [c for c, _ in PROMPTS]
        enc = tok(texts, return_tensors="pt", padding=True)
        input_ids = enc.input_ids.to(dev)         # [7, L]
        attn = enc.attention_mask.to(dev)
        n_batch, L = input_ids.shape
        prompt_lens = attn.sum(dim=1).tolist()
        log(f"  batch {n_batch}x{L} (prompt lens {prompt_lens})")

        # prefill (hooks capture [7, L, 3d]; last position per row is real)
        captured.clear()
        with torch.no_grad():
            out = model(input_ids, attention_mask=attn, use_cache=True)
        past = out.past_key_values
        # states_by_offset[0] = state at last prompt position per row
        mh_pref = torch.cat(captured, dim=-1)     # [7, L, 3d]
        s_prefill = mh_pref[:, -1, :]             # [7, 3d] at offset 0
        captured.clear()
        g_next = out.logits[:, -1].argmax(-1)     # [7]

        # decode loop: states_by_offset[t+1] = state at offset t+1
        states_list = [s_prefill.clone()]         # offset 0
        toks_list = [input_ids[:, -1].clone()]    # token at offset 0
        cur = g_next
        for t in range(N_CAP - 1):
            with torch.no_grad():
                o = model(cur.unsqueeze(1), attention_mask=None,
                          past_key_values=past, use_cache=True)
                past = o.past_key_values
            mh = torch.cat(captured, dim=-1)      # [7, 1, 3d]
            captured.clear()
            states_list.append(mh[:, 0, :].clone())   # offset t+1
            toks_list.append(cur.clone())
            cur = o.logits[:, -1].argmax(-1)
        for h in handles:
            h.remove()

        # per-domain: states [N,3d], tokens [N], mx [N,4096]
        dataset = {}
        for i, c in enumerate(cats):
            mh_d = torch.stack([s[i] for s in states_list])  # [N,3d]
            toks_d = torch.stack([t[i] for t in toks_list])  # [N]
            with torch.no_grad():
                mx_d = stages[0].main_norm(stages[0].main_proj(
                    mh_d.to(torch.bfloat16).to(dev)))
            dataset[c] = {
                "mh": mh_d.half().cpu(),
                "mx": mx_d.half().cpu(),
                "tokens": toks_d.cpu().tolist(),
                "prompt_len": prompt_lens[i],
                "padded_len": L,
            }
            log(f"  [{c}] captured {N_CAP} states")
        torch.save(dataset, cache_path)
        log(f"B1 dataset saved ({time.time()-t00:.0f}s)")

    # manifest (B1 requirement §46)
    manifest = {
        "revision": revision,
        "n_domains": len(PROMPTS), "n_positions": N_CAP,
        "block_spacing": STEP, "state_dims": {"mh": 12288, "mx": 4096},
        "train_domains": sorted(TRAIN_DOMAINS),
        "held_out_domains": ["coding", "math_reasoning"],
        "capture": "batched greedy decode, hooks on layers 40/41/42, "
                   "left-padded prompts, mx via main_proj+main_norm",
        "prompt_hashes": {c: hashlib.sha256(p.encode()).hexdigest()[:16]
                          for c, p in PROMPTS},
    }

    for h in handles:
        h.remove()          # cached path: never removed in capture branch

    # ================= B2 offline: fit + diagnostics =================
    log("B2/B4: fitting predictors on mx (4096-dim)")
    train_d = [c for c, _ in PROMPTS if c in TRAIN_DOMAINS]

    # block-spaced training pairs: (s_o, tok_o) -> s_{o+STEP}
    def build_pairs(domains):
        S, TOK, T = [], [], []
        for d in domains:
            v = dataset[d]
            mx = v["mx"].float().cuda()
            for i in range(0, N_CAP - STEP, STEP):
                S.append(mx[i])
                TOK.append(v["tokens"][i])
                T.append(mx[i + STEP])
        return torch.stack(S), TOK, torch.stack(T)

    S_tr, TOK_tr, T_tr = build_pairs(train_d)
    D_tr = T_tr - S_tr
    n_pairs = S_tr.shape[0]
    log(f"  train pairs: {n_pairs} (5 domains x {N_CAP//STEP - 1} steps)")

    # features: [s; e_tok]
    embed_f = embed_weight.to(torch.float32)
    X_tr = torch.stack([
        torch.cat([s, embed_f[t]]) for s, t in zip(S_tr, TOK_tr)])

    # ---- low-rank residual: PCA V_r + ridge U (B4) ----
    # V_r from PCA of X_tr; U: ridge from r-dim latent to 4096-dim delta
    Xc = X_tr - X_tr.mean(dim=0, keepdim=True)
    U_svd, sig, Vt = torch.linalg.svd(Xc, full_matrices=False)
    Dm = X_tr.shape[1]
    latent_means = X_tr.mean(dim=0)

    lowrank = {}
    for r in RANKS:
        V_r = Vt[:r].T                                    # [8192, r]
        Z = (X_tr - latent_means) @ V_r                   # [n, r]
        Zc = Z - Z.mean(dim=0, keepdim=True)
        # standardize
        Z_std = Z.std(dim=0).clamp_min(1e-6)
        Zn = (Z - Z.mean(dim=0, keepdim=True)) / Z_std
        D_mean = D_tr.mean(dim=0)
        lam = 1.0
        A = Zn.T @ Zn + lam * torch.eye(r, device=dev)
        U_coef = torch.linalg.solve(A, Zn.T @ (D_tr - D_mean))  # [r,4096]
        lowrank[r] = {
            "V": V_r, "U": U_coef,
            "Z_raw_mean": Z.mean(dim=0),        # subtract BEFORE scaling
            "D_mean": D_mean, "x_mean": latent_means, "Z_std": Z_std,
        }
        log(f"  rank-{r} fitted ({r}x{Dm} + {r}x4096 params)")

    # ---- predictor functions ----
    def mk_hold():
        return lambda s_hist, tok: s_hist[-1]

    def mk_damped(alpha):
        def fn(s_hist, tok):
            return s_hist[-1] + alpha * (s_hist[-1] - s_hist[-2])
        return fn

    def mk_velocity():
        return lambda s_hist, tok: \
            s_hist[-1] + (s_hist[-1] - s_hist[-2])

    def mk_lowrank(r):
        p = lowrank[r]
        def fn(s_hist, tok):
            x = torch.cat([s_hist[-1], embed_f[tok]])
            z = (x - p["x_mean"]) @ p["V"]
            z = (z - p["Z_raw_mean"]) / p["Z_std"]
            d = z @ p["U"] + p["D_mean"]
            return s_hist[-1] + d
        return fn

    predictors = {"P0_hold": (mk_hold(), False)}
    for a in ALPHAS:
        predictors[f"P1_damped_{a}"] = (mk_damped(a), False)
    for r in RANKS:
        predictors[f"P3_lowrank_{r}"] = (mk_lowrank(r), False)

    # ---- B2 diagnostics: one-step error on all domains ----
    log("B2: transition diagnostics (one-step, all domains)")
    diag = {}
    for d, _ in PROMPTS:
        v = dataset[d]
        mx_all = v["mx"].float().cuda()
        S_d, TOK_d, T_d = build_pairs([d])
        diag[d] = {}
        for name, (fn, _) in predictors.items():
            cosines, norms, rel_errs = [], [], []
            for i in range(S_d.shape[0]):
                # 2-point history for delta-based predictors
                hist = ([mx_all[i - STEP], S_d[i]]
                        if i >= STEP else [S_d[i], S_d[i]])
                pred = fn(hist, TOK_d[i])
                t = T_d[i]
                c = torch.nn.functional.cosine_similarity(
                    pred.unsqueeze(0), t.unsqueeze(0)).item()
                cosines.append(c)
                norms.append((pred.norm() / t.norm()).item())
                rel_errs.append(((pred - t).norm() / t.norm()).item())
            diag[d][name] = {
                "cosine_mean": round(sum(cosines) / len(cosines), 4),
                "norm_ratio_mean": round(sum(norms) / len(norms), 4),
                "rel_err_mean": round(sum(rel_errs) / len(rel_errs), 4),
                "n_pairs": len(cosines),
            }
    log("  diagnostics done")

    # ---- B2 latency ----
    latencies = {}
    s_test = torch.randn(4096, device=dev, dtype=torch.float32)
    for name, (fn, _) in predictors.items():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            fn([s_test, s_test], 100)
        torch.cuda.synchronize()
        latencies[name] = round((time.perf_counter() - t0) / 100 * 1e3, 3)
    log(f"  latencies (ms): {latencies}")

    # ================= B3/B4/B5: recursive evaluation =================
    log(f"B3/B4: recursive eval ({len(EVAL_DOMAINS)} domains x "
        f"{N_CHAINS} chains x R{R_DEPTH})")

    def draft(input_ids, main_hidden, start_pos):
        h, main_x = stages[0].forward_embed(main_hidden, input_ids,
                                            embed_weight)
        for st in stages:
            h = st(h, start_pos, main_x)
        if start_pos == 0:
            return None
        return stages[-1].forward_head(h, input_ids, head_w)

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

    def eval_predictor(name, step_fn):
        results = {}
        for cat in EVAL_DOMAINS:
            v = dataset[cat]
            prompt_text = dict(PROMPTS)[cat]
            ids = tok(prompt_text, return_tensors="pt").input_ids.to(dev)
            n_prompt = ids.shape[1]
            captured.clear()
            handles2 = [model.model.layers[li].register_forward_hook(
                lambda m, a, o: captured.append(o.detach().mean(dim=2)))
                for li in TARGET_LAYERS]
            with torch.no_grad():
                logits0 = model(ids, use_cache=True)
            mh_full = torch.cat(captured, dim=-1)
            captured.clear()
            for h in handles2:
                h.remove()
            past = logits0.past_key_values
            with torch.no_grad():
                draft(ids[:, -1:], mh_full, 0)
            g_next = logits0.logits[0, -1].argmax().item()
            last_tok = ids[0, -1].item()
            last_mh = mh_full[0, -1:]
            mx_traj = v["mx"].float().cuda()

            acc_list = []
            depth_acc = [[0] * BLOCK_SIZE for _ in range(R_DEPTH)]
            depth_reach = [0] * R_DEPTH
            for blk in range(N_CHAINS):
                L0 = past.get_seq_length()
                s_exact = last_mh.clone()
                idx = min(max(L0 - n_prompt, 0), N_CAP - 1)
                s_hist = [mx_traj[max(0, i)] for i in
                          (idx - 10, idx - 5, idx)]
                chain = []
                cur_tok, sp = last_tok, L0 - 1
                with torch.no_grad():
                    for r in range(R_DEPTH):
                        if r == 0:
                            out = draft(
                                torch.tensor([[cur_tok]], device=dev),
                                s_exact.unsqueeze(0), sp)
                        else:
                            s_pred = step_fn(s_hist, cur_tok)
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
                per_r.append(round(sum(depth_acc[r]) / n, 3)
                             if n else None)
            results[cat] = {"mean_accepted": round(sum(acc_list)
                                                   / len(acc_list), 3),
                            "per_block_per_token": per_r,
                            "chains": acc_list}
            log(f"  [{name}/{cat}] mean={results[cat]['mean_accepted']} "
                f"blocks={per_r}")
        ho = [results[c] for c in results if c not in TRAIN_DOMAINS]
        all_c = [x["chains"] for x in results.values()]
        flat = [c for row in all_c for c in row]
        ho_flat = [c for x in ho for c in x["chains"]]
        return results, {
            "mean": round(sum(flat) / len(flat), 3),
            "held_out": round(sum(ho_flat) / len(ho_flat), 3)
            if ho_flat else None,
            "block3_held_out": None,
        }

    def target_forward(tokens, past=None):
        captured.clear()
        handles2 = [model.model.layers[li].register_forward_hook(
            lambda m, a, o: captured.append(o.detach().mean(dim=2)))
            for li in TARGET_LAYERS]
        try:
            with torch.no_grad():
                o = model(tokens, past_key_values=past, use_cache=True)
            mh = torch.cat(captured, dim=-1)
        finally:
            for h in handles2:
                h.remove()
        captured.clear()
        return o.logits, o.past_key_values, mh

    table = {}
    for name, (fn, _) in predictors.items():
        res, agg = eval_predictor(name, fn)
        # block-3 gate: any held-out domain with block-3 > 0
        b3 = [res[c]["per_block_per_token"][2] for c in res
              if c not in TRAIN_DOMAINS
              and res[c]["per_block_per_token"][2] is not None]
        agg["block3_held_out"] = b3 if b3 else None
        table[name] = {"per_domain": res, "aggregate": agg}
        log(f"  {name}: mean={agg['mean']} held_out={agg['held_out']} "
            f"block3={agg['block3_held_out']}")
        _save(manifest, table, diag, latencies, n_pairs, revision)

    # ================= report =================
    log("\n=== B1-B4 RACE RESULTS ===")
    log(f"{'predictor':<22} {'mean':>7} {'held-out':>9} {'block3':>8}")
    for name, v in table.items():
        a = v["aggregate"]
        log(f"{name:<22} {a['mean']:>7} {a['held_out']:>9} "
            f"{a['block3_held_out']}")
    _save(manifest, table, diag, latencies, n_pairs, revision)
    log(f"=== DONE in {(time.time()-t00)/60:.1f} min ===")
    return 0


def _save(manifest, table, diag, latencies, n_pairs, revision):
    out = {
        "phase": "NanoSpark-B1-B4",
        "model_revision": revision,
        "classification": "BLACKWELL_SANDBOX_ONLY + "
                          "DRAFTER_NUMERICS_DEVIATION_RECORDED",
        "manifest": manifest,
        "n_train_pairs": n_pairs,
        "transition_diagnostics": diag,
        "predictor_latency_ms": latencies,
        "predictors": table,
        "b5_block3_gate": {
            "revived": any(
                v["aggregate"]["block3_held_out"]
                for v in table.values()),
            "detail": {k: v["aggregate"]["block3_held_out"]
                       for k, v in table.items()},
        },
    }
    with open(f"{OUT}/nanospark_b1b4.json", "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
