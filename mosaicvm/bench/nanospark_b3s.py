"""NanoSpark B3-S: sequential Δ rollout (roadmap §63-69).

For each Δ ∈ {1, 2, 3, 5}: run actual sequential recursion where the
drafter state is predicted Δ tokens ahead at each step. DSpark drafts
5-token blocks; at Δ<5 we take only the first Δ proposals per step
(microstep mode).

Chain structure per Δ:
  1. exact target state refresh → draft block (first Δ proposals used)
  2. verify against exact greedy target (first-mismatch stop)
  3. predict state Δ ahead from current state + last accepted token
  4. draft next block from predicted state
  5. repeat until rejection or depth cap

Measure: E[A], mean/p50/p90 accepted depth, full-reject rate,
depth survival curve, per-domain breakdown.

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
    BLOCK_SIZE, N_STAGES, PROMPTS, TARGET_LAYERS,
    DSparkStage, FP4_TABLE_VALUES, load_mtp_tensors,
    precompute_freqs_cis)

OUT = "/marimo/MosaicVM/artifacts"
SPACINGS = (1, 2, 3, 5)
B3_GPU_GIB = 66              # leave ~20 GiB for expert dequant cycling
N_CHAINS = 6                 # chains per domain per Δ
MAX_DEPTH = 30               # max speculative tokens per chain
TRAIN_DOMAINS = {"structured_json", "long_context_retrieval",
                 "long_generation", "agent_tool_use", "general_conversation"}
EVAL_DOMAINS = ["coding", "math_reasoning", "agent_tool_use"]


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
        device_map="auto",
        max_memory={0: f"{B3_GPU_GIB}GiB", "cpu": "100GiB"},
        offload_folder="/tmp/offload", trust_remote_code=True)
    model.eval()

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
    embed_f = embed_weight.to(torch.float32)

    fp4 = torch.tensor(FP4_TABLE_VALUES, dtype=torch.float32, device=dev)
    raw = load_mtp_tensors(model_dir, dev)
    fc = precompute_freqs_cis(64, 8192, 65536, 10000, 16, 32, 1).to(dev)
    stages = [DSparkStage(s, raw[s], fc, dev, fp4) for s in range(N_STAGES)]
    log(f"loaded ({time.time()-t00:.0f}s)")

    dataset = torch.load(f"{OUT}/nanospark_states_210.pt")
    N_CAP = dataset[PROMPTS[0][0]]["mx"].shape[0]
    log(f"dataset: {len(dataset)} domains × {N_CAP} positions")

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

    # train rank-64 at each spacing (same as B3-R)
    log("training rank-64 at each spacing...")
    predictors = {}
    for delta in SPACINGS:
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
        lam = 1.0
        A = Zn.T @ Zn + lam * torch.eye(r, device=dev)
        U_coef = torch.linalg.solve(A, Zn.T @ (D_tr - D_tr.mean(0)))
        predictors[delta] = {
            "V": V_r, "U": U_coef,
            "Z_raw_mean": Z.mean(0), "D_mean": D_tr.mean(0),
            "x_mean": X_tr.mean(0), "Z_std": Z_std,
        }
    log("predictors trained")

    def rank64_pred(s, tok, delta):
        p = predictors[delta]
        x = torch.cat([s, embed_f[tok]])
        z = (x - p["x_mean"]) @ p["V"]
        z = (z - p["Z_raw_mean"]) / p["Z_std"]
        return s + z @ p["U"] + p["D_mean"]

    # hook infrastructure
    captured = []

    def hook(mod, args, output):
        captured.append(output.detach().mean(dim=2))

    def attach():
        return [model.model.layers[li].register_forward_hook(hook)
                for li in TARGET_LAYERS]

    def detach(handles):
        for h in handles:
            h.remove()
        captured.clear()

    def target_forward(tokens, past=None):
        captured.clear()
        hs = attach()
        try:
            with torch.no_grad():
                o = model(tokens, past_key_values=past, use_cache=True)
            mh = torch.cat(captured, dim=-1)
        finally:
            detach(hs)
        return o.logits, o.past_key_values, mh

    # ============ B3-S sequential rollout ============
    log(f"B3-S: sequential rollout ({N_CHAINS} chains × "
        f"{len(EVAL_DOMAINS)} domains × {len(SPACINGS)} spacings)")

    results = {"per_spacing": {}}

    for delta in SPACINGS:
        d_key = str(delta)
        results["per_spacing"][d_key] = {"per_domain": {}}
        all_depths = []          # accepted tokens per chain across domains

        for cat in EVAL_DOMAINS:
            prompt_text = dict(PROMPTS)[cat]
            v = dataset[cat]
            mx_traj = v["mx"].float().cuda()
            tokens_traj = v["tokens"]

            ids = tok(prompt_text, return_tensors="pt").input_ids.to(dev)
            n_prompt = ids.shape[1]

            # prefill
            logits, past, mh_full = target_forward(ids)
            # project raw main_hidden -> main_x for drafter prefill
            with torch.no_grad():
                mx_full = stages[0].main_norm(
                    stages[0].main_proj(mh_full.to(torch.bfloat16)))
                draft_with_mx(ids[:, -1:], mx_full, 0)
            g_next = logits[0, -1].argmax().item()
            last_tok = ids[0, -1].item()
            last_mx = mx_full[0, -1]     # 4096-d projected state

            chain_depths = []

            for chain_i in range(N_CHAINS):
                L0 = past.get_seq_length()
                orig_g = g_next            # greedy at chain start
                ring_snap = [st.attn.kv_cache.clone() for st in stages]
                s_pred = last_mx           # 4096-d projected state
                cur_tok = last_tok
                sp = L0 - 1
                total_accepted = 0

                for step in range(MAX_DEPTH // delta):
                    src_tok = cur_tok   # token at state's position
                    # draft Δ proposals from current predicted state
                    with torch.no_grad():
                        out = draft_with_mx(
                            torch.tensor([[cur_tok]], device=dev),
                            s_pred[None, None, :], sp)
                    block = out["greedy_ids"][0, 1:1 + delta].tolist()

                    # verify: compare against known greedy BEFORE feeding
                    g = g_next
                    step_accepted = 0
                    for t in block:
                        if t == g:
                            step_accepted += 1
                            total_accepted += 1
                            cur_tok = t
                            sp += 1
                            # feed accepted token to get next greedy
                            lg, past, _ = target_forward(
                                torch.tensor([[t]], device=dev), past)
                            g_next = lg[0, -1].argmax().item()
                            g = g_next
                        else:
                            break

                    if step_accepted < delta:
                        break   # chain ended at this step

                    # predict state Δ ahead: feature token = the token
                    # at the INPUT state's position (training semantics)
                    s_pred = rank64_pred(s_pred, src_tok, delta)

                chain_depths.append(total_accepted)

                # re-anchor: rollback + advance one true greedy position
                past.crop(L0)
                for st, snap in zip(stages, ring_snap):
                    st.attn.kv_cache.copy_(snap)
                lg, past, mh_b = target_forward(
                    torch.tensor([[orig_g]], device=dev), past)
                last_tok = orig_g
                # project for next chain's state seed
                with torch.no_grad():
                    mx_b = stages[0].main_norm(
                        stages[0].main_proj(mh_b.to(torch.bfloat16)))
                last_mx = mx_b[0, -1]
                g_next = lg[0, -1].argmax().item()

            all_depths.extend(chain_depths)
            results["per_spacing"][d_key]["per_domain"][cat] = {
                "n_chains": len(chain_depths),
                "depths": chain_depths,
                "mean": round(sum(chain_depths) / len(chain_depths), 2)
                if chain_depths else None,
            }
            log(f"  Δ={delta} [{cat}] mean={sum(chain_depths)/len(chain_depths):.1f} "
                f"depths={chain_depths}")

        # aggregate per spacing
        if all_depths:
            ds = sorted(all_depths)
            n = len(ds)
            results["per_spacing"][d_key]["aggregate"] = {
                "n_chains": n,
                "E[A]": round(sum(ds) / n, 3),
                "mean_depth": round(sum(ds) / n, 3),
                "p50_depth": ds[n // 2],
                "p90_depth": ds[int(n * 0.9)],
                "max_depth": ds[-1],
                "full_reject_rate": round(
                    sum(1 for d in ds if d == 0) / n, 3),
                "depth_survival": {
                    str(d): sum(1 for x in ds if x >= d)
                    for d in [1, 2, 3, 5, 10, 15, 20, 30]},
            }
            agg = results["per_spacing"][d_key]["aggregate"]
            log(f"  Δ={delta} AGG: E[A]={agg['E[A]']} "
                f"p50={agg['p50_depth']} p90={agg['p90_depth']} "
                f"reject={agg['full_reject_rate']}")

    # report
    log("\n=== B3-S SEQUENTIAL ROLLOUT RESULTS ===")
    log(f"{'Δ':>3} {'n':>4} {'E[A]':>6} {'p50':>5} {'p90':>5} "
        f"{'reject':>7} {'max':>4}")
    log("-" * 42)
    for delta in SPACINGS:
        a = results["per_spacing"][str(delta)].get("aggregate", {})
        if a:
            log(f"{delta:>3} {a['n_chains']:>4} {a['E[A]']:>6} "
                f"{a['p50_depth']:>5} {a['p90_depth']:>5} "
                f"{a['full_reject_rate']:>7} {a['max_depth']:>4}")

    out = {
        "phase": "NanoSpark-B3S-sequential-rollout",
        "model_revision": revision,
        "classification": "BLACKWELL_SANDBOX_ONLY + "
                          "DRAFTER_NUMERICS_DEVIATION_RECORDED",
        "design": {
            "spacings": list(SPACINGS),
            "mode": "sequential recursion (Mode R)",
            "n_chains_per_domain": N_CHAINS,
            "max_depth": MAX_DEPTH,
            "eval_domains": EVAL_DOMAINS,
            "predictor": "rank-64 PCA+ridge at each Δ",
        },
        "results": results,
    }
    with open(f"{OUT}/nanospark_b3s_sequential.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    log(f"wrote {OUT}/nanospark_b3s_sequential.json")
    log(f"=== DONE in {(time.time()-t00)/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
