"""NanoSpark B1+B2: functional sensitivity + margin analysis (§36-37).

B1 — Logit-Jacobian subspace: for each position, compute the gradient
of the DSpark draft top-1 logit w.r.t. main_x. Stack these gradients,
SVD → active subspace. Project hold/rank-64 prediction error into vs
out of this subspace. If most improvement lands outside, the B4
behavioral result is explained.

B2 — Margin/interpolation: at each position record the teacher draft
top1-top2 margin, then compare with the predicted-state draft. Inter-
polate s(λ) = ŝ + λ(s* - ŝ) and find the smallest λ that changes the
top-1 draft token. This directly measures how much functionally
relevant correction is required.

Uses the cached 210-position dataset (no recapture).

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
N_PROBE = 6                # positions per domain for gradient probes
STEP = BLOCK_SIZE
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
        max_memory={0: f"{MAIN_GPU_GIB}GiB", "cpu": "100GiB"},
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

    fp4 = torch.tensor(FP4_TABLE_VALUES, dtype=torch.float32, device=dev)
    raw = load_mtp_tensors(model_dir, dev)
    fc = precompute_freqs_cis(64, 8192, 65536, 10000, 16, 32, 1).to(dev)
    stages = [DSparkStage(s, raw[s], fc, dev, fp4) for s in range(N_STAGES)]
    log(f"loaded ({time.time()-t00:.0f}s)")

    # load cached dataset
    cache_path = f"{OUT}/nanospark_states_210.pt"
    dataset = torch.load(cache_path)
    log(f"dataset loaded: {len(dataset)} domains, "
        f"{dataset[PROMPTS[0][0]]['mx'].shape[0]} positions")

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

    # Direct margin measurement — NO gradient estimation. The 16-dir
    # reconstruction lived in a random span and EPS was below bf16 ULP.
    FD_FRAC = 0.05                  # fraction of ||s|| for directional FD

    def margin_at(s):
        """Margin (top1 - top2) of the greedy draft with state s."""
        with torch.no_grad():
            out = draft_with_mx(tok_tensor, s[None, None, :], sp)
            lg = out["base_logits"][0, 0].float()
            top2_idx = lg.topk(2).indices[1].item()
            return (lg.max() - lg[top2_idx]).item(), lg.argmax().item()

    # fit rank-64 for comparison
    log("fitting rank-64 for B1 comparison...")
    train_domains = {"structured_json", "long_context_retrieval",
                     "long_generation", "agent_tool_use",
                     "general_conversation"}
    N_CAP = dataset[PROMPTS[0][0]]["mx"].shape[0]
    embed_f = embed_weight.to(torch.float32)

    S_tr, TOK_tr, T_tr = [], [], []
    for d in train_domains:
        v = dataset[d]
        mx = v["mx"].float().cuda()
        for i in range(0, N_CAP - STEP, STEP):
            S_tr.append(mx[i])
            TOK_tr.append(v["tokens"][i])
            T_tr.append(mx[i + STEP])
    S_tr = torch.stack(S_tr)
    T_tr = torch.stack(T_tr)
    D_tr = T_tr - S_tr
    X_tr = torch.stack([
        torch.cat([s, embed_f[t]]) for s, t in zip(S_tr, TOK_tr)])
    Xc = X_tr - X_tr.mean(dim=0, keepdim=True)
    _, sig, Vt = torch.linalg.svd(Xc, full_matrices=False)
    r = 64
    V_r = Vt[:r].T
    Z = (X_tr - X_tr.mean(0, keepdim=True)) @ V_r
    Z_std = Z.std(dim=0).clamp_min(1e-6)
    Zn = (Z - Z.mean(0, keepdim=True)) / Z_std
    lam = 1.0
    A = Zn.T @ Zn + lam * torch.eye(r, device=dev)
    U_coef = torch.linalg.solve(A, Zn.T @ (D_tr - D_tr.mean(0, keepdim=True)))
    lr64 = {
        "V": V_r, "U": U_coef,
        "Z_raw_mean": Z.mean(dim=0), "D_mean": D_tr.mean(0),
        "x_mean": X_tr.mean(0), "Z_std": Z_std,
    }

    def rank64_pred(s_hist, tok):
        p = lr64
        x = torch.cat([s_hist[-1], embed_f[tok]])
        z = (x - p["x_mean"]) @ p["V"]
        z = (z - p["Z_raw_mean"]) / p["Z_std"]
        return s_hist[-1] + z @ p["U"] + p["D_mean"]

    log("rank-64 fitted")

    # ============ B1: gradient probes ============
    log(f"B1: gradient probes ({N_PROBE} positions x "
        f"{len(EVAL_DOMAINS)} domains)")

    results = {"per_domain": {}, "aggregate": {}}
    all_results = []

    for cat in EVAL_DOMAINS:
        v = dataset[cat]
        mx_all = v["mx"].float().cuda()
        tokens = v["tokens"]
        domain_data = {"probes": []}

        for probe_i in range(N_PROBE):
            pos = STEP + probe_i * (N_CAP // N_PROBE)
            if pos >= N_CAP or pos - STEP < 0:
                continue
            # zero all ring buffers before EVERY probe (isolated:
            # no cross-position KV contamination)
            for st in stages:
                st.attn.kv_cache.zero_()

            s_true = mx_all[pos]
            s_hold = mx_all[pos - STEP]
            tok = tokens[pos]
            tok_tensor = torch.tensor([[tok]], device=dev)
            sp = pos  # position for draft

            # teacher logits (true state)
            with torch.no_grad():
                out_t = draft_with_mx(tok_tensor, s_true[None, None, :], sp)
            t_logits = out_t["base_logits"][0, 0].float()

            # hold logits
            with torch.no_grad():
                out_h = draft_with_mx(tok_tensor, s_hold[None, None, :], sp)
            h_logits = out_h["base_logits"][0, 0].float()

            # rank64 logits
            s_r64 = rank64_pred([s_true, s_hold], tokens[pos - STEP])
            with torch.no_grad():
                out_r = draft_with_mx(tok_tensor, s_r64[None, None, :], sp)
            r_logits = out_r["base_logits"][0, 0].float()

            # metrics
            t_top1 = t_logits.argmax().item()
            h_top1 = h_logits.argmax().item()
            r_top1 = r_logits.argmax().item()
            t_top2 = t_logits.topk(2).indices[1].item()
            margin = (t_logits[t_top1] - t_logits[t_top2]).item()

            # KL divergence
            t_logp = torch.log_softmax(t_logits, dim=-1)
            h_logp = torch.log_softmax(h_logits, dim=-1)
            kl_h = torch.nn.functional.kl_div(
                h_logp, t_logp, reduction="sum", log_target=True).item()
            r_logp = torch.log_softmax(r_logits, dim=-1)
            kl_r = torch.nn.functional.kl_div(
                r_logp, t_logp, reduction="sum", log_target=True).item()

            # state error
            err_hold = (s_hold - s_true)
            err_r64 = (s_r64 - s_true)
            cos_hold = torch.nn.functional.cosine_similarity(
                err_hold.unsqueeze(0), err_hold.unsqueeze(0)).item()
            rel_hold = (err_hold.norm() / s_true.norm()).item()
            rel_r64 = (err_r64.norm() / s_true.norm()).item()

            # Direct margins at each state
            m_true, _ = margin_at(s_true)
            m_hold, _ = margin_at(s_hold)
            m_r64, _ = margin_at(s_r64)

            # Directional FD along the ACTUAL error vectors, with
            # FROZEN teacher top1/top2: move from the predicted state
            # toward the true state by FD_FRAC*||s||, measure the
            # teacher-margin (z_top1 - z_top2, frozen indices) change.
            def frozen_margin(s, t1, t2):
                with torch.no_grad():
                    out = draft_with_mx(tok_tensor, s[None, None, :], sp)
                    lg = out["base_logits"][0, 0].float()
                    return (lg[t1] - lg[t2]).item()

            fm_true = frozen_margin(s_true, t_top1, t_top2)
            fm_hold = frozen_margin(s_hold, t_top1, t_top2)
            fm_r64 = frozen_margin(s_r64, t_top1, t_top2)
            # move along unit error direction by eps = FD_FRAC * ||s||
            eps = FD_FRAC * s_true.norm().item()
            # unit vectors pointing FROM predicted TOWARD true
            u_hold = (s_true - s_hold) / ((s_true - s_hold).norm() + 1e-9)
            u_r64 = (s_true - s_r64) / ((s_true - s_r64).norm() + 1e-9)
            dm_hold = frozen_margin(s_hold + eps * u_hold, t_top1,
                                    t_top2) - fm_hold
            dm_r64 = frozen_margin(s_r64 + eps * u_r64, t_top1,
                                   t_top2) - fm_r64
            err_hold_along = dm_hold     # signed: + = margin improves
            err_r64_along = dm_r64
            mg_along = fm_true

            # B2: interpolation probe — smallest λ that changes top-1
            lam_flip = None
            for lam in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                s_interp = s_hold + lam * (s_true - s_hold)
                with torch.no_grad():
                    out_i = draft_with_mx(
                        tok_tensor, s_interp[None, None, :], sp)
                    i_top1 = out_i["base_logits"][0, 0].float() \
                        .argmax().item()
                if i_top1 == t_top1:
                    lam_flip = lam
                    break

            probe = {
                "pos": pos, "token": tok,
                "teacher_top1": t_top1,
                "hold_top1_match": h_top1 == t_top1,
                "r64_top1_match": r_top1 == t_top1,
                "margin": round(margin, 3),
                "kl_hold": round(kl_h, 4),
                "kl_r64": round(kl_r, 4),
                "rel_err_hold": round(rel_hold, 4),
                "rel_err_r64": round(rel_r64, 4),
                "err_hold_along_margin": round(err_hold_along, 6),
                "err_r64_along_margin": round(err_r64_along, 6),
                "frozen_margin_true": round(mg_along, 4),
                "fm_hold": round(fm_hold, 4),
                "fm_r64": round(fm_r64, 4),
                "lambda_flip": lam_flip,
            }
            domain_data["probes"].append(probe)
            all_results.append(probe)

        # domain summary
        probes = domain_data["probes"]
        if probes:
            domain_data["summary"] = {
                "n_probes": len(probes),
                "hold_top1_rate": round(
                    sum(p["hold_top1_match"] for p in probes)
                    / len(probes), 3),
                "r64_top1_rate": round(
                    sum(p["r64_top1_match"] for p in probes)
                    / len(probes), 3),
                "mean_margin": round(
                    sum(p["margin"] for p in probes) / len(probes), 3),
                "mean_kl_hold": round(
                    sum(p["kl_hold"] for p in probes) / len(probes), 3),
                "mean_kl_r64": round(
                    sum(p["kl_r64"] for p in probes) / len(probes), 3),
                "mean_lambda_flip": round(
                    sum(p["lambda_flip"] or 1.0 for p in probes)
                    / len(probes), 3),
            }
        results["per_domain"][cat] = domain_data
        s = domain_data.get("summary", {})
        log(f"  [{cat}] hold_top1={s.get('hold_top1_rate')} "
            f"r64_top1={s.get('r64_top1_rate')} "
            f"margin={s.get('mean_margin')} "
            f"λ_flip={s.get('mean_lambda_flip')}")

    # ============ B1: margin landscape summary ============
    # (No gradient/subspace estimation — direct margin comparison +
    #  directional FD along actual error vectors)
    log("B1: margin landscape summary")

    # aggregate
    if all_results:
        # margin landscape summary
        mt = [p["frozen_margin_true"] for p in all_results
              if "frozen_margin_true" in p]
        mh = [p["margin"] for p in all_results]
        dmh = [p.get("err_hold_along_margin", 0) for p in all_results]
        dmr = [p.get("err_r64_along_margin", 0) for p in all_results]
        def _m(l):
            return round(sum(l) / len(l), 4) if l else None
        subspace = {
            "analysis_type": "DIRECT_MARGIN_COMPARISON",
            "note": ("No gradient subspace — direct margin at "
                     "true/hold/r64 states + directional FD along "
                     "error vectors (signed)"),
            "frozen_margin_true_mean": _m(mt),
            "dm_hold_mean": _m(dmh),
            "dm_r64_mean": _m(dmr),
        }
        log(f"  frozen_margin(true)={subspace['frozen_margin_true_mean']}")
        log(f"  dm_hold (signed)={subspace['dm_hold_mean']} "
            f"dm_r64 (signed)={subspace['dm_r64_mean']}")
        results["aggregate"] = {
            "n_probes": len(all_results),
            "hold_top1_rate": round(
                sum(p["hold_top1_match"] for p in all_results)
                / len(all_results), 3),
            "r64_top1_rate": round(
                sum(p["r64_top1_match"] for p in all_results)
                / len(all_results), 3),
            "mean_margin": round(
                sum(p["margin"] for p in all_results)
                / len(all_results), 3),
            "mean_kl_hold": round(
                sum(p["kl_hold"] for p in all_results)
                / len(all_results), 3),
            "mean_kl_r64": round(
                sum(p["kl_r64"] for p in all_results)
                / len(all_results), 3),
            "mean_lambda_flip": round(
                sum(p["lambda_flip"] or 1.0 for p in all_results)
                / len(all_results), 3),
            "lambda_flip_distribution": {
                str(l): sum(1 for p in all_results
                            if p["lambda_flip"] == l)
                for l in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.9, 1.0, None]},
        }

    out = {
        "phase": "NanoSpark-B1-B2-sensitivity",
        "model_revision": revision,
        "classification": "BLACKWELL_SANDBOX_ONLY + "
                          "DRAFTER_NUMERICS_DEVIATION_RECORDED",
        "B1_subspace": subspace,
        "B2_margins": results,
        "interpretation_hint": (
            "dm_hold/dm_r64 (signed): positive = correcting the error "
            "along the true direction improves the margin. If both are "
            "~0, the prediction error sits in margin-flat directions "
            "— explaining why state improvement is behaviorally "
            "inert."),
    }
    with open(f"{OUT}/nanospark_b1b2_sensitivity.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    log(f"wrote {OUT}/nanospark_b1b2_sensitivity.json")

    # report
    a = results.get("aggregate", {})
    log("\n=== B1+B2 SENSITIVITY RESULTS ===")
    log(f"hold top-1 match: {a.get('hold_top1_rate')}")
    log(f"r64 top-1 match:  {a.get('r64_top1_rate')}")
    log(f"mean margin:      {a.get('mean_margin')}")
    log(f"mean KL hold:     {a.get('mean_kl_hold')}")
    log(f"mean KL r64:      {a.get('mean_kl_r64')}")
    log(f"mean λ flip:      {a.get('mean_lambda_flip')}")
    log(f"frozen_margin(true)={subspace['frozen_margin_true_mean']}")
    log(f"dm_hold signed={subspace['dm_hold_mean']}")
    log(f"dm_r64 signed={subspace['dm_r64_mean']}")
    log(f"=== DONE in {(time.time()-t00)/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
