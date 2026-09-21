"""D-race evaluator for direct-h42 drafters (roadmap S34-S35,
S39-S41): the student recurses on h42 and drafts through the
frozen production FP32 head — NO source L41/L42 tail in the loop.

Modes per validation prompt (same 21-prompt set, same seeds):
  one_step  gold h42_{t-1} + true driver -> predicted h42_t
            -> FP32 head -> draft vs true token
  cl_loop   chained predicted h42; driver = the model's own
            head-decoded draft token (production semantics)
  tf_loop   chained predicted h42; true tokens

Also measures the PROPOSER WALL (co-primary): per speculative
position, student forward + head projection, in two head
placements (CPU fp32 GEMV as in the current evaluator; GPU fp32
GEMM) — the number that competes against the 13.5 ms/position
source-tail marginal.

Prefix survival/E[A] convention is identical to d3_v1_eval
(exact root excluded from K; prefix+1 emission).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

sys = None
import sys as _sys
_sys.path.insert(0, ".")
_sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_v1 import (build_transitions, load_embed,
                                  make_feat, pos_encode, DeltaNet,
                                  D_STATE)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d1b_tail_robustness import (prefix_metrics,
                                                prompt_mean_ci)
from mosaicvm.bench.source_tail import SourceTail

HORIZON = 32


def load_student(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ck.get("target_state") != "h42":
        raise ValueError("not an h42-target student")
    model = DeltaNet(ck["hidden"], ck["depth"], ck["rank"],
                     aux_out=ck.get("h42_aux_weight", 0) > 0
                     or "aux.weight" in ck["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    return (model, ck["target_mean"], ck["target_std"],
            ck["x_mean"], ck["x_std"])


def student_delta(model, embed, state, delta_in, driver_tok,
                  cur_tok, pos, mean, std, x_mean, x_std):
    feat = make_feat(embed, state, delta_in, driver_tok, cur_tok,
                     pos)
    feat = (feat - x_mean) / x_std
    with torch.no_grad():
        return model(feat.unsqueeze(0))[0] * std + mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True,
                   help="d3_v1 output dir (student.pt, h42 target)")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path,
                   default=Path("artifacts/stage1_prompts.json"))
    a = p.parse_args()
    torch.set_num_threads(4)

    model, mean, std, x_mean, x_std = load_student(
        a.predictions / "student.pt")
    embed = load_embed()
    captures = load_captures(a.data,
                             json.loads(a.manifest.read_text()),
                             None, HORIZON)
    val = [c for c in captures if c["split"] == "val"]
    print(f"val prompts: {len(val)}", flush=True)

    # frozen FP32 head via SourceTail (hc_head + norm + fp32
    # linear); no replay is ever called — no experts touched.
    tail = SourceTail(device="cuda", backend="reference",
                      max_seq=64, cache_cap=4)
    # GPU head variant for the latency comparison
    head_gpu = tail.head.cuda()

    def head_cpu(h42_row):
        lg = tail.logits(h42_row.unsqueeze(0))
        return lg[0]

    def head_gpu_f(h42_row):
        with torch.inference_mode():
            h = tail.blocks[-1].hc_head(
                h42_row.unsqueeze(0).unsqueeze(0).cuda()
                .to(torch.bfloat16), *tail.hc)
            h = tail.norm(h).squeeze(0).float()
            return torch.nn.functional.linear(h, head_gpu)[0].cpu()

    rows = []
    lat = {"cpu_ms": [], "gpu_ms": []}
    t0 = time.time()
    for c in val:
        S = c["S"]
        K = min(HORIZON, c["h42"].shape[0] - S)
        h = c["h42"].float()
        toks = c["tokens"]

        # exact-root sanity: head on the exact committed h42[S-1]
        root_logits = head_cpu(h[S - 1])
        assert int(root_logits.argmax()) == toks[S], \
            f"{c['category']}: exact h42 root does not give the " \
            f"true first token"
        root_draft = toks[S]

        def run(mode):
            state = h[S - 1].flatten()
            prev = h[S - 2].flatten()
            cur_tok = toks[S - 1]
            driver = root_draft
            preds, hits = [], []
            for j in range(K):
                t = S + j
                delta_in = state - prev
                if mode == "one_step":
                    state_in = h[t - 1].flatten()
                    delta_in = h[t - 1].flatten() - h[t - 2].flatten()
                    cur_in = toks[t - 1]
                    drv_in = toks[t]
                else:
                    state_in = state
                    cur_in = cur_tok
                    drv_in = driver
                pred = student_delta(model, embed, state_in,
                                     delta_in, drv_in, cur_in, t,
                                     mean, std, x_mean, x_std)
                new_state = state_in + pred
                lg = head_cpu(new_state.view(4, 4096))
                draft = int(lg.argmax())
                hits.append(draft == toks[t + 1])
                preds.append(new_state)
                cur_tok = drv_in
                driver = draft if mode == "cl_loop" else toks[t + 1]
                prev = state_in
                state = new_state
            return torch.stack(preds).to(torch.bfloat16), hits

        row = {"category": c["category"], "K": K,
               "exact_root_ok": True}
        for mode in ("one_step", "cl_loop", "tf_loop"):
            preds, hits = run(mode)
            row[mode] = prefix_metrics([int(x) for x in hits], K)
            tgt = h[S:S + K].float().flatten(1)
            pf = preds.float().flatten(1)
            row[mode]["state_rel_l2_mean"] = float(
                ((pf - tgt).norm(dim=1)
                 / tgt.norm(dim=1).clamp_min(1e-6)).mean())
        rows.append(row)
        print(f"{c['category']} ({time.time()-t0:.0f}s) "
              f"os {row['one_step']['E_A']} "
              f"cl {row['cl_loop']['E_A']} "
              f"tf {row['tf_loop']['E_A']}", flush=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.with_suffix(".partial.json").write_text(
            json.dumps({"complete": False, "rows": rows}, indent=2))

    # proposer wall: per-position student+head latency
    c = val[0]
    S, K = c["S"], min(HORIZON, c["h42"].shape[0] - S)
    h = c["h42"].float()
    toks = c["tokens"]
    for _ in range(3):        # warm-up
        student_delta(model, embed, h[S - 1].flatten(),
                      h[S - 1].flatten() - h[S - 2].flatten(),
                      toks[S], toks[S - 1], S, mean, std,
                      x_mean, x_std)
        head_cpu(h[S - 1]); head_gpu_f(h[S - 1])
    for rep in range(20):
        t = time.perf_counter()
        st = student_delta(model, embed, h[S - 1].flatten(),
                           h[S - 1].flatten() - h[S - 2].flatten(),
                           toks[S], toks[S - 1], S, mean, std,
                           x_mean, x_std)
        head_cpu(st.view(4, 4096))
        lat["cpu_ms"].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        st = student_delta(model, embed, h[S - 1].flatten(),
                           h[S - 1].flatten() - h[S - 2].flatten(),
                           toks[S], toks[S - 1], S, mean, std,
                           x_mean, x_std)
        head_gpu_f(st.view(4, 4096))
        torch.cuda.synchronize()
        lat["gpu_ms"].append((time.perf_counter() - t) * 1e3)
    lat_ms = {k: round(sorted(v)[len(v) // 2], 3)
              for k, v in lat.items()}

    aggregate = {}
    for mode in ("one_step", "cl_loop", "tf_loop"):
        vals = [r[mode]["E_A"] for r in rows]
        aggregate[mode] = {
            "n_prompts": len(rows),
            "E_A": sum(vals) / len(vals),
            "exploratory_prompt_ci95": prompt_mean_ci(vals),
            "per_prompt_E_A": vals,
            "S8": sum(r[mode]["Sk"][7] for r in rows) / len(rows),
        }
    result = {
        "classification": "D3_DH42_DIRECT_HEAD_EVAL",
        "production_certified": False,
        "student_pass": False,
        "draft_path": "student h42 recurrence + frozen FP32 head "
                      "(NO source tail)",
        "device": "cuda (head) + cpu (student)",
        "proposer_wall_ms_per_position": lat_ms,
        "vs_source_tail_marginal_ms": 13.5,
        "checkpoint": str((a.predictions / "student.pt").resolve()),
        "rows": rows,
        "aggregate": aggregate,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
    }
    a.output.write_text(json.dumps(result, indent=2))
    a.output.with_suffix(".partial.json").unlink(missing_ok=True)
    print(json.dumps({"aggregate": {m: v["E_A"] for m, v in
                                    aggregate.items()},
                      "proposer_wall_ms": lat_ms}, indent=2))


if __name__ == "__main__":
    main()
