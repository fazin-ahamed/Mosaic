"""D3-v1 closed-loop evaluator: student + real source tail + FP32 head.

Implements the roadmap D3-v1 production loop:

    exact committed h40 prefix -> tail prefill -> exact root token
    -> student predicts next h40 (conditioned on its own draft token)
    -> real L41/L42 tail step -> FP32 head -> draft token
    -> draft token fed back as the next driver -> repeat

Modes per validation prompt:
  cl_loop   closed loop: student's own tail-decoded drafts fed back
  tf_loop   true tokens fed back (isolates state-dynamics error from
            token-error compounding)

Both are scored by prefix survival against the true greedy labels,
K speculative decisions excluding the exact root. The exact replay
sanity (h42 bit-equal + greedy labels) gates every prompt.

Requires GPU for the source tail (CUDA) while the student runs on
CPU. This is the D3 promotion metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from mosaicvm.bench.d3_v1 import (build_transitions, load_embed,
                                  pos_encode, make_feat, D_STATE)
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d1b_tail_robustness import (prefix_metrics,
                                                prompt_mean_ci,
                                                reset_decode_state)
from mosaicvm.bench.source_tail import SourceTail
from mosaicvm.bench.d3_v1 import DeltaNet

HORIZON = 32


class TailRunner:
    """Incremental source-tail: prefill once, then step per position.

    Blocks keep their internal KV/DSA caches between step() calls,
    mirroring SourceTail.replay's sequential semantics.
    """

    def __init__(self, device="cuda", max_seq=512, cache_cap=96):
        self.tail = SourceTail(device=device, backend="reference",
                               max_seq=max_seq, cache_cap=cache_cap)
        self.device = device

    def reset(self):
        reset_decode_state(self.tail.blocks)
        for rows in self.tail.route_rows.values():
            rows.clear()

    @torch.inference_mode()
    def prefill(self, h40_prefix, tokens):
        """Process committed positions [0, S) as one batch -> h42 rows.
        Input [S, 4, 4096] -> block batch [1, S, 4, 4096]."""
        assert h40_prefix.ndim == 3 and h40_prefix.shape[1:] == (4, 4096)
        x = h40_prefix.to(device=self.device,
                          dtype=torch.bfloat16).unsqueeze(0)
        ids = torch.tensor(tokens[:len(h40_prefix)], device=self.device,
                           dtype=torch.long).unsqueeze(0)
        with torch.device(self.device):
            for block in self.tail.blocks:
                x = block(x, 0, ids)
        return x.squeeze(0).cpu()

    @torch.inference_mode()
    def step(self, h40_p, p, token_p):
        """Process one position p -> h42[p] (CPU tensor [4, 4096]).
        Input [4, 4096] -> block batch [1, 1, 4, 4096]."""
        assert h40_p.ndim == 2 and h40_p.shape == (4, 4096)
        h = h40_p.to(device=self.device,
                     dtype=torch.bfloat16).view(1, 1, 4, 4096)
        ids = torch.tensor([[token_p]], device=self.device,
                           dtype=torch.long)
        with torch.device(self.device):
            for block in self.tail.blocks:
                h = block(h, p, ids)
        return h.squeeze(0).squeeze(0).cpu()

    def logits(self, h42_row):
        """FP32 head on one h42 row [4, 4096] -> [V] logits."""
        lg = self.tail.logits(h42_row.unsqueeze(0))
        return lg[0]


def load_student(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = DeltaNet(ck["hidden"], ck["depth"], ck["rank"],
                     aux_out=ck.get("h42_aux_weight", 0) > 0
                     or "aux.weight" in ck["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    return (model, ck["target_mean"], ck["target_std"],
            ck["x_mean"], ck["x_std"])


def student_step(model, embed, state, delta_in, driver_tok, cur_tok,
                 pos, mean, std, x_mean, x_std):
    feat = make_feat(embed, state, delta_in, driver_tok, cur_tok, pos)
    feat = (feat - x_mean) / x_std
    with torch.no_grad():
        pred = model(feat.unsqueeze(0))[0] * std + mean
    return pred


def run_loop(runner, model, embed, mean, std, x_mean, x_std,
             capture, mode, K):
    """Closed-loop or teacher-forced rollout for one prompt.

    Returns (predicted_h40 [K,4,4096] bf16, hits list of K bools).
    """
    c = capture
    S = c["S"]
    toks = c["tokens"]
    h = c["h40"].float()

    runner.reset()
    h42_pref = runner.prefill(h[:S], toks)     # positions 0..S-1
    root_draft = int(runner.logits(h42_pref[-1]).argmax())
    assert root_draft == toks[S], "root token mismatch (exact root)"

    state = h[S - 1].flatten()
    prev_state = h[S - 2].flatten()
    cur_tok = toks[S - 1]                      # last committed token
    driver = root_draft                        # token at position S

    preds, hits = [], []
    for j in range(K):
        p = S + j                              # position being predicted
        delta_in = state - prev_state
        pred = student_step(model, embed, state, delta_in, driver,
                            cur_tok, p, mean, std, x_mean, x_std)
        new_state = state + pred
        h42_p = runner.step(new_state.view(4, 4096), p, driver)
        draft = int(runner.logits(h42_p).argmax())
        hits.append(draft == toks[p + 1])
        preds.append(new_state.view(4, 4096))
        # feedback for the next position. cl_loop contract (roadmap
        # S16): after the exact root, ONLY the model's own drafts —
        # no gold future tokens, no gold state injection, no
        # re-anchoring, no gold-based clamps.
        cur_tok = driver
        if mode == "cl_loop":
            driver = draft    # structural guarantee: this branch
            # never reads future gold tokens; enforced by
            # test_d3v1_alignment.py (roadmap S16)
        else:
            driver = toks[p + 1]
        prev_state = state
        state = new_state
    return torch.stack(preds).to(torch.bfloat16), hits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True,
                   help="d3_v1 output dir (student.pt)")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path,
                   default=Path("artifacts/stage1_prompts.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-cap", type=int, default=96)
    p.add_argument("--max-K", type=int, default=HORIZON)
    a = p.parse_args()

    model, mean, std, x_mean, x_std = load_student(
        a.predictions / "student.pt")
    embed = load_embed()
    captures = load_captures(a.data,
                             json.loads(a.manifest.read_text()),
                             None, a.max_K)
    val = [c for c in captures if c["split"] == "val"]
    print(f"val prompts: {len(val)}", flush=True)

    runner = TailRunner(device=a.device, cache_cap=a.cache_cap)
    rows = []
    t0 = time.time()
    for c in val:
        K = min(a.max_K, c["h40"].shape[0] - c["S"])
        S = c["S"]
        # exact-replay sanity via full replay (gates the prompt)
        exact = torch.cat([c["h40"][:S], c["h40"][S:S + K]])
        y0 = runner.tail.replay(exact, c["tokens"], S)
        lg0 = runner.tail.logits(y0)
        target = torch.tensor(c["tokens"][S:S + K + 1],
                              dtype=torch.long)
        bit_equal = torch.equal(y0.view(torch.int16),
                                c["h42"][:S + K].view(torch.int16))
        greedy_equal = torch.equal(
            lg0[S - 1:S + K].argmax(-1), target)
        if not (bit_equal and greedy_equal):
            raise RuntimeError(
                f'{c["category"]}: exact replay failed '
                f'h42={bit_equal} head={greedy_equal}')

        prompt_row = {"category": c["category"], "K": K,
                      "exact_h42_bit_equal": True,
                      "exact_head_greedy_equal": True}
        for mode in ("cl_loop", "tf_loop"):
            preds, hits = run_loop(runner, model, embed, mean, std,
                                   x_mean, x_std, c, mode, K)
            metr = prefix_metrics([int(x) for x in hits], K)
            tgt = c["h40"][S:S + K].float()
            pf = preds.float().flatten(1)
            rel = ((pf - tgt.flatten(1)).norm(dim=1)
                   / tgt.flatten(1).norm(dim=1).clamp_min(1e-6))
            metr["state_rel_l2_mean"] = float(rel.mean())
            prompt_row[mode] = metr
        rows.append(prompt_row)
        print(f'{c["category"]} ({time.time()-t0:.0f}s) '
              f'cl E[A]={prompt_row["cl_loop"]["E_A"]} '
              f'tf E[A]={prompt_row["tf_loop"]["E_A"]} '
              f'cl relL2={prompt_row["cl_loop"]["state_rel_l2_mean"]:.3f}',
              flush=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.with_suffix(".partial.json").write_text(
            json.dumps({"complete": False, "rows": rows}, indent=2))

    aggregate = {}
    for mode in ("cl_loop", "tf_loop"):
        vals = [r[mode]["E_A"] for r in rows]
        aggregate[mode] = {
            "n_prompts": len(rows),
            "E_A": sum(vals) / len(vals),
            "exploratory_prompt_ci95": prompt_mean_ci(vals),
            "per_prompt_E_A": vals,
            "S8": sum(r[mode]["Sk"][7] for r in rows) / len(rows),
            "S16": sum(r[mode]["Sk"][15] for r in rows) / len(rows),
        }
    result = {
        "classification": "D3_V1_CLOSED_LOOP_SOURCE_TAIL_EVAL",
        "production_certified": False,
        "student_pass": False,
        "geometry": "K uncertain decisions exclude exact root; "
                    "prefix+1 (E[A] convention)",
        "modes": {"cl_loop": "student's own tail-decoded drafts fed "
                             "back",
                  "tf_loop": "true tokens fed back"},
        "device": a.device,
        "checkpoint": str((a.predictions / "student.pt").resolve()),
        "rows": rows,
        "aggregate": aggregate,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
    }
    a.output.write_text(json.dumps(result, indent=2))
    a.output.with_suffix(".partial.json").unlink(missing_ok=True)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
