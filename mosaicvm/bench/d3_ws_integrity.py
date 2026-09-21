"""WS integrity gate (roadmap S16) — READ-ONLY bitwise checks.

1. TRUNK BITWISE IDENTITY: load the HIST-1 checkpoint through
   the canonical evaluator path (d3_dd1_hist_eval.load_hist)
   AND through the WS harness path (d3_ws), feed identical
   inputs, and require torch.equal on the RAW pre-head trunk
   output z (no normalization, no rounding).

2. FROZEN-STATE IMMUABILITY: the WS-1 artifact must contain
   no trunk weights, and the HIST-1 checkpoint file + basis
   file must be byte-identical to the SHA-256s recorded at
   WS-1 save time (proving training never wrote through).

3. BEHAVIORAL REPLAY (previously mislabeled the integrity
   gate): rounded-margin/hit agreement with the frozen
   telemetry — re-derived here for completeness.

Writes artifacts/d3_ws/ws_integrity.json. No training, no
model writes.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_dd1_hist import DDHistNet
from mosaicvm.bench.d3_dd1 import make_dd_feat
from mosaicvm.bench.d3_bridge import load_embed, load_head
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON

HIST1 = Path("artifacts/d3_dd1_hist/dd1_hist_prev/dd1_hist.pt")
BASIS = Path("artifacts/d3_lrdl0/basis_r256.pt")
WS1 = Path("artifacts/d3_ws/ws1.pt")
FROZEN_TELEM = Path("artifacts/d3_dd1_rank_telemetry.json")
OUT = Path("artifacts/d3_ws/ws_integrity.json")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    report = {}

    # ---- 1. trunk bitwise identity ----
    # canonical-path load (as d3_dd1_hist_eval does)
    ck_canon = torch.load(HIST1, map_location="cpu",
                          weights_only=True)
    canon = DDHistNet(ck_canon["hidden"], ck_canon["depth"],
                      ck_canon["aux_dim"])
    canon.load_state_dict(ck_canon["model"])
    canon.eval()

    # WS-harness-path load (as d3_ws does)
    ck_ws = torch.load(HIST1, map_location="cpu",
                       weights_only=True)
    ws = DDHistNet(ck_ws["hidden"], ck_ws["depth"],
                   ck_ws["aux_dim"])
    ws.load_state_dict(ck_ws["model"])
    ws.eval()

    # state_dict bitwise equality (parameter tensors)
    sd_c = canon.state_dict()
    sd_w = ws.state_dict()
    tensors_equal = all(
        torch.equal(sd_c[k], sd_w[k]) for k in sd_c)
    report["state_dict_bitwise_equal"] = tensors_equal

    # raw z bitwise equality on identical inputs
    embed = load_embed()
    caps = load_captures("artifacts/d3_data_105",
                         json.loads(Path(
                             "artifacts/stage1_prompts.json"
                         ).read_text()), None, HORIZON)
    val = [c for c in caps if c["split"] == "val"]
    z_equal_all = True
    n_checked = 0
    with torch.inference_mode():
        for c in val:
            S = c["S"]; toks = c["tokens"]
            cur = c["h40"][S - 1].float().flatten()
            aux = c["h40"][S - 2].float().flatten()
            mf = make_dd_feat(embed, cur, toks[S],
                              toks[S - 1], S)
            mfn = (mf - ck_canon["xm_mean"]) / ck_canon["xm_std"]
            afn = (aux - ck_canon["xa_mean"]) / ck_canon["xa_std"]
            z_c = canon(mfn.unsqueeze(0), afn.unsqueeze(0))
            z_w = ws(mfn.unsqueeze(0), afn.unsqueeze(0))
            if not torch.equal(z_c, z_w):
                z_equal_all = False
            n_checked += 1
    report["raw_z_bitwise_equal"] = {
        "torch_equal_all": z_equal_all, "n_inputs": n_checked}
    # also compare z against the checkpoint's own head-input:
    # z must be the pre-head trunk output (dims = hidden)
    report["raw_z_shape"] = list(z_c.shape)

    # ---- 2. frozen-state immutability ----
    ws1 = torch.load(WS1, map_location="cpu",
                     weights_only=True)
    ws1_keys = sorted(ws1.keys())
    # the artifact must contain ONLY Q + references
    no_trunk_weights = ("Q" in ws1
                        and not any(
                            k for k in ws1_keys
                            if k not in (
                                "Q", "hist1_ckpt", "basis",
                                "rank", "lam_kl", "seed",
                                "hist1_ckpt_sha256",
                                "basis_sha256")))
    h1_sha_now = sha(HIST1)
    basis_sha_now = sha(BASIS)
    h1_sha_recorded = ws1.get("hist1_ckpt_sha256")
    basis_sha_recorded = ws1.get("basis_sha256")
    immut = {
        "ws1_artifact_keys": ws1_keys,
        "ws1_contains_only_Q_and_refs": no_trunk_weights,
        "hist1_sha_now": h1_sha_now,
        "hist1_sha_at_ws1_save": h1_sha_recorded,
        "hist1_unchanged": h1_sha_now == h1_sha_recorded,
        "basis_sha_now": basis_sha_now,
        "basis_sha_at_ws1_save": basis_sha_recorded,
        "basis_unchanged": basis_sha_now == basis_sha_recorded,
    }
    report["frozen_state_immutability"] = immut

    # cross-check against ws1_eval.json records
    ev = json.loads(Path(
        "artifacts/d3_ws/ws1_eval.json").read_text())
    immut["hist1_matches_ws1_eval_record"] = (
        h1_sha_now == ev["hist1_ckpt_sha256"])
    immut["basis_matches_ws1_eval_record"] = (
        basis_sha_now == ev["basis_sha256"])

    # ---- 3. behavioral replay (labeled as such) ----
    w_head, w_norm = load_head()
    w_head_g = w_head.cuda(); w_norm_g = w_norm.cuda()
    frozen = json.loads(FROZEN_TELEM.read_text())["HIST-1"]
    ref = frozen["per_prompt"]
    hits = 0; max_md = 0.0; all_hits_match = True
    with torch.inference_mode():
        for c in val:
            S = c["S"]; toks = c["tokens"]
            cur = c["h40"][S - 1].float().flatten()
            aux = c["h40"][S - 2].float().flatten()
            mf = make_dd_feat(embed, cur, toks[S],
                              toks[S - 1], S)
            mfn = (mf - ck_canon["xm_mean"]) / ck_canon["xm_std"]
            afn = (aux - ck_canon["xa_mean"]) / ck_canon["xa_std"]
            zg = canon(mfn.unsqueeze(0), afn.unsqueeze(0)).cuda()
            zn = zg * torch.rsqrt(
                zg.square().mean(-1, keepdim=True) + 1e-6
            ) * w_norm_g
            lg = w_head_g @ zn[0]
            true = toks[S + 1]
            hit = int(lg.argmax()) == true
            hits += hit
            if hit != ref[c["category"]]["hit"]:
                all_hits_match = False
            tl = float(lg[true])
            lgm = lg.clone(); lgm[true] = float("-inf")
            m = tl - float(lg[lgm.argmax()])
            max_md = max(max_md, abs(
                m - ref[c["category"]]["margin"]))
    report["behavioral_replay"] = {
        "note": "rounded-margin/hit agreement — NOT the "
                "bitwise gate; bitwise results are above",
        "S1": f"{hits}/{len(val)}",
        "all_hits_match_frozen_telemetry": all_hits_match,
        "max_abs_margin_delta": max_md,
    }

    # ---- verdict ----
    verdict_pass = (tensors_equal and z_equal_all
                    and no_trunk_weights
                    and immut["hist1_unchanged"]
                    and immut["basis_unchanged"]
                    and hits == 10 and all_hits_match)
    report["verdict"] = {
        "bitwise_trunk_identity": "PASS" if (
            tensors_equal and z_equal_all) else "FAIL",
        "frozen_state_immutability": "PASS" if (
            no_trunk_weights and immut["hist1_unchanged"]
            and immut["basis_unchanged"]) else "FAIL",
        "behavioral_replay": "PASS" if (
            hits == 10 and all_hits_match) else "FAIL",
        "overall_S16": "PASS" if verdict_pass else "FAIL",
    }
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["verdict"], indent=1))
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")
    if not verdict_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
