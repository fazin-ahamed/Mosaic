"""G2 calibration: kit-source (Blackwell) vs kit-source (RTX 3060).

Compares the two source-class corpus records on the SAME calibration
prompts to measure natural cross-hardware source-implementation
variability — the basis for the G2 envelope thresholds (roadmap §21-34):

  tau_hidden    layerwise rel_l2 / cosine envelope at matched steps
  tau_router    Top-6 boundary band: score gaps around rank 6 where
                route sets flip between the two source runs
  tau_w         route-weight |dw| on commonly selected experts
  tau_logit     |dlogit| on common top-5 tokens + the top1-top2 margin
                band where greedy flips occur

Neither side is the MosaicVM production runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

A_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
             "artifacts/g2_calibration_corpus")   # Blackwell kit
B_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else
             "artifacts/g2_calib_3060")           # RTX 3060 kit
OUT = Path("artifacts/g2_calibration_result.json")


def load(d):
    out = {}
    for f in sorted(Path(d).glob("*.pt")):
        r = torch.load(f, map_location="cpu", weights_only=False)
        out[f.stem] = r
    return out


def main():
    a, b = load(A_DIR), load(B_DIR)
    common = sorted(set(a) & set(b))
    assert common, "no common calibration files"
    stats = {
        "files": common,
        "prompts": [],
    }
    hid_rel, hid_cos = [], []
    w_diffs = []
    route_flips = []       # (layer, boundary gap) where sets differ
    route_agree = 0
    route_tot = 0
    logit_diffs = []
    flip_margins = []
    agree_margins = []
    tok_exact = 0
    tok_tot = 0

    for name in common:
        ra, rb = a[name], b[name]
        assert ra["prompt"] == rb["prompt"], f"prompt mismatch {name}"
        # token agreement
        gen_a, gen_b = ra["generated"], rb["generated"]
        n = min(len(gen_a), len(gen_b))
        prefix = 0
        for x, y in zip(gen_a, gen_b):
            if x != y:
                break
            prefix += 1
        tok_exact += prefix
        tok_tot += n
        first_div = prefix if prefix < n else None

        # hidden drift at matched decode steps
        ha = {(h["step"], h["layer"]): h["h_mean"].float()
              for h in ra["hiddens"]}
        hb = {(h["step"], h["layer"]): h["h_mean"].float()
              for h in rb["hiddens"]}
        prompt_rel = []
        for k in ha.keys() & hb.keys():
            if first_div is not None and k[0] >= first_div:
                continue  # only matched-prefix steps
            x, y = ha[k], hb[k]
            rel = ((x - y).norm() / y.norm().clamp_min(1e-6)).item()
            cos = torch.nn.functional.cosine_similarity(
                x, y, dim=0).item()
            hid_rel.append(rel)
            hid_cos.append(cos)
            prompt_rel.append((k, rel))

        # routes at matched steps (decode only, step < prefix; and the
        # prefill last row for token 0)
        ga = {}
        gb = {}
        for g in ra["gates"]:
            key = ("p", g["layer"]) if g["phase"] == "prefill" \
                else ("d", g["step"], g["layer"])
            ga[key] = g
        for g in rb["gates"]:
            key = ("p", g["layer"]) if g["phase"] == "prefill" \
                else ("d", g["step"], g["layer"])
            gb[key] = g
        for key in ga.keys() & gb.keys():
            step_ok = (key[0] == "p") or \
                (first_div is None or key[1] < first_div - 1)
            if not step_ok:
                continue
            if key[0] == "d" and first_div is not None \
                    and key[1] >= first_div - 1:
                continue
            ia = ga[key]["idx"][-1]
            ib = gb[key]["idx"][-1]
            wa = ga[key]["w"][-1]
            wb = gb[key]["w"][-1]
            route_tot += 1
            if set(ia) == set(ib):
                route_agree += 1
                for x, y in zip(wa, wb):
                    w_diffs.append(abs(x - y))
            else:
                # boundary gap: rank-6 vs rank-7 score separation from
                # the top-10 records (when present)
                t10 = ga[key].get("bias_top10_vals")
                gap = (t10[5] - t10[6]) if t10 and len(t10) > 6 else None
                route_flips.append({"file": name, "key": list(key),
                                    "gap_rank67": gap,
                                    "a_ids": ia, "b_ids": ib})
                for e in set(ia) & set(ib):
                    i, j = ia.index(e), ib.index(e)
                    w_diffs.append(abs(wa[i] - wb[j]))

        # logits at matched positions (incl. the divergence position)
        for k in range(min(prefix + 1, n)):
            ta = ra["top5_per_token"][k]
            tb = rb["top5_per_token"][k]
            ma = dict(zip(ta["ids"], ta["vals"]))
            mb = dict(zip(tb["ids"], tb["vals"]))
            for t in set(ma) & set(mb):
                logit_diffs.append(abs(ma[t] - mb[t]))
            margin = ta["vals"][0] - ta["vals"][1]
            if k == prefix and prefix < n:
                flip_margins.append(margin)   # a's top1-top2 at flip
            elif k < prefix and k >= 1:
                agree_margins.append(margin)

        stats["prompts"].append({
            "file": name, "prefix": prefix, "n": n,
            "gen_a": gen_a[:n], "gen_b": gen_b[:n],
        })

    def q(v, p):
        v = sorted(v)
        return round(v[int(p * (len(v) - 1))], 6) if v else None

    result = {
        "classification": "G2_SOURCE_VS_SOURCE_CALIBRATION",
        "sides": {
            "a": "kit inference/model.py + torch shims, RTX PRO 6000 "
                 "Blackwell",
            "b": "kit inference/model.py + torch shims, RTX 3060 "
                 "(CPU head/embed, streamed experts)",
        },
        "token_exact_total": tok_exact,
        "token_total": tok_tot,
        "tau_hidden": {
            "rel_l2": {"p50": q(hid_rel, .5), "p90": q(hid_rel, .9),
                       "p99": q(hid_rel, .99), "max": q(hid_rel, 1.0),
                       "n": len(hid_rel)},
            "cosine": {"p01": q(hid_cos, .01), "p10": q(hid_cos, .1),
                       "p50": q(hid_cos, .5), "n": len(hid_cos)},
        },
        "tau_router": {
            "route_set_agree": f"{route_agree}/{route_tot}",
            "route_flips": len(route_flips),
            "flip_gap_rank67": {
                "p50": q([f["gap_rank67"] for f in route_flips
                          if f["gap_rank67"] is not None], .5),
                "p90": q([f["gap_rank67"] for f in route_flips
                          if f["gap_rank67"] is not None], .9),
                "max": q([f["gap_rank67"] for f in route_flips
                          if f["gap_rank67"] is not None], 1.0),
                "n": sum(1 for f in route_flips
                         if f["gap_rank67"] is not None),
            },
        },
        "tau_w": {
            "abs_dw": {"p50": q(w_diffs, .5), "p90": q(w_diffs, .9),
                       "p99": q(w_diffs, .99), "max": q(w_diffs, 1.0),
                       "n": len(w_diffs)},
        },
        "tau_logit": {
            "abs_dlogit_common_top5": {
                "p50": q(logit_diffs, .5), "p90": q(logit_diffs, .9),
                "p99": q(logit_diffs, .99), "max": q(logit_diffs, 1.0),
                "n": len(logit_diffs)},
            "flip_margin_top12": {
                "vals": [round(v, 4) for v in flip_margins]},
            "agree_margin_top12_p10": q(agree_margins, .1),
        },
        "per_prompt": stats["prompts"],
        "route_flip_examples": route_flips[:20],
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("per_prompt", "route_flip_examples")},
                     indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
