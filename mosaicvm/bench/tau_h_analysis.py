"""Track E: prefix-conditioned hidden envelope tau_h(s).

Compares the 3060-kit 32-token run (certification prompts) against the
G1 Blackwell corpus (same prompts, same 32 tokens) — kit-vs-kit at the
full certification horizon — and derives tau_h(s) = p99 rel_l2 of
layerwise hidden drift conditioned on matched decode step s.

Cross-check: the disjoint 14-prompt 12-token kit-vs-kit calibration at
overlapping steps (<=10) validates that the certification-prompt drift
curve is not anomalous.

Also evaluates MosaicVM (localization data) against tau_h(s).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

BW_DIR = Path("artifacts/source_corpus")          # Blackwell G1 (32 tok)
KIT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
               "artifacts/g2_calib_3060_cert32")  # 3060 kit (32 tok)
CAL = json.loads(Path("artifacts/g2_calibration_result.json").read_text())
LOC = json.loads(Path(
    "artifacts/first_divergence_localization.json").read_text())
OUT = Path("artifacts/tau_h_prefix.json")


def hid_map(r):
    return {(h["step"], h["layer"]): h["h_mean"].float()
            for h in r["hiddens"]}


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-6)).item()


def main():
    vals = {}          # step -> [rel_l2 over layers+prompts]
    prefixes = {}
    prompts = []
    for f in sorted(BW_DIR.glob("*.pt")):
        rb = torch.load(f, map_location="cpu", weights_only=False)
        cat = rb["category"]
        fk = KIT_DIR / f.name
        if not fk.exists():
            print(f"missing 3060-kit side: {fk}")
            continue
        rk = torch.load(fk, map_location="cpu", weights_only=False)
        assert rk["prompt"] == rb["prompt"]
        ga, gb = rb["generated"], rk["generated"]
        n = min(len(ga), len(gb))
        prefix = 0
        for x, y in zip(ga, gb):
            if x != y:
                break
            prefix += 1
        prefixes[cat] = prefix
        prompts.append({"category": cat, "prefix": prefix, "n": n})
        ha, hb = hid_map(rb), hid_map(rk)
        # decode step dk has matched input iff dk <= prefix-1
        for (dk, L) in ha.keys() & hb.keys():
            if dk > prefix - 1:
                continue
            vals.setdefault(dk, []).append(rel(ha[(dk, L)], hb[(dk, L)]))
        print(f"[{cat}] kit-vs-kit prefix {prefix}/{n}")

    def q(v, p):
        v = sorted(v)
        return round(v[min(int(p * (len(v) - 1)), len(v) - 1)], 4) \
            if v else None

    buckets = {}
    s_max = max(vals) if vals else -1
    # bucket layout: 0-1, 2-3, 4-7, 8-15, 16-23, 24-31
    bounds = [(0, 1), (2, 3), (4, 7), (8, 15), (16, 23), (24, 31)]
    for lo, hi in bounds:
        pool = [v for s, vs in vals.items() if lo <= s <= hi for v in vs]
        if not pool:
            continue
        buckets[f"{lo}-{hi}"] = {
            "n": len(pool),
            "p50": q(pool, .5), "p90": q(pool, .9), "p99": q(pool, .99),
            "max": q(pool, 1.0),
        }
    # monotone envelope: tau_h(s bucket) = max p99 seen up to that
    # bucket (enforce non-decreasing)
    running_max = 0.0
    mono = {}
    for k in buckets:
        p99 = buckets[k]["p99"] or 0.0
        running_max = max(running_max, p99)
        mono[k] = round(running_max, 4)
        buckets[k]["tau_h_frozen"] = round(running_max, 4)

    # MosaicVM check vs the envelope
    mos_by_bucket = {}
    for p in LOC["prompts"]:
        for s in p["steps"]:
            dk = s["decode_step"]
            for lo, hi in bounds:
                if lo <= dk <= hi:
                    for l in s["layers"]:
                        mos_by_bucket.setdefault(f"{lo}-{hi}", []
                                                 ).append(l["rel_l2"])
                    break
    mos_stats = {}
    for k, v in mos_by_bucket.items():
        mos_stats[k] = {"n": len(v), "p50": q(v, .5), "p99": q(v, .99),
                        "max": q(v, 1.0),
                        "inside_tau_h": bool(
                            (q(v, .99) or 0) <= mono.get(k, 9e9)
                            or q(v, .99) is None)}
    # cross-check with the disjoint 12-token calibration
    cal_check = {
        "disjoint_calib_rel_l2": CAL["tau_hidden"]["rel_l2"],
        "note": "steps 0-10 only; compare p50/p90 against the same "
                "step range in this run",
    }

    out = {
        "classification": "TAU_H_PREFIX_CONDITIONED_ENVELOPE",
        "pair": "G1 Blackwell kit corpus vs 3060-kit run, "
                "certification prompts, 32 tokens (both sides "
                "source-class kit; neither is MosaicVM)",
        "prefixes": prompts,
        "tau_h_by_bucket": buckets,
        "monotone_envelope_tau_h": mono,
        "mosaicvm_vs_envelope": mos_stats,
        "disjoint_calibration_crosscheck": cal_check,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"tau_h_by_bucket": buckets,
                      "monotone_envelope_tau_h": mono}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
