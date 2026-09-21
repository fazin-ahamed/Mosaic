"""Equivalence harness: prove FP32 mmap path == in-RAM FP32 path.

Builds the orig84 subset both ways:
  A. in-RAM: emit_core rows -> torch.stack -> mean/std -> normalize (old semantics)
  B. mmap:   emit_core rows -> disk-backed FP32 tensor -> mean/std -> chunked normalize

Requires torch.equal on: raw xm, raw xa, xm_mean, xm_std, xa_mean, xa_std,
normalized xm, normalized xa, and row ordering (labels y). If all pass, the
mmap change is an execution-only transform and valid for the primary ledger.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "mosaicvm/bench")

from mosaicvm.bench.d3_bridge import load_embed, TEACHER, AUDIT
from mosaicvm.bench.d3_student import load_captures
from mosaicvm.bench.d3_v1 import HORIZON
from mosaicvm.bench.d3_scale1000_train import (
    emit_core, open_mmap, ORIG_CAP, ORIG_MANIFEST, DM, DA)


def build_inram(cats, by, embed):
    xm, xa, labels = [], [], []
    for cat in cats:
        rec = torch.load(TEACHER / f"{cat}.pt", map_location="cpu",
                         weights_only=True)
        rm, ra, labs = emit_core(rec, by[cat], embed)
        xm.append(rm); xa.append(ra); labels.extend(labs)
    return torch.cat(xm), torch.cat(xa), labels


def build_mmap(cats, by, embed, cache):
    # count rows first
    recs = {cat: torch.load(TEACHER / f"{cat}.pt", map_location="cpu",
                            weights_only=True) for cat in cats}
    n = sum(r["K"] for r in recs.values())
    xm = open_mmap(cache / "xm.f32", n, DM)
    xa = open_mmap(cache / "xa.f32", n, DA)
    labels = []
    off = 0
    for cat in cats:
        rm, ra, labs = emit_core(recs[cat], by[cat], embed)
        k = rm.shape[0]
        xm[off:off + k] = rm
        xa[off:off + k] = ra
        labels.extend(labs)
        off += k
    assert off == n
    return xm, xa, labels


def norm_inram(xm, xa):
    xm_mean = xm.mean(0); xm_std = xm.std(0).clamp_min(1e-4)
    xa_mean = xa.mean(0); xa_std = xa.std(0).clamp_min(1e-4)
    xmn = (xm - xm_mean) / xm_std
    xan = (xa - xa_mean) / xa_std
    return xmn, xan, (xm_mean, xm_std, xa_mean, xa_std)


def norm_mmap(xm, xa, cache):
    n = xm.shape[0]
    xm_mean = xm.mean(0); xm_std = xm.std(0).clamp_min(1e-4)
    xa_mean = xa.mean(0); xa_std = xa.std(0).clamp_min(1e-4)
    xmn = open_mmap(cache / "xmn.f32", n, DM)
    xan = open_mmap(cache / "xan.f32", n, DA)
    CH = 1024
    for i in range(0, n, CH):
        sl = slice(i, min(i + CH, n))
        xmn[sl] = (xm[sl] - xm_mean) / xm_std
        xan[sl] = (xa[sl] - xa_mean) / xa_std
    return xmn, xan, (xm_mean, xm_std, xa_mean, xa_std)


def main():
    ver = json.loads(AUDIT.read_text())
    assert ver["verdict"] == "PASS"
    cats = sorted(ver["record_hashes_full"])
    spec = json.loads(ORIG_MANIFEST.read_text())
    caps = load_captures(str(ORIG_CAP), spec, None, HORIZON)
    by = {c["category"]: c for c in caps}
    embed = load_embed()

    xm_a, xa_a, lab_a = build_inram(cats, by, embed)
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        xm_b, xa_b, lab_b = build_mmap(cats, by, embed, cache)

        checks = {}
        checks["raw_xm"] = torch.equal(xm_a, xm_b)
        checks["raw_xa"] = torch.equal(xa_a, xa_b)
        checks["row_order_y"] = ([m["y"] for m in lab_a]
                                 == [m["y"] for m in lab_b])
        checks["row_order_cat"] = ([m["cat"] for m in lab_a]
                                   == [m["cat"] for m in lab_b])

        xmn_a, xan_a, sa = norm_inram(xm_a, xa_a)
        xmn_b, xan_b, sb = norm_mmap(xm_b, xa_b, cache)
        checks["xm_mean"] = torch.equal(sa[0], sb[0])
        checks["xm_std"] = torch.equal(sa[1], sb[1])
        checks["xa_mean"] = torch.equal(sa[2], sb[2])
        checks["xa_std"] = torch.equal(sa[3], sb[3])
        checks["norm_xm"] = torch.equal(xmn_a, xmn_b)
        checks["norm_xa"] = torch.equal(xan_a, xan_b)

    print(f"rows: {xm_a.shape[0]}  DM={xm_a.shape[1]} DA={xa_a.shape[1]}")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("EQUIVALENCE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
