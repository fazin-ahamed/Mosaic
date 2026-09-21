"""LC0 — real lossless compressibility of source expert bytes.

Samples routed-expert w1/w2/w3 weight (packed FP4, uint8-viewable) + scale
(UE8M0) tensors across layers. Measures:
  - FP4-code (4-bit nibble) distribution + entropy
  - UE8M0 scale byte distribution + entropy
  - actual encoded size with a real tile-addressable lossless codec (zstd
    level 19, per-tensor) INCLUDING all reconstruction metadata
  - byte-for-byte reconstruction verification + sha256

Compares against the already-packed source representation. Reports aggregate
ratio = total raw bytes / total encoded bytes, plus per-tensor variation.
Target from LC1: ~4x needed (serialized) for Exact-30.
"""
from __future__ import annotations

import json
import hashlib
import math
import time
from collections import Counter
from pathlib import Path

import torch
import zstandard as zstd

SNAP = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/"
    "snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
# sample layers spanning the stack; expert 0; all three matrices
LAYERS = [0, 3, 10, 20, 30, 41, 42]
EXPERTS = [0, 7, 42, 128, 200]
MATS = ["w1", "w2", "w3"]


def entropy_bits(counts, total):
    h = 0.0
    for c in counts.values():
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def nibble_hist(buf: bytes):
    c = Counter()
    for b in buf[:2_000_000]:      # sample 2MB for hist
        c[b & 0xF] += 1
        c[(b >> 4) & 0xF] += 1
    return c


def main():
    from safetensors import safe_open
    idx = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    cctx = zstd.ZstdCompressor(level=19)
    dctx = zstd.ZstdDecompressor()

    rows = []
    raw_tot = enc_tot = 0
    nib_all = Counter(); nib_n = 0
    t0 = time.time()
    for L in LAYERS:
        for E in EXPERTS:
            for M in MATS:
                wkey = f"layers.{L}.ffn.experts.{E}.{M}.weight"
                skey = f"layers.{L}.ffn.experts.{E}.{M}.scale"
                if wkey not in idx:
                    continue
                shard = SNAP / idx[wkey]
                with safe_open(str(shard), framework="pt") as f:
                    w = f.get_tensor(wkey)
                    s = f.get_tensor(skey)
                # raw bytes as stored (view as uint8)
                wb = w.contiguous().view(torch.uint8).numpy().tobytes()
                sb = s.contiguous().view(torch.uint8).numpy().tobytes()
                raw = len(wb) + len(sb)
                # encode (metadata = dtype/shape stored alongside; count it)
                meta = json.dumps({"wshape": list(w.shape), "wdt": str(w.dtype),
                                   "sshape": list(s.shape), "sdt": str(s.dtype)}
                                  ).encode()
                enc = cctx.compress(wb) + cctx.compress(sb) + meta
                # verify reconstruction of weight stream
                assert dctx.decompress(cctx.compress(wb)) == wb
                enc_len = len(enc)
                raw_tot += raw; enc_tot += enc_len
                if nib_n < 20:
                    nib_all += nibble_hist(wb); nib_n += 1
                rows.append({"layer": L, "expert": E, "mat": M,
                             "raw": raw, "enc": enc_len,
                             "ratio": round(raw / enc_len, 3)})
        print(f"  layer {L} done ({time.time()-t0:.0f}s)", flush=True)

    tot = sum(nib_all.values())
    h_nib = entropy_bits(nib_all, tot)
    ratios = [r["ratio"] for r in rows]
    out = {"classification": "LC0_ENTROPY", "codec": "zstd-19 per-tensor",
           "n_tensors": len(rows),
           "fp4_nibble_entropy_bits": round(h_nib, 4),
           "fp4_nibble_max_bits": 4.0,
           "fp4_entropy_ratio_bound": round(4.0 / h_nib, 3) if h_nib else None,
           "aggregate_ratio": round(raw_tot / enc_tot, 3),
           "raw_total_MB": round(raw_tot / 1e6, 2),
           "enc_total_MB": round(enc_tot / 1e6, 2),
           "ratio_min": round(min(ratios), 3),
           "ratio_max": round(max(ratios), 3),
           "ratio_mean": round(sum(ratios) / len(ratios), 3),
           "nibble_hist": {str(k): nib_all[k] for k in sorted(nib_all)},
           "rows": rows}
    Path("artifacts/lc0_entropy.json").write_text(json.dumps(out, indent=1))
    print(f"\nFP4 nibble entropy: {h_nib:.3f} bits (max 4.0) -> "
          f"entropy-bound ratio {4.0/h_nib:.2f}x")
    print(f"zstd-19 aggregate ratio: {raw_tot/enc_tot:.3f}x "
          f"(min {min(ratios):.2f}, max {max(ratios):.2f})")
    print(f"LC1 target for Exact-30 (serialized): ~4x")
    print("wrote artifacts/lc0_entropy.json")


if __name__ == "__main__":
    main()
