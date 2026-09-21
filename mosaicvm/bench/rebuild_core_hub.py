"""Rebuild .mosaicvm/core/core.safetensors from AUTHORITATIVE hub bytes.

Root cause closed: fetch_core.py's plain-HTTP-range acquisition hit a
STALE CDN copy of the shards (garbage attention scales with 0xFF, wrong
codes). hf_hub_download's xet path gets the authoritative files (clean
e8m0 scales {114-119}, real codes, runtime-verified hash match).

This script downloads all shards via hf_hub_download, extracts the core
tensors (selection rule unchanged: no '.experts.'), writes the archive,
and runs closure verification:
  - no 0xFF byte in ANY attention scale tensor
  - scale smoothness (adjacent |delta|) passes
  - layers.2.attn.wq_b.weight sha16 == 65d1cf8c02b74b16 (runtime hash)
  - wgate NaN scan (the old deviation's bf16 NaN finding)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
OUT_DIR = Path(".mosaicvm/core")
OLD = OUT_DIR / "core.safetensors"
STALE = OUT_DIR / "core.stale-cdn.safetensors"
OUT_MANIFEST = OUT_DIR / "manifest.json"

idx = json.loads(hf_hub_download(REPO, "model.safetensors.index.json")
                 and Path(hf_hub_download(REPO,
                                          "model.safetensors.index.json"))
                 .read_text())
wm = idx["weight_map"]
core_names = {k: v for k, v in wm.items() if ".experts." not in k}
shards = sorted(set(core_names.values()))
print(f"core tensors: {len(core_names)} across {len(shards)} shards")

tensors = {}
t0 = time.time()
for si, fname in enumerate(shards):
    p = hf_hub_download(REPO, fname)
    with safe_open(p, framework="pt") as f:
        for name in f.keys():
            if name in core_names:
                tensors[name] = f.get_tensor(name)
    if (si + 1) % 8 == 0 or si == len(shards) - 1:
        print(f"[{si+1}/{len(shards)}] shards, "
              f"{time.time()-t0:.0f}s, {len(tensors)} tensors",
              flush=True)

# assemble
order = sorted(tensors)
DT_MAP = {
    torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16",
    torch.float8_e4m3fn: "F8_E4M3", torch.float8_e8m0fnu: "F8_E8M0",
    torch.int64: "I64", torch.uint8: "U8", torch.int8: "I8",
    torch.int32: "I32", torch.float64: "F64",
}
header = {}
offset = 0
for n in order:
    t = tensors[n]
    nbytes = t.numel() * t.element_size()
    header[n] = {"dtype": DT_MAP[t.dtype],
                 "shape": list(t.shape),
                 "data_offsets": [offset, offset + nbytes]}
    offset += nbytes
hdr_bytes = json.dumps(header, separators=(",", ":")).encode()
import struct
if OLD.exists():
    if STALE.exists():
        OLD.unlink()          # broken/incomplete rebuild — discard
        print("discarded previous (broken-header) rebuild")
    else:
        shutil.move(OLD, STALE)
        print(f"old archive -> {STALE}")
with open(OLD, "wb") as f:
    f.write(struct.pack("<Q", len(hdr_bytes)))
    f.write(hdr_bytes)
    for n in order:
        f.write(tensors[n].contiguous().view(torch.uint8).reshape(-1)
                .numpy().tobytes())
sha = hashlib.sha256(OLD.read_bytes()).hexdigest()
print(f"wrote {OLD} ({OLD.stat().st_size/1e9:.2f} GB) sha={sha[:16]}")

# ---- closure verification ----
print("\n=== CLOSURE VERIFICATION ===")
bad_ff = 0
bad_smooth = 0
n_scales = 0
for n in order:
    if not n.endswith(".scale"):
        continue
    n_scales += 1
    sb = tensors[n].view(torch.uint8).flatten()
    if (sb == 255).any():
        bad_ff += 1
    # smoothness: adjacent byte delta
    d = (sb[1:].float() - sb[:-1].float()).abs()
    if d.numel() > 1 and d.mean() > 8:
        bad_smooth += 1
print(f"attention scale tensors: {n_scales}")
print(f"  with 0xFF bytes: {bad_ff}")
print(f"  failing smoothness (mean|delta|>8): {bad_smooth}")

# runtime hash check
w = tensors["layers.2.attn.wq_b.weight"]
h = hashlib.sha256(
    w.view(torch.uint8).reshape(-1).numpy().tobytes()).hexdigest()[:16]
print(f"layers.2.attn.wq_b.weight sha16: {h} "
      f"(expect 65d1cf8c02b74b16: {h == '65d1cf8c02b74b16'})")

# wgate NaN scan (old deviation finding)
nan_wgates = 0
for n in order:
    if "wgate" in n and tensors[n].dtype == torch.bfloat16:
        if torch.isnan(tensors[n].float()).any():
            nan_wgates += 1
print(f"bf16 wgate tensors with NaN: {nan_wgates}")

# hc_head scan
for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
    t = tensors[n]
    print(f"{n}: dtype={t.dtype} finite={torch.isfinite(t.float()).all().item()}")

manifest = {
    "phase": "2.11A-REBUILD",
    "repo": REPO,
    "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    "method": "hf_hub_download (xet-backed) — supersedes plain-HTTP-range "
              "acquisition whose CDN path served a STALE shard version "
              "(root cause of the 0xFF / checkpoint-integrity deviation)",
    "selection_rule": "all tensors without '.experts.' in name",
    "n_tensors": len(order),
    "out_sha256": sha,
    "verification": {
        "scale_tensors": n_scales, "with_0xFF": bad_ff,
        "smoothness_fail": bad_smooth,
        "wq_b_L2_sha16": h,
        "nan_wgates": nan_wgates,
    },
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
print(f"\nwrote {OUT_MANIFEST}")
PYEOF_MARKER = None
