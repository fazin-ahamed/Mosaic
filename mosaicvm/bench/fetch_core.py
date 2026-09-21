"""Phase 2.11A: acquire exact core tensors locally via HTTP range requests.

The checkpoint (167 GB, 48 shards) interleaves ~35 core tensors among ~1570
expert tensors per shard, so whole-shard download is impossible. safetensors
headers are fetched per shard (~300 KB each), core-tensor byte spans are
resolved and range-fetched (~6 GB total), and the tensors are reassembled
into one local safetensors file with a full manifest (revision, shards,
ranges, dtypes, shapes, hashes). No Modal budget required.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
from collections import defaultdict
from pathlib import Path

import requests
from huggingface_hub import HfApi

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
OUT_DIR = Path(".mosaicvm/core")
OUT_SAFETENSORS = OUT_DIR / "core.safetensors"
OUT_MANIFEST = OUT_DIR / "manifest.json"
SESSION = requests.Session()
MERGE_GAP = 1 << 20  # merge fetched spans closer than 1 MB


def _revision() -> str:
    return HfApi().model_info(REPO).sha


def _url(fname: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/main/{fname}"


def _get_range(fname: str, start: int, end: int) -> bytes:
    """bytes [start, end] inclusive."""
    r = SESSION.get(_url(fname), headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    r.raise_for_status()
    data = r.content
    if len(data) != end - start + 1:
        raise RuntimeError(f"short range read {fname} {start}-{end}: got {len(data)}")
    return data


def _shard_header(fname: str) -> dict:
    n = struct.unpack("<Q", _get_range(fname, 0, 7))[0]
    # fetch header JSON (allow one extension if the first chunk was short)
    hdr = _get_range(fname, 8, 8 + n - 1)
    if len(hdr) != n:
        hdr = _get_range(fname, 8, 8 + n - 1)
    return json.loads(hdr)


# NOTE: safetensors requires contiguous tensor payloads (no gaps);
# 8-byte alignment padding between tensors breaks the Rust validator.
def run(max_shards: int | None = None, include_mtp: bool = True) -> dict:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rev = _revision()

    idx = json.loads(SESSION.get(_url("model.safetensors.index.json"), timeout=60).content)
    wm = idx["weight_map"]
    core_names = {k: v for k, v in wm.items() if ".experts." not in k}
    if not include_mtp:
        core_names = {k: v for k, v in core_names.items() if not k.startswith("mtp.")}
    per_shard = defaultdict(list)
    for name, shard in core_names.items():
        per_shard[shard].append(name)
    shards = sorted(per_shard)
    if max_shards:
        shards = shards[:max_shards]

    tensors: dict[str, dict] = {}      # name -> {dtype, shape, data(bytes)}
    shard_log = []
    header_bytes = data_bytes = 0
    for si, fname in enumerate(shards):
        hdr = _shard_header(fname)
        header_bytes += 8 + len(json.dumps({})) + 0  # approx below
        entries = {k: v for k, v in hdr.items() if k != "__metadata__"}
        wanted = per_shard[fname]
        missing = [n for n in wanted if n not in entries]
        if missing:
            raise RuntimeError(f"{fname}: core tensors missing from header: {missing[:5]}")
        # merge wanted spans
        spans = sorted(
            (entries[n]["data_offsets"][0], entries[n]["data_offsets"][1], n)
            for n in wanted
        )
        merged: list[list] = []
        for s, e, n in spans:
            if merged and s - merged[-1][1] <= MERGE_GAP:
                merged[-1][1] = max(merged[-1][1], e)
                merged[-1][2].append(n)
            else:
                merged.append([s, e, [n]])
        fetched = 0
        for s, e, names in merged:
            blob = _get_range(fname, s, e - 1) if e > s else b""
            fetched += len(blob)
            for n in names:
                off0, off1 = entries[n]["data_offsets"]
                tensors[n] = {
                    "dtype": entries[n]["dtype"],
                    "shape": entries[n]["shape"],
                    "data": blob[off0 - s: off1 - s],
                }
        data_bytes += fetched
        shard_log.append({
            "shard": fname, "n_core_tensors": len(wanted),
            "spans": len(merged), "bytes_fetched": fetched,
        })
        done = si + 1
        print(f"[{done}/{len(shards)}] {fname}: {len(wanted)} tensors, "
              f"{fetched/1e6:.1f} MB (total {data_bytes/1e9:.2f} GB)", flush=True)

    # assemble safetensors: 8-byte header len + JSON header + data
    order = sorted(tensors)
    header = {}
    offset = 0
    for n in order:
        t = tensors[n]
        nbytes = len(t["data"])
        header[n] = {
            "dtype": t["dtype"],
            "shape": t["shape"],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(OUT_SAFETENSORS, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for n in order:
            f.write(tensors[n]["data"])

    sha = hashlib.sha256()
    with open(OUT_SAFETENSORS, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            sha.update(chunk)

    manifest = {
        "phase": "2.11A",
        "repo": REPO,
        "revision": rev,
        "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": ("HTTP range fetch of safetensors headers + core-tensor byte "
                   "spans; experts excluded by name filter"),
        "selection_rule": "all tensors without '.experts.' in name"
                          + ("" if include_mtp else " and not starting with 'mtp.'"),
        "n_tensors": len(order),
        "n_shards_touched": len(shards),
        "shards": shard_log,
        "bytes_core": sum(len(t["data"]) for t in tensors.values()),
        "out_file": str(OUT_SAFETENSORS),
        "out_sha256": sha.hexdigest(),
        "out_bytes": OUT_SAFETENSORS.stat().st_size,
        "dtypes": sorted({t["dtype"] for t in tensors.values()}),
        "duration_s": time.time() - t0,
        "elapsed_note": "GPU untouched; network-only acquisition",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("n_tensors", "n_shards_touched", "bytes_core",
                       "out_bytes", "out_sha256", "duration_s", "dtypes")}, indent=2))
    return manifest


if __name__ == "__main__":
    import sys
    max_shards = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(max_shards=max_shards)
