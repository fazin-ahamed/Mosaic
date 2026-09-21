"""Phase 2.12W-B: exact expert entropy census.

Fetches a representative sample of routed expert tensors via HTTP range
requests (no full checkpoint download), measures E2M1 nibble entropy,
duplicate-tile rate, XOR-delta entropy, and lossless compression ratios.
Determines whether ~3x lossless byte reduction is information-theoretically
plausible.
"""
from __future__ import annotations

import json
import math
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
ARTIFACT = Path("artifacts/expert_entropy.json")
SESSION = requests.Session()
_HEADER_CACHE: dict[str, dict] = {}

# sample: layers spread across the model, experts spread across 256
SAMPLE_LAYERS = [0, 20, 42]
SAMPLE_EXPERTS_PER_LAYER = [0, 128, 255]  # 3 per layer = 9 experts
ARTIFACT_DIR = Path(".mosaicvm/expert_sample")


def _url(fname: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/main/{fname}"


def _get_range(fname: str, start: int, end: int) -> bytes:
    for attempt in range(3):
        try:
            r = SESSION.get(_url(fname), headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def _shard_header(fname: str) -> dict:
    if fname in _HEADER_CACHE:
        return _HEADER_CACHE[fname]
    n = struct.unpack("<Q", _get_range(fname, 0, 7))[0]
    hdr = json.loads(_get_range(fname, 8, 8 + n - 1))
    _HEADER_CACHE[fname] = hdr
    return hdr


def _nibble_entropy(packed: bytes) -> dict:
    """E2M1 nibble histogram + Shannon entropy."""
    arr = np.frombuffer(packed, dtype=np.uint8)
    lo = arr & 0x0F
    hi = arr >> 4
    all_nibbles = np.concatenate([lo, hi])
    counts = np.bincount(all_nibbles, minlength=16)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    h = -np.sum(probs * np.log2(probs))
    return {
        "nibble_histogram": counts.tolist(),
        "entropy_bits_per_nibble": float(h),
        "max_entropy": 4.0,
        "entropy_ratio": float(h / 4.0),
        "n_bytes": len(packed),
        "n_nibbles": len(all_nibbles),
    }


def _duplicate_tile_rate(packed: bytes, tile_size: int = 32) -> float:
    """Fraction of tiles that are exact duplicates of an earlier tile."""
    arr = np.frombuffer(packed, dtype=np.uint8)
    n_tiles = len(arr) // tile_size
    tiles = arr[:n_tiles * tile_size].reshape(n_tiles, tile_size)
    seen = set()
    dups = 0
    for i in range(n_tiles):
        key = tiles[i].tobytes()
        if key in seen:
            dups += 1
        else:
            seen.add(key)
    return dups / n_tiles if n_tiles else 0.0


def _xor_delta_entropy(packed: bytes) -> dict:
    """Entropy of consecutive-byte XOR (captures local redundancy)."""
    arr = np.frombuffer(packed, dtype=np.uint8)
    if len(arr) < 2:
        return {"entropy_bits_per_byte": 8.0, "ratio": 1.0}
    delta = arr[1:] ^ arr[:-1]
    counts = np.bincount(delta, minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    h = -np.sum(probs * np.log2(probs))
    return {
        "xor_delta_entropy_bits_per_byte": float(h),
        "original_entropy_bits_per_byte": 8.0,
        "xor_ratio": float(h / 8.0),
    }


def _try_zstd(packed: bytes) -> dict:
    try:
        import zstandard
        c = zstandard.ZstdCompressor(level=19)
        compressed = c.compress(packed)
        return {"ratio": len(packed) / len(compressed) if compressed else 0.0,
                "method": "zstd-19"}
    except ImportError:
        pass
    try:
        import zlib
        compressed = zlib.compress(packed, 9)
        return {"ratio": len(packed) / len(compressed) if compressed else 0.0,
                "method": "zlib-9"}
    except Exception:
        return {"ratio": None, "method": "unavailable"}


def run() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    idx = json.loads(SESSION.get(_url("model.safetensors.index.json"), timeout=60).content)
    wm = idx["weight_map"]

    t0 = time.time()
    results = []
    for layer in SAMPLE_LAYERS:
        for expert in SAMPLE_EXPERTS_PER_LAYER:
            # w1=gate, w2=down, w3=up (per streaming_model contract)
            for proj in ("w1", "w2", "w3"):
                names = [
                    f"layers.{layer}.ffn.experts.{expert}.{proj}.weight",
                    f"layers.{layer}.ffn.experts.{proj}.{expert}.weight",
                ]
                name = next((n for n in names if n in wm), None)
                if name is None:
                    continue
                shard = wm[name]
                hdr = _shard_header(shard)
                if name not in hdr or name == "__metadata__":
                    continue
                off0, off1 = hdr[name]["data_offsets"]
                # also try to get the scale tensor
                scale_names = [
                    f"layers.{layer}.ffn.experts.{expert}.{proj}.scale",
                    f"layers.{layer}.ffn.experts.{proj}.{expert}.scale",
                ]
                scale_name = next((n for n in scale_names if n in hdr), None)
                if scale_name:
                    s0, s1 = hdr[scale_name]["data_offsets"]
                    blob = _get_range(shard, min(off0, s0), max(off1, s1) - 1)
                    weights = blob[off0 - min(off0, s0): off1 - min(off0, s0)]
                    scales = blob[s0 - min(off0, s0): s1 - min(off0, s0)]
                else:
                    weights = _get_range(shard, off0, off1 - 1)
                    scales = b""

                ent = _nibble_entropy(weights)
                dup_rate = _duplicate_tile_rate(weights)
                xor = _xor_delta_entropy(weights)
                comp = _try_zstd(weights)
                scale_ent = _nibble_entropy(scales) if scales else None

                results.append({
                    "layer": layer, "expert": expert, "proj": proj,
                    "name": name, "shard": shard,
                    "shape": hdr[name]["shape"], "dtype": hdr[name]["dtype"],
                    "weight_bytes": len(weights), "scale_bytes": len(scales),
                    "nibble_entropy": ent,
                    "duplicate_tile_rate": dup_rate,
                    "xor_delta": xor,
                    "compression": comp,
                    "scale_entropy": scale_ent,
                })
                print(f"L{layer} E{expert} {proj}: {len(weights)/1e6:.1f}MB "
                      f"H={ent['entropy_bits_per_nibble']:.3f}/4 "
                      f"dup={dup_rate:.4f} "
                      f"zstd={comp['ratio']:.2f}x" if comp.get("ratio") else
                      f"L{layer} E{expert} {proj}: {len(weights)/1e6:.1f}MB "
                      f"H={ent['entropy_bits_per_nibble']:.3f}/4 "
                      f"dup={dup_rate:.4f}", flush=True)

    # aggregate
    entropies = [r["nibble_entropy"]["entropy_bits_per_nibble"] for r in results]
    dup_rates = [r["duplicate_tile_rate"] for r in results]
    comp_ratios = [r["compression"]["ratio"] for r in results if r["compression"].get("ratio")]

    mean_h = float(np.mean(entropies)) if entropies else None
    mean_dup = float(np.mean(dup_rates)) if dup_rates else None
    mean_comp = float(np.mean(comp_ratios)) if comp_ratios else None
    max_comp = float(np.max(comp_ratios)) if comp_ratios else None

    # theoretical: if entropy is H bits/nibble, Huffman/rANS achieves ~H/4 ratio
    theoretical_ratio = (4.0 / mean_h) if mean_h else None

    if mean_comp is not None:
        if mean_comp < 1.15:
            classification = "COMPRESSION_WEAK"
        elif mean_comp < 1.5:
            classification = "ASSISTIVE_ONLY"
        elif mean_comp < 2.0:
            classification = "USEFUL_IN_COMBINATION"
        elif mean_comp < 3.0:
            classification = "STRONG"
        else:
            classification = "POTENTIAL_STANDALONE_BREAKTHROUGH"
    else:
        classification = "INCOMPLETE"

    three_x_plausible = bool(theoretical_ratio and theoretical_ratio >= 2.5)

    out = {
        "phase": "2.12W-B",
        "n_experts_sampled": len({(r["layer"], r["expert"]) for r in results}),
        "n_tensors": len(results),
        "layers_sampled": SAMPLE_LAYERS,
        "experts_per_layer": SAMPLE_EXPERTS_PER_LAYER,
        "aggregate": {
            "mean_nibble_entropy_bits": mean_h,
            "theoretical_compression_ratio": theoretical_ratio,
            "mean_measured_compression_ratio": mean_comp,
            "max_measured_compression_ratio": max_comp,
            "mean_duplicate_tile_rate": mean_dup,
            "classification": classification,
            "three_x_plausible": three_x_plausible,
            "note": f"Mean E2M1 nibble entropy {mean_h:.3f}/4 bits → theoretical "
                    f"entropy-coding limit {theoretical_ratio:.2f}x. Measured "
                    f"zstd ratio {mean_comp:.2f}x. The 3x bandwidth target "
                    f"requires ~3.0x; {'PLAUSIBLE' if three_x_plausible else 'NOT PLAUSIBLE'} "
                    f"from entropy alone.",
        },
        "results": results,
        "duration_s": time.time() - t0,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["aggregate"], indent=2, default=str))
    return out


if __name__ == "__main__":
    run()
