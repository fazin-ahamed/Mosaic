"""Print per-tensor bf16 vs W4 bytes for the resident core. No snapshot needed."""
from __future__ import annotations

import json
from pathlib import Path

from mosaicvm.model.config import V4Config

GROUP = 128


def w4_bytes(numel: int) -> int:
    packed = numel // 2  # uint8, two values per byte
    scales = (numel // GROUP) * 2  # fp16
    return packed + scales


def bf16_bytes(numel: int) -> int:
    return numel * 2

def mxfp4_bytes(numel: int) -> int:
    return numel // 2 + numel // 32  # uint8 packed + e8m0 scale (1 byte)

def fp8_e4m3_block128_bytes(numel: int) -> int:
    return numel + (numel // 128) * 4  # e4m3 + fp32 block scale (FineGrainedFP8-style)


def inventory(cfg: V4Config | None = None) -> dict:
    cfg = cfg or V4Config()
    h = cfg.hidden_size
    layers = cfg.num_hidden_layers
    inner = cfg.num_attention_heads * cfg.head_dim
    group_head_dim = inner // cfg.o_groups
    mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
    hc_dim = cfg.hc_mult * h

    per_layer = {
        "attn.q_a": (h, cfg.q_lora_rank),
        "attn.q_b": (cfg.q_lora_rank, inner),
        "attn.kv_proj": (h, cfg.head_dim),
        "attn.o_lora_a": (group_head_dim, cfg.o_groups * cfg.o_lora_rank),
        "attn.o_lora_b": (cfg.o_groups * cfg.o_lora_rank, h),
        "shared.gate": (h, cfg.moe_intermediate_size),
        "shared.up": (h, cfg.moe_intermediate_size),
        "shared.down": (cfg.moe_intermediate_size, h),
    }
    # 1-D params stay bf16 (tiny)
    per_layer_vec = {
        "attn_sink": cfg.num_attention_heads,
        "attn_norm": h,
        "ffn_norm": h,
        "q_norm": cfg.q_lora_rank,
        "kv_norm": cfg.head_dim,
        "hc_attn.base": mix_hc,
        "hc_attn.scale": 3,
        "hc_ffn.base": mix_hc,
        "hc_ffn.scale": 3,
    }
    per_layer_hc = {
        "hc_attn.fn": mix_hc * hc_dim,
        "hc_ffn.fn": mix_hc * hc_dim,
    }
    router_w = cfg.n_routed_experts * h
    router_b = cfg.n_routed_experts

    rows = []
    w4_core = 0
    bf16_core = 0

    def add(name, numel, fmt, n_copies=1):
        nonlocal w4_core, bf16_core
        b16 = bf16_bytes(numel) * n_copies
        if fmt == "w4":
            b = w4_bytes(numel) * n_copies
            w4_core += b
        else:
            b = b16
            w4_core += b
        bf16_core += b16
        rows.append({"name": name, "numel": numel, "copies": n_copies, "fmt": fmt,
                     "bytes": b, "bf16_bytes": b16, "gb": b / 1024**3})

    for name, (k, n) in per_layer.items():
        add(f"layer.{name}", k * n, "w4", layers)
    for name, n in per_layer_vec.items():
        add(f"layer.{name}", n, "bf16", layers)
    for name, n in per_layer_hc.items():
        add(f"layer.{name}", n, "bf16", layers)
    add("router.weight", router_w, "bf16", layers)
    add("router.bias", router_b, "bf16", layers)
    add("lm_head", h * cfg.vocab_size, "w4", 1)
    add("norm", h, "bf16", 1)
    add("hc_head_fn", cfg.hc_mult * (cfg.hc_mult * h), "bf16", 1)
    add("hc_head_base", cfg.hc_mult, "bf16", 1)
    add("hc_head_scale", 1, "bf16", 1)

    embed = bf16_bytes(cfg.vocab_size * h)
    gpu_gb = w4_core / 1024**3
    out = {
        "rows": rows,
        "gpu_w4_core_plus_lm_head_plus_router_bytes": w4_core,
        "gpu_gb": gpu_gb,
        "bf16_core_gb": bf16_core / 1024**3,
        "embed_host_gb": embed / 1024**3,
        "under_6gb": gpu_gb < 6.0,
        "limit_gb": 6.0,
        # Phase 2.6: FP8 / MXFP4 expert inventory (does not load onto GPU this increment).
        "gpu_fp8_core_plus_lm_head_plus_router_gb": (
            w4_core - sum(w4_bytes(k * n) * layers for k, n in per_layer.values())
            + sum(fp8_e4m3_block128_bytes(k * n) * layers for k, n in per_layer.values())
        ) / 1024**3,
        "gpu_mxfp4_experts_one_layer_gb": 256 * mxfp4_bytes(3 * 4096 * 2048) / 1024**3,
    }
    print(f"W4 core+lm_head+router+HC = {gpu_gb:.3f} GB  (limit 6.0, pass={gpu_gb < 6.0})")
    print(f"embed host = {embed / 1024**3:.3f} GB (pinned RAM, not GPU)")
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/core_inventory.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    inv = inventory()
    assert inv["under_6gb"], f"W4 core {inv['gpu_gb']:.3f} GB exceeds 6.0 GB"
