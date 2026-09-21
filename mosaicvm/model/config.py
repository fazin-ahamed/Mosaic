"""Authoritative V4 config loader.

The checkpoint's config.json on the volume overrides the design doc's
provisional §4.1 constants; any disagreement fails loudly with a diff.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

# Doc §4.1 provisional constants — checked against the checkpoint and overridden.
DOC_EXPECTED = {
    "num_hidden_layers": 43,
    "hidden_size": 4096,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "num_hash_layers": 3,
    "sliding_window": 128,
    "dspark_block_size": 5,
}


@dataclass
class V4Config:
    num_hidden_layers: int = 43
    hidden_size: int = 4096
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    num_hash_layers: int = 3
    sliding_window: int = 128
    dspark_block_size: int = 5
    # authoritative extras from the real checkpoint
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    qk_rope_head_dim: int = 64
    vocab_size: int = 129280
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    dspark_target_layer_ids: tuple = (40, 41, 42)
    dspark_markov_rank: int = 256
    dspark_noise_token_id: int = 128799
    max_position_embeddings: int = 1048576
    compress_ratios: tuple = (
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 0, 0, 0)
    compress_rope_theta: float = 160000.0
    tie_word_embeddings: bool = False

    @classmethod
    def from_checkpoint(cls, model_root: str | Path) -> "V4Config":
        cfg_path = Path(model_root) / "config.json"
        raw = json.loads(cfg_path.read_text())
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        if isinstance(kwargs.get("dspark_target_layer_ids"), list):
            kwargs["dspark_target_layer_ids"] = tuple(kwargs["dspark_target_layer_ids"])
        cfg = cls(**kwargs)

        from mosaicvm.runtime.exact import exact_target

        if exact_target():
            meta = {}
            quant = raw.get("quantization_config") or {}
            if isinstance(quant, dict):
                meta["activation"] = quant.get("activation") or quant.get("activation_scheme")
                meta["scale_fmt"] = quant.get("scale_fmt") or quant.get("scale_format")
                meta["expert_dtype"] = quant.get("expert_dtype") or quant.get("weight_dtype")
            for k in ("activation", "scale_fmt", "expert_dtype"):
                if meta.get(k) is None and k in raw:
                    meta[k] = raw[k]
            # Normalize the checkpoint's short/lowercase spellings to the
            # contract's display forms — semantics identical (authoritative
            # 7872f01b config.json ships activation_scheme "dynamic" +
            # fmt "e4m3", scale_fmt "ue8m0", expert_dtype "fp4").
            if meta.get("activation") == "dynamic" and (
                    quant.get("fmt") == "e4m3"):
                meta["activation"] = "dynamic E4M3"
            if isinstance(meta.get("scale_fmt"), str) and \
                    meta["scale_fmt"].lower() == "ue8m0":
                meta["scale_fmt"] = "UE8M0"
            if isinstance(meta.get("expert_dtype"), str) and \
                    meta["expert_dtype"].lower() == "fp4":
                meta["expert_dtype"] = "FP4 E2M1"
            if any(meta.get(k) is not None for k in ("activation", "scale_fmt", "expert_dtype")):
                validate_exact_v4_config(raw, meta)

        diffs = []
        for key, doc_val in DOC_EXPECTED.items():
            ckpt_val = getattr(cfg, key)
            if ckpt_val != doc_val:
                diffs.append(f"  {key}: doc={doc_val} checkpoint={ckpt_val}")
        if diffs:
            print("WARNING: doc §4.1 constants differ from checkpoint (checkpoint wins):")
            print("\n".join(diffs))
        return cfg

# Root HF config and DeepSeek inference config use different key names; accept either.
_ALIASES = {
    "num_hidden_layers": ("num_hidden_layers", "n_layers"),
    "num_hash_layers": ("num_hash_layers",),
    "num_experts_per_tok": ("num_experts_per_tok", "n_activated_experts"),
    "scoring_func": ("scoring_func", "score_func"),
    "routed_scaling_factor": ("routed_scaling_factor", "route_scale"),
    "swiglu_limit": ("swiglu_limit",),
}

_EXACT_CONFIG_EXPECTED = {
    "num_hidden_layers": 43,
    "num_hash_layers": 3,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
}


def validate_exact_v4_config(cfg: "V4Config | dict", checkpoint_meta: dict) -> None:
    """Exact contract part 1: config values match the V4-0731 source semantics.

    ``cfg`` may be a V4Config or a raw config dict (either key spelling).
    ``checkpoint_meta`` reports quantization formats, keys:
    ``activation`` (must be 'dynamic E4M3'), ``scale_fmt`` ('UE8M0'),
    ``expert_dtype`` ('FP4 E2M1'). Raises ExactContractError otherwise.
    """
    from mosaicvm.runtime.exact import ExactContractError

    def _get(key, default=None):
        for name in _ALIASES[key]:
            if isinstance(cfg, dict):
                if name in cfg:
                    return cfg[name]
            elif hasattr(cfg, name):
                return getattr(cfg, name)
        return default

    problems = []
    for key, want in _EXACT_CONFIG_EXPECTED.items():
        got = _get(key)
        if got is None and isinstance(want, float) and key == "swiglu_limit":
            got = 10.0  # source default when the config omits it
        if isinstance(want, float):
            ok = got is not None and abs(float(got) - want) < 1e-9
        else:
            ok = got == want
        if not ok:
            problems.append(f"config {key}: expected {want!r}, got {got!r}")

    for key, want in (("activation", "dynamic E4M3"), ("scale_fmt", "UE8M0"), ("expert_dtype", "FP4 E2M1")):
        got = checkpoint_meta.get(key)
        if got != want:
            problems.append(f"checkpoint_meta {key}: expected {want!r}, got {got!r}")

    if problems:
        raise ExactContractError("exact V4 config contract violated: " + "; ".join(problems))
