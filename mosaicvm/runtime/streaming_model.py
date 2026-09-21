"""Checkpoint-backed streaming V4 forward (Phase 1 correctness).

Resident set = embeddings, lm_head, per-layer attention/router/shared-expert/HC
+ currently selected routed experts (LRU). Routed expert tensors are loaded
from safetensors shards on demand and charged against the simulated VRAM budget.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from ..model.config import V4Config
from .exact import ExactContractError, exact_target
from safetensors import safe_open as _safe_open
from ..model.layers import (
    Attention,
    DenseMoELayer,
    HashMoELayer,
    RMSNorm,
    V4Block,
    grouped_expert_forward,
)
from .fp8 import dequant_fp8, fit_linear_weight
from .loader import StreamingLoader
from .memory_budget import budget


def validate_exact_v4_tensors(loader: "StreamingLoader", prefix: str, layer_idx: int) -> None:
    """Exact contract part 2: actual expert tensor layout on disk.

    Verifies FP4 E2M1 packed [N, K/2] uint8 expert weights, [N, K/32] E8M0 scales,
    ``tid2eid`` present on hash layers (0..num_hash_layers-1), bias absent on hash
    layers and present on normal routed layers. Raises ExactContractError otherwise.
    """
    idx = loader.shards.shards
    hidden, inter = 4096, 2048
    expected = {
        "w1": (inter, hidden),
        "w3": (inter, hidden),
        "w2": (hidden, inter),
    }
    problems = []

    def _meta(name):
        with _safe_open(loader.root / idx[name], framework="pt") as f:
            sl = f.get_slice(name)
            return sl.get_dtype(), sl.get_shape()

    def _is_packed_int8(dt) -> bool:
        # safetensors reports the packed FP4 container as I8 (int8) or
        # U8 (uint8); bit patterns are identical under view(torch.uint8)
        s = dt if isinstance(dt, str) else str(dt)
        return s in ("I8", "U8", "torch.int8", "torch.uint8")

    def _is_e8m0(dt) -> bool:
        s = dt if isinstance(dt, str) else str(dt)
        return s in ("F8_E8M0", "torch.float8_e8m0fnu")

    # per-expert V3-style layout: layers.N.ffn.experts.<i>.<w>.weight
    n_experts = 0
    while f"{prefix}ffn.experts.{n_experts}.w1.weight" in idx:
        n_experts += 1
    if n_experts == 0:
        problems.append(f"no per-expert tensors under {prefix}ffn.experts.*")
    for w, (n, k) in expected.items():
        wname = f"{prefix}ffn.experts.0.{w}.weight"
        sname = f"{prefix}ffn.experts.0.{w}.scale"
        if wname not in idx:
            problems.append(f"missing {wname}")
            continue
        dt, shape = _meta(wname)
        if not _is_packed_int8(dt):
            problems.append(f"{wname}: expected packed uint8 (FP4 E2M1), got {dt}")
        if tuple(shape) != (n, k // 2):
            problems.append(f"{wname}: expected shape ({n}, {k // 2}), got {tuple(shape)}")
        if sname not in idx:
            problems.append(f"missing {sname}")
        else:
            sdt, sshape = _meta(sname)
            if not _is_e8m0(sdt):
                problems.append(f"{sname}: expected E8M0, got {sdt}")
            if tuple(sshape) != (n, k // 32):
                problems.append(f"{sname}: expected shape ({n}, {k // 32}), got {tuple(sshape)}")

    is_hash = layer_idx < loader.cfg.num_hash_layers
    tid_name = f"{prefix}ffn.gate.tid2eid"
    bias_name = f"{prefix}ffn.gate.bias"
    if is_hash:
        if tid_name not in idx:
            problems.append(f"missing {tid_name}")
        if bias_name in idx:
            problems.append(f"hash layer must not have {bias_name}")
    else:
        if bias_name not in idx:
            problems.append(f"routed layer missing {bias_name}")

    if problems:
        raise ExactContractError(f"exact V4 tensor contract violated at layer {layer_idx}: " + "; ".join(problems))


def _maybe_dequant(loader: StreamingLoader, name: str, device: str) -> torch.Tensor:
    t = loader.load_tensor(name, device="cpu")
    scale_name = name.replace(".weight", ".scale")
    if scale_name in loader.shards.shards and scale_name != name:
        scale = loader.load_tensor(scale_name, device="cpu")
        t = dequant_fp8(t, scale)
    else:
        t = t.to(torch.bfloat16)
    return t.to(device)


def load_streaming_model(model_root: str, device: str = "cuda"):
    """Construct V4Model and copy core checkpoint tensors into it.

    Routed expert stacks (w_gate/w_up/w_down) stay empty; MoE layers are
    patched to pull selected experts through the LRU at forward time.
    """
    from ..model.model import V4Model

    loader = StreamingLoader(model_root)
    cfg = loader.cfg
    b = budget()
    b.reset()
    model = V4Model(cfg)
    # embeddings / head / norms
    mapping_root = {
        "embed_tokens.weight": "embed.weight",
        "lm_head.weight": "head.weight",
        "norm.weight": "norm.weight",
        "hc_head_fn": "hc_head_fn",
        "hc_head_base": "hc_head_base",
        "hc_head_scale": "hc_head_scale",
    }
    missing = []
    with torch.no_grad():
        for dst, src in mapping_root.items():
            if src not in loader.shards.shards:
                missing.append(src)
                continue
            t = _maybe_dequant(loader, src, "cpu")
            param = dict(model.named_parameters())[dst]
            if param.shape != t.shape:
                missing.append(f"{src} shape {tuple(t.shape)} != {tuple(param.shape)}")
                continue
            param.copy_(t)
            b.charge_device(param, dst)

        for i, layer in enumerate(model.layers):
            prefix = f"layers.{i}."
            pairs = [
                (layer.attn_norm.weight, prefix + "attn_norm.weight"),
                (layer.ffn_norm.weight, prefix + "ffn_norm.weight"),
                (layer.attn.q_a.weight, prefix + "attn.wq_a.weight"),
                (layer.attn.q_norm.weight, prefix + "attn.q_norm.weight"),
                (layer.attn.q_b.weight, prefix + "attn.wq_b.weight"),
                (layer.attn.kv_proj.weight, prefix + "attn.wkv.weight"),
                (layer.attn.kv_norm.weight, prefix + "attn.kv_norm.weight"),
                (layer.attn.o_lora_a.weight, prefix + "attn.wo_a.weight"),
                (layer.attn.o_lora_b.weight, prefix + "attn.wo_b.weight"),
                (layer.attn.attn_sink, prefix + "attn.attn_sink"),
                (layer.hc_attn.fn, prefix + "hc_attn_fn"),
                (layer.hc_attn.base, prefix + "hc_attn_base"),
                (layer.hc_attn.scale, prefix + "hc_attn_scale"),
                (layer.hc_ffn.fn, prefix + "hc_ffn_fn"),
                (layer.hc_ffn.base, prefix + "hc_ffn_base"),
                (layer.hc_ffn.scale, prefix + "hc_ffn_scale"),
            ]
            if exact_target():
                validate_exact_v4_tensors(loader, prefix, i)
            if isinstance(layer.moe, HashMoELayer):
                # hash layers: gate weight only (no bias); tid2eid bound below
                pairs.append((layer.moe.hash_router.weight, prefix + "ffn.gate.weight"))
            else:
                pairs.append((layer.moe.router.weight, prefix + "ffn.gate.weight"))
                pairs.append((layer.moe.router.e_score_correction_bias, prefix + "ffn.gate.bias"))
            pairs += [
                (layer.moe.shared_expert.gate_proj.weight, prefix + "ffn.shared_experts.w1.weight"),
                (layer.moe.shared_expert.up_proj.weight, prefix + "ffn.shared_experts.w3.weight"),
                (layer.moe.shared_expert.down_proj.weight, prefix + "ffn.shared_experts.w2.weight"),
            ]
            for param, src in pairs:
                if src not in loader.shards.shards:
                    missing.append(src)
                    continue
                t = _maybe_dequant(loader, src, "cpu")
                if param.shape != t.shape:
                    missing.append(f"{src} shape {tuple(t.shape)} != {tuple(param.shape)}")
                    continue
                param.copy_(t)
                try:
                    b.charge_host(param, src)
                except Exception:
                    missing.append(f"ram_overflow:{src}")

            tid_name = prefix + "ffn.gate.tid2eid"
            if tid_name in loader.shards.shards and hasattr(layer.moe, "hash_router"):
                table = loader.load_tensor(tid_name, device="cpu").long()
                layer.moe.hash_router.tid2eid = table
                b.charge_host(table, tid_name)

            if hasattr(layer.moe, "w_gate"):
                layer.moe.w_gate = None
                layer.moe.w_up = None
                layer.moe.w_down = None
            layer.moe.loader = loader
            layer.moe.layer_idx = i
            layer.moe.forward = _make_streaming_moe_forward(layer.moe, loader, i, cfg)

        # Peak VRAM = embeddings + lm_head + one transformer layer (layer-streamed).
        # Full 43-layer core does not fit the 10.5 GB simulated cap (trips ~layer 13).
        peak_layer = sum(p.numel() * p.element_size() for p in model.layers[0].parameters())
        try:
            b.charge_device(peak_layer, "peak_resident_layer")
        except Exception as e:
            missing.append(f"peak_layer_budget: {e}")

        def _wrap_layer(layer):
            orig = layer.forward

            def _fwd(*args, **kwargs):
                layer.to(device)
                try:
                    return orig(*args, **kwargs)
                finally:
                    layer.to("cpu")

            layer.forward = _fwd

        if os.environ.get("MOSAICVM_RESIDENT_CORE") != "1":
            for layer in model.layers:
                _wrap_layer(layer)



    model.embed_tokens.to(device)
    model.lm_head.to(device)
    model.norm.to(device)
    model.hc_head_fn.data = model.hc_head_fn.data.to(device)
    model.hc_head_base.data = model.hc_head_base.data.to(device)
    model.hc_head_scale.data = model.hc_head_scale.data.to(device)
    model.eval()
    return model, loader, missing


def _make_streaming_moe_forward(moe, loader: StreamingLoader, layer_idx: int, cfg: V4Config):
    def forward(x, token_ids=None):
        bsz, t, _ = x.shape
        xf = x.view(-1, x.shape[-1])
        if isinstance(moe, HashMoELayer):
            scores, idx = moe.hash_router(xf, token_ids)
        else:
            scores, idx = moe.router(xf)
        out_shared = moe.shared_expert(xf)
        # load selected unique experts
        unique = idx.unique().tolist()
        h, inter = cfg.hidden_size, cfg.moe_intermediate_size
        wg = torch.empty(len(unique), h, inter, dtype=torch.bfloat16, device=xf.device)
        wu = torch.empty_like(wg)
        wd = torch.empty(len(unique), inter, h, dtype=torch.bfloat16, device=xf.device)
        remap = {}
        for j, e in enumerate(unique):
            remap[e] = j
            prefix = f"layers.{layer_idx}.ffn.experts.{e}."
            # FP4 E2M1 packed + E8M0 block scale: LUT-unpack FIRST, then
            # apply the per-32 scale (the old _maybe_dequant+fit path
            # scaled packed bytes before unpacking — wrong values).
            from .fp8 import dequant_fp4
            w1 = dequant_fp4(
                loader.load_raw_tensor(prefix + "w1.weight"),
                loader.load_raw_tensor(prefix + "w1.scale"),
                inter, h)
            w3 = dequant_fp4(
                loader.load_raw_tensor(prefix + "w3.weight"),
                loader.load_raw_tensor(prefix + "w3.scale"),
                inter, h)
            w2 = dequant_fp4(
                loader.load_raw_tensor(prefix + "w2.weight"),
                loader.load_raw_tensor(prefix + "w2.scale"),
                h, inter)
            # checkpoint stores [out, in]; grouped_expert_forward wants
            # [in, out] (x @ W orientation)
            wg[j] = w1.t().to(xf.device)
            wu[j] = w3.t().to(xf.device)
            wd[j] = w2.t().to(xf.device)
        idx_local = idx.clone()
        for e, j in remap.items():
            idx_local[idx == e] = j
        routed = grouped_expert_forward(xf, scores, idx_local, (wg, wu), wd, len(unique))
        return (out_shared + routed).view(bsz, t, -1)

    return forward
