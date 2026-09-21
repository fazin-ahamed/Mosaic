"""Resident compressed core: W4 attention/shared/lm_head on GPU, embeddings in RAM.

Routed expert stacks are not loaded (still streamed/LRU; LBEF is later).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from ..kernels.w4a16_gemm import pack_w4, w4a16_gemm
from ..model.config import V4Config
from ..model.model import V4Model
from .memory_budget import BudgetExceeded, budget


def _pack_linear(linear: nn.Linear, tag: str, b, device: str) -> None:
    w = linear.weight.detach().to(device=device, dtype=torch.bfloat16)
    packed, scale = pack_w4(w.t().contiguous())
    b.charge_device(packed, tag + ".packed")
    b.charge_device(scale, tag + ".scale")
    linear.register_buffer("_w4_packed", packed)
    linear.register_buffer("_w4_scale", scale)
    del linear.weight

    def _fwd(x, packed=packed, scale=scale):
        orig = x.shape
        y = w4a16_gemm(x.reshape(-1, orig[-1]).contiguous(), packed, scale)
        return y.view(*orig[:-1], y.shape[-1])

    linear.forward = _fwd


def _pack_o_lora_a(attn, tag: str, b, device: str) -> None:
    """o_lora_a is consumed via grouped einsum, not Linear.forward."""
    cfg = attn.cfg
    w = attn.o_lora_a.weight.detach().to(device=device, dtype=torch.bfloat16)
    g = cfg.o_groups
    r = cfg.o_lora_rank
    d = w.shape[1]
    w_g = w.view(g, r, d)  # [g, r, d]; GEMM is x[d] @ W[d, r]
    packed_l, scale_l = [], []
    for gi in range(g):
        p, s = pack_w4(w_g[gi].t().contiguous())
        b.charge_device(p, f"{tag}.g{gi}.packed")
        b.charge_device(s, f"{tag}.g{gi}.scale")
        packed_l.append(p)
        scale_l.append(s)
    attn.register_buffer("_w4_o_a_dummy", packed_l[0][:1, :1])  # keep module on device
    attn._w4_o_a_packed = packed_l
    attn._w4_o_a_scale = scale_l
    del attn.o_lora_a.weight

    orig_fwd = attn.forward

    def _fwd(x, kv_cache=None, _orig=orig_fwd):
        # run until o_lora_a by replacing the einsum path: wrap orig after q/kv
        return _attn_forward_w4(attn, x, kv_cache)

    attn.forward = _fwd


def _attn_forward_w4(self, x, kv_cache=None):
    import torch.nn.functional as F

    b, t, _ = x.shape
    h, hd = self.n_heads, self.head_dim
    q = self.q_b(self.q_norm(self.q_a(x))).view(b, t, h, hd)
    kv = self.kv_norm(self.kv_proj(x)).unsqueeze(2)
    if kv_cache is not None:
        if kv_cache[0] is not None and torch.is_tensor(kv_cache[0]) and kv_cache[0].numel() > 0:
            kv = torch.cat([kv_cache[0], kv], dim=1)
        kv_cache[0], kv_cache[1] = kv, kv
    k = kv
    v = kv
    scale = hd ** -0.5
    qh = q.transpose(1, 2)
    kh = k.expand(-1, -1, h, -1).permute(0, 2, 1, 3)
    vh = v.expand(-1, -1, h, -1).permute(0, 2, 1, 3)
    attn_out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=None, scale=scale)
    attn_out = attn_out.transpose(1, 2).reshape(b, t, h * hd)
    attn_out = attn_out.reshape(b, t, self.o_groups, -1)
    parts = []
    for gi in range(self.o_groups):
        xg = attn_out[:, :, gi].reshape(b * t, -1).contiguous()
        yg = w4a16_gemm(xg, self._w4_o_a_packed[gi], self._w4_o_a_scale[gi])
        parts.append(yg.view(b, t, -1))
    attn_out = torch.stack(parts, dim=2).flatten(2)
    return self.o_lora_b(attn_out)


def load_resident_core(model_root: str | None = None, device: str = "cuda"):
    """Instantiate V4Model with W4-packed core tensors charged through budget().

    If model_root is missing, uses V4Config defaults (shape check / inventory path).
    v4_forward_check stays on the streaming correctness path.
    """
    b = budget()
    b.reset()
    cfg = V4Config()
    if model_root and Path(model_root).joinpath("config.json").exists():
        cfg = V4Config.from_checkpoint(model_root)
    model = V4Model(cfg)
    model._device = torch.device(device if torch.cuda.is_available() else "cpu")
    dev = str(model._device)

    # embeddings stay host-pinned
    embed = model.embed_tokens.weight.detach().to(dtype=torch.bfloat16)
    model.embed_tokens.weight = nn.Parameter(embed, requires_grad=False)
    model.embed_tokens.to("cpu")
    b.charge_host(model.embed_tokens.weight, "embed_tokens.weight")

    w4_linears = []
    for i, layer in enumerate(model.layers):
        w4_linears += [
            (layer.attn.q_a, f"L{i}.q_a"),
            (layer.attn.q_b, f"L{i}.q_b"),
            (layer.attn.kv_proj, f"L{i}.kv_proj"),
            (layer.attn.o_lora_b, f"L{i}.o_lora_b"),
            (layer.moe.shared_expert.gate_proj, f"L{i}.shared.gate"),
            (layer.moe.shared_expert.up_proj, f"L{i}.shared.up"),
            (layer.moe.shared_expert.down_proj, f"L{i}.shared.down"),
        ]
        _pack_o_lora_a(layer.attn, f"L{i}.o_lora_a", b, dev)
        # router stays bf16
        layer.moe.router.to(dev)
        b.charge_device(layer.moe.router.weight, f"L{i}.router.weight")
        b.charge_device(layer.moe.router.e_score_correction_bias, f"L{i}.router.bias")
        for p, tag in [
            (layer.attn_norm.weight, f"L{i}.attn_norm"),
            (layer.ffn_norm.weight, f"L{i}.ffn_norm"),
            (layer.attn.q_norm.weight, f"L{i}.q_norm"),
            (layer.attn.kv_norm.weight, f"L{i}.kv_norm"),
            (layer.attn.attn_sink, f"L{i}.attn_sink"),
            (layer.hc_attn.fn, f"L{i}.hc_attn.fn"),
            (layer.hc_attn.base, f"L{i}.hc_attn.base"),
            (layer.hc_attn.scale, f"L{i}.hc_attn.scale"),
            (layer.hc_ffn.fn, f"L{i}.hc_ffn.fn"),
            (layer.hc_ffn.base, f"L{i}.hc_ffn.base"),
            (layer.hc_ffn.scale, f"L{i}.hc_ffn.scale"),
        ]:
            p.data = p.data.to(dev)
            b.charge_device(p, tag)
        layer.moe.w_gate = layer.moe.w_up = layer.moe.w_down = None

    for lin, tag in w4_linears:
        _pack_linear(lin, tag, b, dev)

    _pack_linear(model.lm_head, "lm_head", b, dev)
    model.norm.weight.data = model.norm.weight.data.to(dev)
    b.charge_device(model.norm.weight, "norm")
    for p, tag in [
        (model.hc_head_fn, "hc_head_fn"),
        (model.hc_head_base, "hc_head_base"),
        (model.hc_head_scale, "hc_head_scale"),
    ]:
        p.data = p.data.to(dev)
        b.charge_device(p, tag)

    model.eval()
    return model, b.peak_summary()


def resident_core_check(model_root: str | None = None, device: str = "cuda") -> dict:
    """Load resident core, report budget, 1-token random forward shape check."""
    try:
        model, peak = load_resident_core(model_root, device=device)
    except BudgetExceeded as e:
        b = budget()
        print("BudgetExceeded:", e)
        print("ledger:")
        for row in b.ledger:
            print(" ", row)
        return {"pass": False, "error": str(e), "ledger": list(b.ledger), "budget": b.peak_summary()}
    ids = torch.randint(0, model.cfg.vocab_size, (1, 1))
    with torch.no_grad():
        logits = model(ids)
    want = (1, 1, model.cfg.vocab_size)
    ok = tuple(logits.shape) == want
    out = {"pass": ok, "logits_shape": list(logits.shape), "budget": peak, "want": list(want)}
    print(out)
    return out


if __name__ == "__main__":
    # Full 43-layer construct is slow to pack; inventory is the size gate.
    # Shape check uses a 1-layer cfg override via env MOSAICVM_RESIDENT_LAYERS.
    n = int(os.environ.get("MOSAICVM_RESIDENT_LAYERS", "0"))
    if n:
        from mosaicvm.model.config import V4Config as C

        orig = C.__init__

        def _init(self, *a, **k):
            orig(self, *a, **k)
            self.num_hidden_layers = n

        C.__init__ = _init
    resident_core_check()
