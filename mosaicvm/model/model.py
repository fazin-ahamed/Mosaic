"""V4Model: full forward + greedy generate (Phase 1, correctness only).

Speculation (K<=5 byte-budgeted) deliberately absent — later phase.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import V4Config
from .layers import RMSNorm, V4Block


class V4Model(nn.Module):
    def __init__(self, cfg: V4Config, exact_dsa: bool = False):
        """exact_dsa=True swaps the eager attention stand-in for the source-
        exact DSA path (mosaicvm.model.dsa_layers.ExactAttentionAdapter).
        Prefill/chunk semantics only: decode against a KV history raises
        NotImplementedError until the decode path lands."""
        super().__init__()
        self.cfg = cfg
        self.exact_dsa = exact_dsa
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        hash_layers = set(range(cfg.num_hash_layers))
        self.layers = nn.ModuleList(
            V4Block(cfg, i, is_hash=(i in hash_layers), exact_dsa=exact_dsa)
            for i in range(cfg.num_hidden_layers)
        )
        mix_hc = cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_head_base = nn.Parameter(torch.zeros(mix_hc))
        self.hc_head_scale = nn.Parameter(torch.ones(1))
        nn.init.normal_(self.hc_head_fn, std=0.01)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def hc_head(self, x):
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        mixes = torch.nn.functional.linear(x_flat, self.hc_head_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float()) + self.cfg.hc_eps
        y = (pre.unsqueeze(-1) * x.float()).sum(dim=2)
        return y.to(x.dtype)

    def forward(self, input_ids, kv_caches=None):
        device = next(self.lm_head.parameters()).device
        x = self.embed_tokens(
            input_ids.to(self.embed_tokens.weight.device)).to(device)
        x = x.unsqueeze(2).expand(-1, -1, self.cfg.hc_mult, -1).contiguous()
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, token_ids=input_ids, kv_cache=cache)
        x = self.hc_head(x)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int):
        """Greedy loop: prefill then last-token decode. No speculation."""
        kv_caches = [[None, None] for _ in self.layers]
        ids = input_ids.clone()
        logits = self(ids, kv_caches)
        for _step in range(max_new_tokens):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            logits = self(nxt, kv_caches)
        return ids

    @torch.no_grad()
    def generate_oracle(self, input_ids, future_ids):
        """Prefill on input_ids, then accept every future block in one forward.

        future_ids: [B, T_future]. Processed in blocks of K = future_ids.shape[1]
        for a single-block call, or chunked if a later caller stacks blocks.
        Oracle: every drafted token is accepted; no rejection sampling.
        """
        kv_caches = [[None, None] for _ in self.layers]
        ids = input_ids.clone()
        self(ids, kv_caches)
        k = future_ids.shape[1]
        for start in range(0, future_ids.shape[1], k):
            block = future_ids[:, start:start + k]
            self(block, kv_caches)
            ids = torch.cat([ids, block], dim=1)
        return ids
