"""Memory-budgeted streaming safetensors loader.

Resident set = core weights (embeddings, attention, routers, shared experts,
lm_head) + current layer. Routed experts are memory-mapped and loaded
per-selected-expert per layer through an LRU (slow-but-correct baseline).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open

from ..model.config import V4Config
from .memory_budget import budget


class ShardIndex:
    """Maps parameter name -> (shard file, dtype, shape)."""

    def __init__(self, model_root: str | Path):
        self.root = Path(model_root)
        self.index_path = self.root / "model.safetensors.index.json"
        self.shards: dict[str, str] = {}  # param -> shard filename
        if self.index_path.exists():
            idx = json.loads(self.index_path.read_text())
            for param, shard in idx["weight_map"].items():
                self.shards[param] = shard
        else:
            single = self.root / "model.safetensors"
            if single.exists():
                with safe_open(single, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        self.shards[k] = "model.safetensors"
            else:
                raise FileNotFoundError(f"no safetensors under {self.root}")

    def keys(self):
        return list(self.shards.keys())

    def load_tensor(self, param: str) -> torch.Tensor:
        with safe_open(self.root / self.shards[param], framework="pt", device="cpu") as f:
            return f.get_tensor(param)


class ExpertLRU:
    """Small LRU over routed-expert tensors charged to the simulated VRAM budget."""

    def __init__(self, shards: ShardIndex, capacity_bytes: int):
        self.shards = shards
        self.capacity = capacity_bytes
        self.cache: OrderedDict[str, tuple[int, torch.Tensor]] = OrderedDict()
        self.used = 0

    def get(self, param: str) -> torch.Tensor:
        if param in self.cache:
            self.cache.move_to_end(param)
            return self.cache[param][1]
        t = self.shards.load_tensor(param).to(torch.bfloat16)
        nb = t.numel() * t.element_size()
        b = budget()
        while self.used + nb > self.capacity and self.cache:
            _, (old_nb, _) = self.cache.popitem(last=False)
            b.release(old_nb)
            self.used -= old_nb
        b.charge_device(nb, f"lru:{param}")
        self.cache[param] = (nb, t)
        self.used += nb
        return t


def load_core_weights(cfg: V4Config, shards: ShardIndex, device="cuda"):
    """Load embeddings + lm_head into device memory within the VRAM budget."""
    b = budget()
    core = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        if name in shards.shards:
            t = shards.load_tensor(name).to(device=device, dtype=torch.bfloat16)
            b.charge_device(t, name)
            core[name] = t
    return core


class StreamingLoader:
    """Opens shards lazily; loads tensors on demand with budget accounting."""

    def __init__(self, model_root: str | Path, expert_cache_bytes: int | None = None):
        self.root = Path(model_root)
        self.shards = ShardIndex(self.root)
        cfg = V4Config.from_checkpoint(self.root)
        self.cfg = cfg
        if expert_cache_bytes is None:
            # leave headroom: experts get whatever fits after a rough core estimate
            embed_gb = (cfg.vocab_size * cfg.hidden_size * 2) / 1024**3
            expert_cache_bytes = int(10.5e9 - 2 * embed_gb * 1024**3)
        self.experts = ExpertLRU(self.shards, expert_cache_bytes)

    def load_tensor(self, param: str, device="cuda") -> torch.Tensor:
        with safe_open(self.root / self.shards.shards[param], framework="pt", device="cpu") as f:
            t = f.get_tensor(param)
        return t.to(device=device, dtype=torch.bfloat16)

    def load_raw_tensor(self, param: str, device="cpu") -> torch.Tensor:
        """Load WITHOUT dtype cast — required for packed FP4 int8/uint8
        containers whose bit patterns must survive (a bf16 cast destroys
        them; downstream view(torch.uint8) then misreads byte pairs)."""
        with safe_open(self.root / self.shards.shards[param], framework="pt", device="cpu") as f:
            t = f.get_tensor(param)
        return t.to(device=device)

    def load_expert_tensor(self, param: str) -> torch.Tensor:
        return self.experts.get(param)
