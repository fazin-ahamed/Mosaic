"""Allocator-accounting singleton enforcing simulated consumer budgets.

Mirrors doc §42/§43: 10.5 GB "VRAM" for device tensors and ~14 GB "RAM" for
pinned host tensors, regardless of the physical cloud GPU pool. Exceeding a
budget raises, keeping the experiment honest about the consumer target.
"""
from __future__ import annotations

import torch

VRAM_BUDGET_BYTES = int(10.5 * 1024**3)
RAM_BUDGET_BYTES = int(14 * 1024**3)


class BudgetExceeded(RuntimeError):
    pass


class MemoryBudget:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self, vram=VRAM_BUDGET_BYTES, ram=RAM_BUDGET_BYTES):
        self.vram_budget = vram
        self.ram_budget = ram
        self.vram_used = 0
        self.ram_used = 0
        self.ledger: list[tuple[str, str, int]] = []

    def charge_device(self, tensor_or_bytes, tag: str):
        nbytes = tensor_or_bytes.numel() * tensor_or_bytes.element_size() if torch.is_tensor(tensor_or_bytes) else int(tensor_or_bytes)
        if self.vram_used + nbytes > self.vram_budget:
            raise BudgetExceeded(
                f"simulated VRAM budget exceeded: {self.vram_used + nbytes:,} > {self.vram_budget:,} charging '{tag}'"
            )
        self.vram_used += nbytes
        self.ledger.append(("vram", tag, nbytes))
        return nbytes

    def charge_host(self, tensor_or_bytes, tag: str):
        nbytes = tensor_or_bytes.numel() * tensor_or_bytes.element_size() if torch.is_tensor(tensor_or_bytes) else int(tensor_or_bytes)
        if self.ram_used + nbytes > self.ram_budget:
            raise BudgetExceeded(
                f"simulated RAM budget exceeded: {self.ram_used + nbytes:,} > {self.ram_budget:,} charging '{tag}'"
            )
        self.ram_used += nbytes
        self.ledger.append(("ram", tag, nbytes))
        return nbytes

    def release(self, nbytes: int, device: str = "vram"):
        if device == "vram":
            self.vram_used -= nbytes
        else:
            self.ram_used -= nbytes

    def peak_summary(self) -> dict:
        return {
            "vram_used_gb": round(self.vram_used / 1024**3, 2),
            "vram_budget_gb": round(self.vram_budget / 1024**3, 2),
            "ram_used_gb": round(self.ram_used / 1024**3, 2),
            "ram_budget_gb": round(self.ram_budget / 1024**3, 2),
        }


def budget() -> MemoryBudget:
    return MemoryBudget()
