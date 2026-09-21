"""Exact-target contract for the production MoE path.

When ``MOSAICVM_EXACT_TARGET`` is truthy (default **on**) the model must route
exactly ``num_experts_per_tok`` (6) experts with source FP4/FP8 values and no
approximate shortcuts.  Allowed under exact: invertible repack, different
kernels, CUDA graphs, cache placement, lossless compression, speculative
decode with exact-target verification, approximate drafter.  Forbidden under
exact: ``Router.active_k != 6``, ``Router.aed``, ``MoELayer.use_aec``, REAP,
Ghost Expert, neuron dropping, lossy LBEF, ``pack_w4`` INT4 amax/7 on the
Exact-30 MoE/core path.
"""
from __future__ import annotations

import os

_FALSE = ("0", "false", "False", "no", "")


def exact_target() -> bool:
    """True unless ``MOSAICVM_EXACT_TARGET`` is explicitly disabled."""
    return os.environ.get("MOSAICVM_EXACT_TARGET", "1") not in _FALSE

class ExactContractError(RuntimeError):
    """Raised when the loaded model/config violates the V4-0731 exact contract."""
