"""Exact-30 production MoE entry point.

Gated on ``MOSAICVM_EXACT_TARGET=1``.  Delegates to the Phase 2.7 grouped
XSIMT+QPA dispatcher.  ``skinny_fp4.fused_moe_skinny`` remains for 2.6 tests.
"""
from __future__ import annotations

from mosaicvm.kernels.qpa_mma import fused_moe_qpa
from mosaicvm.runtime.exact import exact_target


def exact_moe(x, expert_ids, expert_weights, packed_gate_up, scale_gate_up, packed_down, scale_down):
    """Exact-30 MoE: native grouped XSIMT+QPA. Requires exact_target()."""
    if not exact_target():
        raise RuntimeError("exact_moe requires MOSAICVM_EXACT_TARGET=1")
    return fused_moe_qpa(
        x, expert_ids, expert_weights,
        packed_gate_up, scale_gate_up, packed_down, scale_down,
    )


def exact_moe_mxint(x, expert_ids, expert_weights, packed_gate_up, scale_gate_up, packed_down, scale_down, bank=None):
    """MXINT fused MoE. Gated on exact_target(); QPA remains the Exact default."""
    if not exact_target():
        raise RuntimeError("exact_moe_mxint requires MOSAICVM_EXACT_TARGET=1")
    from mosaicvm.kernels.mxint import fused_moe_mxint
    return fused_moe_mxint(
        x, expert_ids, expert_weights,
        packed_gate_up, scale_gate_up, packed_down, scale_down, bank=bank,
    )

