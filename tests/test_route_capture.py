"""Route-capture pipeline correctness tests (AGENTS §10 of route-capture
authorization).

Tests the harness logic with synthetic fixtures — no checkpoint required.
Covers: sliding-window boundaries, N160-genuineness, no wrapping, layer
count, top-6, malformed rejection, metadata, deterministic frontier.
"""
from __future__ import annotations

import pytest

from mosaicvm.bench.route_capture_v2 import (
    WIDTHS, extract_expert_set, full_frontier_analysis,
    neighboring_overlap, sliding_window_analysis, validate_capture,
    working_set_estimate, _gini)


def _make_record(capture_id, pos, layer, expert_ids, weights=None):
    """Build a synthetic routing record."""
    n = len(expert_ids[0]) if isinstance(expert_ids[0], list) else 1
    if weights is None:
        weights = [[1.0 / 6] * 6 for _ in range(len(expert_ids))] \
            if isinstance(expert_ids[0], list) else [1.0 / 6] * 6
    return {
        "capture_id": capture_id, "pos": pos, "token": 100 + pos,
        "layer": layer, "router_class": "learned",
        "expert_ids": expert_ids, "weights": weights,
    }


def _synthetic_capture(n_positions=200, n_layers=43, capture_id="test",
                       expert_pattern="rotating"):
    """Generate a synthetic capture with known structure."""
    records = []
    for pos in range(n_positions):
        for layer in range(n_layers):
            if expert_pattern == "rotating":
                # each position uses a different (but overlapping) expert set
                base = (pos * 3 + layer * 7) % 250
                ids = [[(base + j) % 256 for j in range(6)]]
            elif expert_pattern == "fixed":
                ids = [[1, 2, 3, 4, 5, 6]]
            elif expert_pattern == "unique_per_pos":
                base = (pos * 6 + layer) % 251
                ids = [[(base + j) % 256 for j in range(6)]]
            records.append(_make_record(capture_id, pos, layer, ids))
    return records


# ---------------------------------------------------------- sanity gates ---

class TestValidateCapture:
    def test_valid_capture_passes(self):
        records = _synthetic_capture(20)
        result = validate_capture(records)
        assert result["ok"], result["errors"]
        assert result["n_layers"] == 43
        assert result["n_positions"] == 20

    def test_wrong_layer_count_fails(self):
        records = _synthetic_capture(10, n_layers=42)
        result = validate_capture(records)
        assert not result["ok"]
        assert any("layer count" in e for e in result["errors"])

    def test_wrong_expert_count_fails(self):
        records = _synthetic_capture(5)
        records[0]["expert_ids"] = [[1, 2, 3]]     # only 3 experts
        result = validate_capture(records)
        assert not result["ok"]
        assert any("expert count" in e for e in result["errors"])

    def test_out_of_range_expert_fails(self):
        records = _synthetic_capture(5)
        records[0]["expert_ids"] = [[1, 2, 3, 4, 5, 300]]   # 300 > 255
        result = validate_capture(records)
        assert not result["ok"]
        assert any("out of range" in e for e in result["errors"])

    def test_non_consecutive_positions_fail(self):
        records = _synthetic_capture(10)
        # remove a middle position
        records = [r for r in records if r["pos"] != 5]
        result = validate_capture(records)
        assert not result["ok"]
        assert any("non-consecutive" in e for e in result["errors"])

    def test_nan_weight_fails(self):
        records = _synthetic_capture(5)
        records[0]["weights"] = [[float("nan")] * 6]
        result = validate_capture(records)
        assert not result["ok"]
        assert any("non-finite" in e for e in result["errors"])

    def test_routing_collapse_fails(self):
        records = _synthetic_capture(20, expert_pattern="fixed")
        result = validate_capture(records)
        assert not result["ok"]
        assert any("collapse" in e for e in result["errors"])


# ------------------------------------------------------ sliding windows ----

class TestSlidingWindows:
    def test_n160_requires_genuine_positions(self):
        """A 160-wide window from only 100 positions must produce nothing."""
        records = _synthetic_capture(100)
        result = sliding_window_analysis(records, 160)
        assert result["windows"] == []
        assert result["union_stats"] == {}

    def test_200_positions_gives_41_windows_at_160(self):
        """200 consecutive positions -> 41 sliding windows at width 160."""
        records = _synthetic_capture(200)
        result = sliding_window_analysis(records, 160)
        assert len(result["windows"]) == 41
        assert result["union_stats"]["n_windows"] == 41

    def test_no_wrapping(self):
        """Windows must be contiguous slices, never wrapping."""
        records = _synthetic_capture(200)
        result = sliding_window_analysis(records, 160)
        for w in result["windows"]:
            assert w["end_pos"] - w["start_pos"] == 159
            assert w["start_pos"] >= 0
            assert w["end_pos"] < 200

    def test_unique_expert_count_increases_with_width(self):
        """Wider windows should see more or equal unique experts."""
        records = _synthetic_capture(
            200, expert_pattern="unique_per_pos")
        u64 = sliding_window_analysis(records, 64)["union_stats"]["mean"]
        u160 = sliding_window_analysis(records, 160)["union_stats"]["mean"]
        assert u160 >= u64

    def test_per_layer_unique_recorded(self):
        records = _synthetic_capture(50)
        result = sliding_window_analysis(records, 32)
        assert result["windows"]
        w = result["windows"][0]
        assert "per_layer_unique" in w
        assert len(w["per_layer_unique"]) == 43


# ------------------------------------------------------- analysis ----------

class TestFrontierAnalysis:
    def test_all_widths_present(self):
        records = _synthetic_capture(200)
        result = full_frontier_analysis(records)
        for w in WIDTHS:
            assert str(w) in result["widths"]

    def test_frequency_analysis(self):
        records = _synthetic_capture(50)
        freq = full_frontier_analysis(records)["frequency"]
        assert freq["total_assignments"] == 50 * 43 * 6
        assert freq["unique_experts_used"] > 0
        assert 0 <= freq["gini_coefficient"] <= 1

    def test_neighboring_overlap(self):
        records = _synthetic_capture(50)
        ov = neighboring_overlap(records)
        assert 0 <= ov["mean"] <= 1
        assert ov["n_comparisons"] > 0

    def test_deterministic(self):
        """Same input produces identical output."""
        records = _synthetic_capture(100)
        r1 = full_frontier_analysis(records)
        r2 = full_frontier_analysis(records)
        assert r1 == r2

    def test_working_set_estimate(self):
        records = _synthetic_capture(200)
        ws = working_set_estimate(records, 112)
        assert ws is not None
        assert ws["width"] == 112
        assert ws["mean_unique_experts"] > 0
        assert ws["hot8_est_gb"] > 0


# ------------------------------------------------------- gini -------------

class TestGini:
    def test_uniform_is_zero(self):
        assert abs(_gini([1, 1, 1, 1])) < 1e-10

    def test_maximal_inequality(self):
        # 4 values with 3 zeros: gini = 0.75 (the theoretical max for n=4
        # with this many zeros is 3/4 = 0.75, not 1.0)
        assert _gini([0, 0, 0, 100]) == 0.75
        assert _gini([0, 0, 0, 0, 0, 0, 0, 0, 0, 100]) > 0.89

    def test_empty(self):
        assert _gini([]) == 0.0


# ------------------------------------------------------- extract ----------

class TestExtractExpertSet:
    def test_nested_list(self):
        r = {"expert_ids": [[1, 2, 3], [4, 5, 6]]}
        assert extract_expert_set(r) == {1, 2, 3, 4, 5, 6}

    def test_flat_list(self):
        r = {"expert_ids": [1, 2, 3, 4, 5, 6]}
        assert extract_expert_set(r) == {1, 2, 3, 4, 5, 6}
