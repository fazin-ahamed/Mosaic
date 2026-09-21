from __future__ import annotations

import pytest

from mosaicvm.bench.wide_verify import (
    HOT8_BYTES_PER_EXPERT,
    SOURCE_BYTES_PER_EXPERT,
    _backend_mix,
    window_stats,
)


def test_window_stats_basic():
    ids = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 6, 7, 8],
        [0, 1, 9, 10, 11, 12],
    ]
    st = window_stats(ids)
    assert st["assignments"] == 18
    assert st["unique"] == 13
    assert st["mean_m"] == pytest.approx(18 / 13)
    assert st["r_k"] == pytest.approx(13 / 18)
    # counts: 0->3, 1->3, 2->2, rest->1 (10 experts)
    assert st["median_m"] == 1.0
    assert st["p90_m"] >= 1


def test_window_stats_flat_reuse():
    ids = [[7, 8, 9, 10, 11, 12]] * 8
    st = window_stats(ids)
    assert st["unique"] == 6
    assert st["assignments"] == 48
    assert st["mean_m"] == 8.0
    assert st["r_k"] == pytest.approx(0.125)


def test_window_stats_single_token():
    st = window_stats([[5, 6, 7, 8, 9, 10]])
    assert st["unique"] == 6
    assert st["assignments"] == 6
    assert st["mean_m"] == 1.0
    assert st["median_m"] == 1.0
    assert st["p90_m"] == 1.0
    assert st["r_k"] == 1.0


def test_backend_mix_queues():
    # experts 0,1 in every token (count 4 > 3 -> IMMA queue); the rest once each
    ids = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 6, 7, 8, 9],
        [0, 1, 10, 11, 12, 13],
        [0, 1, 14, 15, 16, 17],
    ]
    mix = _backend_mix(ids)
    assert mix["imma_share"] == pytest.approx(8 / 24)
    assert mix["dp4a_assignments"] == 16
    assert not mix["hexa"]
    # single token -> hexa path
    mix1 = _backend_mix([[9, 8, 7, 6, 5, 4]])
    assert mix1["hexa"]


def test_bytes_per_expert_ratio():
    # Hot8 carries ~2x the packed weight bytes per expert
    assert HOT8_BYTES_PER_EXPERT > 1.9 * SOURCE_BYTES_PER_EXPERT * 0.9
    assert SOURCE_BYTES_PER_EXPERT == (
        (2 * 2048 * 4096) // 2 + (4096 * 2048) // 2
        + (2 * 2048 * 4096 + 4096 * 2048) // 32
    )
