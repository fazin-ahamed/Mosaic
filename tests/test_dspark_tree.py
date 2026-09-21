"""D0–D3 analysis tests (synthetic records, no checkpoint)."""
from __future__ import annotations

from mosaicvm.bench.dspark_tree import (
    BLOCK_SIZE, REAL_BUDGETS, accepted_prefix, analyze, d0_sampled,
    d1_greedy, d2_coverage, d3_score_mass_surrogate, _surrogate_accepted)


def _rec(target, sampled, greedy, topk_ids, topk_scores=None, confidence=None):
    if topk_scores is None:
        topk_scores = [[float(16 - i) for i in range(len(ids))]
                       for ids in topk_ids]
    return {
        "target": target,
        "sampled": sampled,
        "greedy": greedy,
        "topk_ids": topk_ids,
        "topk_scores": topk_scores,
        "confidence": confidence,
        "target_greedy_path": target,
    }


def _perfect(n=4):
    """n blocks where greedy AND sampled match the full target."""
    recs = []
    for i in range(n):
        tgt = [10 + i, 20, 30, 40, 50]
        topk = [[tgt[d]] + list(range(100, 115)) for d in range(5)]
        recs.append(_rec(tgt, tgt, tgt, topk))
    return recs


def _miss_first(n=4):
    """n blocks where nothing matches at depth 0."""
    recs = []
    for i in range(n):
        tgt = [1, 2, 3, 4, 5]
        samp = [9, 9, 9, 9, 9]
        greed = [8, 8, 8, 8, 8]
        topk = [[8] + list(range(100, 115)) for _ in range(5)]
        recs.append(_rec(tgt, samp, greed, topk))
    return recs


class TestAcceptedPrefix:
    def test_full_match(self):
        assert accepted_prefix([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 5

    def test_first_mismatch(self):
        assert accepted_prefix([1, 9, 3, 4, 5], [1, 2, 3, 4, 5]) == 1

    def test_zero(self):
        assert accepted_prefix([9, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 0

    def test_empty(self):
        assert accepted_prefix([], [1]) == 0


class TestD0D1:
    def test_perfect_sampled_and_greedy(self):
        recs = _perfect(8)
        d0 = d0_sampled(recs)
        d1 = d1_greedy(recs)
        assert d0["E[A]"] == 5.0
        assert d1["E[A]"] == 5.0
        assert d0["P(depth>=d)"]["5"] == 1.0

    def test_sampled_worse_than_greedy(self):
        recs = []
        tgt = [1, 2, 3, 4, 5]
        # greedy matches all; sampled misses at 0
        recs.append(_rec(tgt, [9, 2, 3, 4, 5], tgt,
                         [[1] + list(range(10, 25)) for _ in range(5)]))
        d0 = d0_sampled(recs)
        d1 = d1_greedy(recs)
        assert d0["E[A]"] == 0.0
        assert d1["E[A]"] == 5.0

    def test_d1_prefers_greedy_path_target(self):
        recs = [_rec([1, 99, 99, 99, 99], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
                     [[1] + list(range(10, 25)) for _ in range(5)])]
        recs[0]["target_greedy_path"] = [1, 2, 3, 4, 5]
        assert d1_greedy(recs)["E[A]"] == 5.0
        assert d0_sampled(recs)["E[A]"] == 1.0


class TestD2Coverage:
    def test_top1_perfect(self):
        recs = _perfect(6)
        d2 = d2_coverage(recs)
        assert d2["P"]["1"] == [1.0] * 5
        assert d2["P"]["16"] == [1.0] * 5

    def test_target_in_top4_not_top1(self):
        tgt = [7, 7, 7, 7, 7]
        # target is 4th candidate at every depth
        topk = [[10, 11, 12, 7] + list(range(20, 32)) for _ in range(5)]
        recs = [_rec(tgt, [10] * 5, [10] * 5, topk)]
        d2 = d2_coverage(recs)
        assert d2["P"]["1"][0] == 0.0
        assert d2["P"]["2"][0] == 0.0
        assert d2["P"]["4"][0] == 1.0
        assert d2["P"]["8"][0] == 1.0
        # deeper depths not scored: greedy missed at 0
        assert d2["n_at_depth"][0] == 1
        assert d2["n_at_depth"][1] == 0

    def test_deeper_scored_only_after_greedy_match(self):
        tgt = [1, 2, 3, 4, 5]
        topk = [[1] + list(range(10, 25)),
                [2] + list(range(10, 25)),
                [99] + list(range(10, 25)),
                [4] + list(range(10, 25)),
                [5] + list(range(10, 25))]
        recs = [_rec(tgt, tgt, [1, 2, 9, 4, 5], topk)]
        d2 = d2_coverage(recs)
        assert d2["n_at_depth"] == [1, 1, 1, 0, 0]


class TestD3Tree:
    def test_perfect_tree_any_budget(self):
        recs = _perfect(3)
        d3 = d3_score_mass_surrogate(recs, budgets=(16, 80, 160))
        for n in ("16", "80", "160"):
            assert d3["budgets"][n]["E[A]"] == 5.0

    def test_zero_when_target_absent(self):
        recs = _miss_first(3)
        d3 = d3_score_mass_surrogate(recs, budgets=(16, 160))
        assert d3["budgets"]["16"]["E[A]"] == 0.0
        assert d3["budgets"]["160"]["E[A]"] == 0.0

    def test_top4_hit_rescues_with_enough_budget(self):
        # target is 4th candidate at depth 0, then top-1 thereafter
        tgt = [7, 2, 3, 4, 5]
        topk = [[10, 11, 12, 7] + list(range(20, 32))]
        topk += [[tgt[d]] + list(range(100, 115)) for d in range(1, 5)]
        recs = [_rec(tgt, [10] * 5, [10] * 5, topk)]
        # tiny budget: all mass on top-1 (token 10) → miss
        a16 = _surrogate_accepted(recs[0], 16)
        # large budget: 4th candidate gets nodes → hit, then rest match
        a160 = _surrogate_accepted(recs[0], 160)
        assert a16 == 0
        assert a160 == 5

    def test_budget_monotonic(self):
        tgt = [7, 2, 3, 4, 5]
        topk = [[10, 11, 12, 7] + list(range(20, 32))]
        topk += [[tgt[d]] + list(range(100, 115)) for d in range(1, 5)]
        recs = [_rec(tgt, [10] * 5, [10] * 5, topk)]
        prev = -1
        for n in REAL_BUDGETS:
            a = _surrogate_accepted(recs[0], n)
            assert a >= prev
            prev = a

    def test_all_budgets_present(self):
        recs = _perfect(2)
        d3 = d3_score_mass_surrogate(recs)
        for n in REAL_BUDGETS:
            assert str(n) in d3["budgets"]


class TestAnalyze:
    def test_keys(self):
        recs = _perfect(2)
        out = analyze(recs)
        assert set(out) >= {"n_records", "D0", "D1", "D2",
                            "D3_surrogate"}
        assert out["n_records"] == 2

    def test_empty(self):
        out = analyze([])
        assert out["n_records"] == 0
        assert out["D0"]["E[A]"] is None
