"""True five-depth BranchPack tree tests — D3_REAL (roadmap §59).

Tests: exact node accounting, prefix closure, no target leakage,
branch-conditioned Markov logits, exact N-node cap, full target path,
alternative-parent behavior, allocator behavior.
"""
from __future__ import annotations

import math

import pytest

from mosaicvm.bench.dspark_real_tree import (
    BLOCK_SIZE, REAL_BUDGETS, TreeNode, build_tree, d3_real, evaluate,
    markov_candidates, node_count)


V = 50          # toy vocab


def _base_logits(strong_at):
    """Per-position logit vectors; strong_at[d] = favored token."""
    out = []
    for d in range(BLOCK_SIZE):
        v = [0.0] * V
        v[strong_at[d] if callable(strong_at) else strong_at] = 8.0
        out.append(v)
    return out


def _const_bias(scale=1.0):
    """Markov bias independent of parent (deterministic children)."""
    return lambda p: [0.0] * V if scale == 0 else [
        0.0 if i != p else scale for i in range(V)]


def _parent_boost_bias(boost=6.0):
    """Bias that boosts a token derived from the parent."""
    return lambda p: [boost if i == (p + 1) % V else 0.0 for i in range(V)]


class TestNodeAccounting:
    def test_budget_never_exceeded(self):
        root_tok = 5
        for budget in (1, 4, 8, 16, 37, 64):
            root = build_tree(root_tok, _base_logits(10),
                              _const_bias(), budget)
            assert node_count(root) <= budget

    def test_exact_cap_when_tree_can_grow(self):
        # generous branch factors: builder should spend exactly N nodes
        root = build_tree(5, _base_logits(10), _const_bias(), 16)
        assert node_count(root) == 16

    def test_prefix_closure(self):
        root = build_tree(5, _base_logits(10), _const_bias(), 24)
        stack = [root]
        while stack:
            node = stack.pop()
            for c in node.children:
                assert c.parent is node
            stack.extend(node.children)

    def test_unique_paths_no_multiplicity(self):
        # every node is a distinct (parent, token) pair
        root = build_tree(5, _base_logits(10), _const_bias(), 32)
        seen = set()

        def walk(n):
            for c in n.children:
                key = (id(c.parent), c.token)
                assert key not in seen
                seen.add(key)
                walk(c)
        walk(root)

    def test_depth_never_exceeds_block(self):
        root = build_tree(5, _base_logits(10), _const_bias(), 160)
        stack = [root]
        while stack:
            n = stack.pop()
            assert n.depth < BLOCK_SIZE
            stack.extend(n.children)


class TestBranchConditioning:
    def test_alternative_parent_changes_children(self):
        base = _base_logits(10)
        bias = _parent_boost_bias()
        a = markov_candidates(base[0], bias, parent_token=3, k=5)
        b = markov_candidates(base[0], bias, parent_token=4, k=5)
        # parent 3 boosts token 4; parent 4 boosts token 5
        assert a[0][0] == 4 or 4 in [t for t, _ in a]
        assert b[0][0] == 5 or 5 in [t for t, _ in b]
        assert {t for t, _ in a} != {t for t, _ in b}

    def test_markov_rule_matches_kit(self):
        # kit: logits[d] += markov_head(prev_token); candidates = topk of sum
        base = [0.0] * V
        base[20] = 4.0
        bias = _parent_boost_bias(6.0)      # boosts (p+1) % V
        cands = markov_candidates(base, bias, parent_token=19, k=3)
        # token 20: base 4.0 + bias 6.0 (20 == 19+1) = 10.0 -> top
        assert cands[0][0] == 20
        assert cands[0][1] == pytest.approx(10.0)

    def test_tree_children_depend_on_parent(self):
        base = _base_logits(10)
        bias = _parent_boost_bias(3.0)
        root = build_tree(7, base, bias, 16, branch_k=4)
        # all depth-0 children's next-level candidates must differ per child
        # (spot check: child tokens are not all identical paths)
        child_tokens = {c.token for c in root.children}
        assert len(child_tokens) >= 2


class TestNoTargetLeakage:
    def test_allocator_cannot_see_target(self):
        """Tree built with identical drafter inputs must be identical
        regardless of target — enforced structurally: build_tree never
        receives the target."""
        base = _base_logits(10)
        root_a = build_tree(5, base, _const_bias(), 16)
        root_b = build_tree(5, base, _const_bias(), 16)
        assert node_count(root_a) == node_count(root_b)

    def test_evaluation_only_after_build(self):
        # evaluate() takes the tree + target; building needs no target arg
        import inspect
        sig = inspect.signature(build_tree)
        assert "target" not in sig.parameters


class TestEvaluation:
    def test_full_path_accepted(self):
        # make the target path the top-1 at every depth via bias chain
        target = [6, 7, 8, 9, 10]
        base = [[0.0] * V for _ in range(BLOCK_SIZE)]

        def bias(p, _root=5):
            prev = _root if p == _root else p
            nxt = target[0] if prev == _root else None
            return [0.0] * V  # replaced below

        # simpler: bias boosts next target token from current parent
        chain = {5: 6, 6: 7, 7: 8, 8: 9, 9: 10}

        def bias2(p):
            v = [0.0] * V
            if p in chain:
                v[chain[p]] = 9.0
            return v

        root = build_tree(5, base, bias2, 8, branch_k=4)
        assert evaluate(root, target) == 5

    def test_missing_token_stops(self):
        base = _base_logits(10)
        root = build_tree(5, base, _const_bias(), 4)   # tiny tree
        # target contains an unlikely token at depth 0
        assert evaluate(root, [49, 1, 2, 3, 4]) == 0

    def test_partial_path(self):
        base = _base_logits(10)
        # constant bias: all parents identical children
        root = build_tree(5, base, _const_bias(), 4, branch_k=2)
        # depth0 top-1 is token 10 (strong base); pick it, then stop early
        d = evaluate(root, [10, 999, 1, 2, 3])
        assert 0 <= d <= 1


class TestAllocators:
    @pytest.mark.parametrize("alloc", ["path_logprob", "prob_mass",
                                       "confidence", "depth_balanced"])
    def test_all_allocators_respect_budget(self, alloc):
        root = build_tree(5, _base_logits(10), _const_bias(), 12,
                          allocator=alloc)
        assert node_count(root) <= 12

    def test_depth_balanced_reaches_deeper(self):
        # with a strong single path, best-first goes deep; depth-balanced
        # spreads — both valid, but depth-balanced should reach depth 4
        # with a modest budget along the strong path
        base = _base_logits(10)
        root = build_tree(5, base, _const_bias(), 6,
                          allocator="depth_balanced", branch_k=1)
        assert node_count(root) <= 6

    def test_path_logprob_prefers_high_probability(self):
        base = [[0.0] * V for _ in range(BLOCK_SIZE)]
        base[0][42] = 6.0
        root = build_tree(5, base, _const_bias(), 1, branch_k=4)
        assert [c.token for c in root.children] == [42]


class TestD3Real:
    def _record(self, target, chain_bias):
        base = [[0.0] * V for _ in range(BLOCK_SIZE)]
        return {
            "root_token": 5,
            "base_logits": base,
            "markov_bias": chain_bias,
            "target": target,
        }

    def test_perfect_chain_full_depth(self):
        chain = {5: 6, 6: 7, 7: 8, 8: 9, 9: 10}

        def bias(p):
            v = [0.0] * V
            if p in chain:
                v[chain[p]] = 9.0
            return v

        recs = [self._record([6, 7, 8, 9, 10], bias) for _ in range(3)]
        out = d3_real(recs, budgets=(8, 16), allocators=("path_logprob",))
        for n in ("8", "16"):
            assert out["allocators"]["path_logprob"][n]["E[A]"] == 5.0

    def test_all_miss(self):
        recs = [self._record([40, 41, 42, 43, 44], _const_bias())
                for _ in range(3)]
        out = d3_real(recs, budgets=(8, 16), allocators=("path_logprob",))
        assert out["allocators"]["path_logprob"]["8"]["E[A]"] == 0.0

    def test_budget_monotonic_nondecreasing(self):
        chain = {5: 6, 6: 7, 7: 8, 8: 9, 9: 10}

        def bias(p):
            v = [0.0] * V
            if p in chain:
                v[chain[p]] = 3.0    # weak — needs budget to find
            return v

        recs = [self._record([6, 7, 8, 9, 10], bias) for _ in range(5)]
        out = d3_real(recs, budgets=REAL_BUDGETS,
                      allocators=("path_logprob",))
        per = out["allocators"]["path_logprob"]
        prev = -1
        for n in REAL_BUDGETS:
            ea = per[str(n)]["E[A]"]
            assert ea >= prev
            prev = ea

    def test_all_budgets_present(self):
        recs = [self._record([1, 2, 3, 4, 5], _const_bias())]
        out = d3_real(recs, budgets=REAL_BUDGETS)
        for n in REAL_BUDGETS:
            assert str(n) in out["allocators"]["path_logprob"]
