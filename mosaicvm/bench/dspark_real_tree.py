"""True five-depth BranchPack tree — D3_REAL (roadmap §33-39).

A real prefix-closed tree under an EXACT total node budget, with
branch-conditioned candidate distributions per roadmap §34-35:

  candidates(depth d, parent p) = top-k of  base_logits[d] + markov_bias(p)

(the source-kit head rule: forward_head adds the Markov head's bias for
the chain-specific PREVIOUS token to the shared per-position base
logits — verified against .mosaicvm/source/inference_model.py).

Guarantees:
  * prefix-closed: a node exists only if its parent exists
  * exact budget: total non-root nodes <= N (unique (path) nodes;
    duplicate multiplicity never counts as extra nodes)
  * NO TARGET LEAKAGE: allocators see only drafter scores/confidence;
    the target is used exclusively by evaluate(), after the tree is built.

Allocators (drafter-info only):
  path_logprob   best-first by accumulated path log-probability
  prob_mass      best-first by per-node probability mass
  confidence     best-first by confidence-head value
  depth_balanced  level-wise round-robin expansion

Evaluation: longest prefix of the target path present in the tree.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

BLOCK_SIZE = 5
REAL_BUDGETS = (8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160)


@dataclass
class TreeNode:
    depth: int                      # 0 = committed root's children depth
    token: int
    parent: "TreeNode | None"
    score: float = 0.0              # drafter candidate score (logit)
    path_logprob: float = 0.0       # accumulated log-softmax
    confidence: float = 0.0
    children: list = field(default_factory=list)

    @property
    def path(self):
        node, out = self, []
        while node is not None and node.parent is not None:
            out.append(node.token)
            node = node.parent
        return list(reversed(out))


def markov_candidates(base_logit, markov_bias, parent_token, k):
    """Branch-conditioned candidate list at one depth.

    base_logit : [V] shared per-position logits (drafter body)
    markov_bias: callable(parent_token) -> [V] bias vector
    Returns [(token, logit)] sorted descending, length k.
    """
    bias = markov_bias(parent_token)
    if hasattr(base_logit, "add"):
        logits = base_logit.add(bias) if hasattr(bias, "shape") \
            else base_logit + bias
    else:                       # plain lists: element-wise
        logits = [b + x for b, x in zip(base_logit, bias)]
    k = min(k, logits.numel()) if hasattr(logits, "numel") else min(k, len(logits))
    if hasattr(logits, "topk"):
        vals, ids = logits.topk(k)
        return list(zip(ids.tolist(), vals.tolist()))
    order = sorted(range(len(logits)), key=lambda i: -logits[i])[:k]
    return [(i, logits[i]) for i in order]


def _logsoftmax_pairs(cands):
    m = max(s for _, s in cands) if cands else 0.0
    exps = [math.exp(s - m) for _, s in cands]
    z = sum(exps) or 1.0
    return [(tok, s, math.exp(s - m) / z) for tok, s in cands]


def build_tree(root_token, base_logits, markov_bias, budget,
               allocator="path_logprob", branch_k=8,
               confidence_fn=None):
    """Build a prefix-closed tree with total non-root nodes <= budget.

    base_logits : list of per-position logit vectors, len BLOCK_SIZE
    markov_bias : callable(parent_token) -> bias vector
    allocator   : one of path_logprob | prob_mass | confidence | depth_balanced
    branch_k    : max children considered per expansion
    confidence_fn: callable(node) -> float (drafter-side only)

    The allocator never sees the target.
    """
    assert budget >= 1
    root = TreeNode(depth=-1, token=root_token, parent=None)

    def expand(parent):
        """Return candidate children (TreeNode, not yet attached)."""
        d = parent.depth + 1
        if d >= BLOCK_SIZE or d >= len(base_logits):
            return []
        cands = markov_candidates(base_logits[d], markov_bias,
                                  parent.token, branch_k)
        out = []
        for tok, score, p in _logsoftmax_pairs(cands):
            node = TreeNode(depth=d, token=tok, parent=parent, score=score,
                            path_logprob=parent.path_logprob + math.log(
                                max(p, 1e-12)))
            if confidence_fn is not None:
                node.confidence = confidence_fn(node)
            out.append(node)
        return out

    if allocator == "depth_balanced":
        return _build_levelwise(root, expand, budget)
    return _build_bestfirst(root, expand, budget, allocator)


def _build_bestfirst(root, expand, budget, allocator):
    frontier = []           # heap of (-priority, tiebreak, node)
    counter = 0
    for child in expand(root):
        counter += 1
        pr = _priority(child, allocator)
        heapq.heappush(frontier, (-pr, counter, child))
    used = 0
    while used < budget and frontier:
        _, _, node = heapq.heappop(frontier)
        node.parent.children.append(node)   # each node pushed exactly once
        used += 1
        if used >= budget:
            break
        for child in expand(node):
            counter += 1
            pr = _priority(child, allocator)
            heapq.heappush(frontier, (-pr, counter, child))
    return root


def _build_levelwise(root, expand, budget):
    used = 0
    level = [root]
    while used < budget and level:
        nxt = []
        for parent in level:
            if used >= budget:
                break
            for child in expand(parent):
                if used >= budget:
                    break
                parent.children.append(child)
                nxt.append(child)
                used += 1
        level = nxt
    return root


def _priority(node, allocator):
    if allocator == "path_logprob":
        return node.path_logprob
    if allocator == "prob_mass":
        return math.exp(min(node.path_logprob, 0.0))
    if allocator == "confidence":
        return node.confidence if node.confidence else node.path_logprob
    raise ValueError(f"unknown allocator {allocator}")


def node_count(root):
    """Total non-root nodes (unique paths) — the budget actually spent."""
    n = 0
    stack = [root]
    while stack:
        node = stack.pop()
        n += len(node.children)
        stack.extend(node.children)
    return n


def evaluate(root, target_path):
    """Longest prefix of target_path present in the tree (target used
    ONLY here, after construction)."""
    node = root
    accepted = 0
    for tok in target_path[:BLOCK_SIZE]:
        nxt = next((c for c in node.children if c.token == tok), None)
        if nxt is None:
            break
        node = nxt
        accepted += 1
    return accepted


def d3_real(records, budgets=REAL_BUDGETS, allocators=("path_logprob",),
            branch_k=8):
    """D3_REAL over saved records.

    Record fields needed:
        root_token   int               committed token
        base_logits  list[[float]]     per-position shared logits (pre-Markov)
        markov_bias  callable | dict   fn(parent_tok) -> [V] bias, or a
                                       dict token -> bias list
        target       list[int]         FULL greedy target continuation
        confidence   optional list[float] per-depth (drafter-side)
    """
    out = {"label": "D3_REAL_TREE", "allocators": {}}
    for alloc in allocators:
        per_n = {}
        for n in budgets:
            depths = []
            for r in records:
                mb = r["markov_bias"]
                if isinstance(mb, dict):
                    mb = _dict_bias(mb)
                conf = r.get("confidence")
                conf_fn = (lambda node, c=conf:
                           c[node.depth] if c and node.depth < len(c)
                           else 0.0) if conf else None
                root = build_tree(r["root_token"], r["base_logits"], mb, n,
                                  allocator=alloc, branch_k=branch_k,
                                  confidence_fn=conf_fn)
                depths.append(evaluate(root, r["target"]))
            per_n[str(n)] = _stats(depths, f"D3_REAL_TREE_N{n}")
        out["allocators"][alloc] = per_n
    return out


def _dict_bias(table):
    def fn(parent_token):
        return table[parent_token]
    return fn


def _stats(depths, label):
    n = len(depths)
    if n == 0:
        return {"label": label, "n_blocks": 0, "E[A]": None}
    return {
        "label": label,
        "n_blocks": n,
        "E[A]": round(sum(depths) / n, 3),
        "hist": {str(i): depths.count(i) for i in range(BLOCK_SIZE + 1)},
        "P(depth>=d)": {
            str(d): round(sum(1 for x in depths if x >= d) / n, 4)
            for d in range(1, BLOCK_SIZE + 1)},
        "tokens_per_block_incl_bonus": round(sum(depths) / n + 1, 3),
    }
