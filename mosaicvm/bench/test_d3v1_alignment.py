"""Roadmap S14 regression: token-indexing alignment across ALL D3-v1
code paths, and the S16 cl_loop gold-leak guard.

Contract (d3_v1.make_feat): predicting the state at position `pos`
consumes (input state h40[pos-1], delta, DRIVER = tokens[pos] — the
token that produces h40[pos]; in production the draft chosen from
h42[pos-1] — CUR = tokens[pos-1], pos). All paths route through
make_feat, which this test monkeypatches to record every call.

Asserts:
  1. build_transitions (training), rollout_onestep (TF export) and
     build_exposure emit IDENTICAL (driver, cur, pos) sequences for
     the same capture.
  2. rollout_chain (chained states, true tokens) emits the same
     token triples (states differ by design — chaining).
  3. cl_loop feeds the student ONLY its own tail-decoded drafts as
     driver after the exact root — no gold future token, no gold
     state (stub drafts are disjoint from the true-token range).
  4. tf_loop feeds the true tokens.
  5. The exact root (position S) driver is the root draft.

run_loop mechanics run against a deterministic stub tail (no GPU).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mosaicvm.bench.d3_v1 as v1
import mosaicvm.bench.d3_v1_eval as v1e
from mosaicvm.bench.d3_v1 import (build_transitions, build_exposure,
                                  rollout_chain, rollout_onestep)

run_loop = v1e.run_loop

POS_DIM = v1.POS_DIM
D_STATE = v1.D_STATE
VOCAB = 500          # true tokens live in [100, 500)
ROOT_MARK = 1.0
STEP_MARK = 2.0
DRAFT_TOKEN = 7      # stub drafts; disjoint from true-token range

_real_feat = v1.make_feat
LOG = []


def rec_feat(embed, state, delta_in, driver_tok, cur_tok, pos):
    LOG.append((int(driver_tok), int(cur_tok), int(pos)))
    return _real_feat(embed, state, delta_in, driver_tok, cur_tok,
                      pos)


v1.make_feat = rec_feat
v1e.make_feat = rec_feat


class ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], D_STATE)


class StubRunner:
    """Deterministic tail: root logits -> true root token; step
    logits -> DRAFT_TOKEN for every position."""

    def __init__(self, root_token):
        self.root = root_token
        self.resets = 0

    def reset(self):
        self.resets += 1

    def prefill(self, h40_prefix, tokens):
        assert h40_prefix.ndim == 3 and h40_prefix.shape[1:] == (4, 4096), \
            f"prefill shape contract violated: {tuple(h40_prefix.shape)}"
        n = h40_prefix.shape[0]
        rows = torch.zeros(n, 4, 4096)
        rows[-1, 0, 0] = ROOT_MARK
        return rows

    def step(self, h40_p, p, token_p):
        assert h40_p.ndim == 2 and h40_p.shape == (4, 4096), \
            f"step shape contract violated: {tuple(h40_p.shape)}"
        row = torch.zeros(4, 4096)
        row[0, 0] = STEP_MARK
        return row

    def logits(self, h42_row):
        lg = torch.zeros(VOCAB)
        if h42_row[0, 0] == ROOT_MARK:
            lg[self.root] = 10.0
        else:
            lg[DRAFT_TOKEN] = 10.0
        return lg


def make_capture(S=6, K=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = S + K
    toks = torch.randint(100, VOCAB, (n + 1,), generator=g).tolist()
    h = (torch.randn(n, 4, 4096, generator=g) * 2
         ).to(torch.bfloat16)
    return {"category": "fake_01", "split": "train", "S": S,
            "tokens": toks, "h40": h, "h42": h.clone()}


def triples_since(n0):
    return LOG[n0:]


def main():
    cap = make_capture()
    S, K, toks = cap["S"], 8, cap["tokens"]
    embed = torch.randn(VOCAB, 32)

    # ---- 1: training / one-step export / exposure alignment ----
    n0 = len(LOG)
    ds = build_transitions([cap], embed, "train")
    tr_train = triples_since(n0)
    assert len(tr_train) == K, len(tr_train)
    for j, (drv, cur, pos) in enumerate(tr_train):
        p = S + j
        assert (drv, cur, pos) == (toks[p], toks[p - 1], p), \
            (j, (drv, cur, pos), (toks[p], toks[p - 1], p))

    model = ZeroModel()
    zero = torch.zeros(D_STATE)          # target mean
    ones = torch.ones(D_STATE)           # target std
    feat_dim = 2 * D_STATE + 2 * 32 + POS_DIM   # test embed dim 32
    x_mean = torch.zeros(feat_dim)
    x_std = torch.ones(feat_dim)
    pm = {"category": cap["category"], "S": S, "K": K,
          "capture": cap}

    n0 = len(LOG)
    rollout_onestep(model, pm, embed, ones, zero)
    tr_os = triples_since(n0)
    assert tr_os == tr_train, "one_step != training pairing"

    n0 = len(LOG)
    build_exposure(model, ds, embed, ones, zero, x_mean, x_std)
    tr_exp = triples_since(n0)
    assert tr_exp == tr_train, "exposure != training pairing"

    # ---- 2: chained rollout token triples (states differ) ----
    n0 = len(LOG)
    rollout_chain(model, pm, embed, ones, zero, x_mean, x_std)
    tr_chain = triples_since(n0)
    assert tr_chain == tr_train, "chain token pairing != training"

    # ---- 3/4/5: loop modes with the stub tail ----
    stub = StubRunner(root_token=toks[S])
    n0 = len(LOG)
    _, hits_cl = run_loop(stub, model, embed, zero, ones,
                          x_mean, x_std, cap, "cl_loop", K)
    tr_cl = triples_since(n0)
    assert len(tr_cl) == K
    # root driver then own drafts only
    assert tr_cl[0][0] == toks[S], "root driver must be root draft"
    for j in range(1, K):
        assert tr_cl[j][0] == DRAFT_TOKEN, \
            f"cl_loop step {j}: gold token leaked as driver"
    # no gold future token ever fed as driver
    for j, (drv, cur, pos) in enumerate(tr_cl):
        assert drv not in toks[S + 1:], "gold future token fed"
    # every cl hit must be False (draft 7 != true tokens by design)
    assert not any(hits_cl), "stub drafts cannot match true tokens"

    n0 = len(LOG)
    _, hits_tf = run_loop(stub, model, embed, zero, ones,
                          x_mean, x_std, cap, "tf_loop", K)
    tr_tf = triples_since(n0)
    for j, (drv, cur, pos) in enumerate(tr_tf):
        p = S + j
        assert drv == toks[p], "tf_loop must feed true driver"

    print("ALL D3-v1 ALIGNMENT REGRESSIONS PASS")
    print(f"  training/one_step/exposure/chain triples identical "
          f"({K} steps)")
    print("  cl_loop: root + own drafts only (no gold leak)")
    print("  tf_loop: true tokens")


if __name__ == "__main__":
    main()
