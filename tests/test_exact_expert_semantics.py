"""V4-0731 expert-semantics tests (Phase 2.8 Step 3).

Gate: reverting the router-weight-before-down multiply must fail these; the
hash case asserts real gathered/normalized/x1.5 weights and that a missing
tid2eid raises ExactContractError under the exact target.
"""
from __future__ import annotations

import os

os.environ.setdefault("MOSAICVM_EXACT_TARGET", "1")

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from mosaicvm.model.config import V4Config, validate_exact_v4_config
from mosaicvm.model.layers import HashRouter, _router_scores, grouped_expert_forward
from mosaicvm.runtime.exact import ExactContractError

SWIGLU_LIMIT = 10.0
ROUTE_SCALE = 1.5


def _exact_cfg():
    cfg = V4Config()
    cfg.n_routed_experts = 16
    cfg.hidden_size = 64
    cfg.moe_intermediate_size = 32
    cfg.num_experts_per_tok = 4
    return cfg


def _inline_expert_reference(x, idx, w, wg, wu, wd, swiglu_limit=SWIGLU_LIMIT):
    """Independent implementation of the documented expert math."""
    bt, k = idx.shape
    h = x.shape[-1]
    out = torch.zeros(bt, h, dtype=torch.float32, device=x.device)
    for t in range(bt):
        order = sorted(range(k), key=lambda s: int(idx[t, s]))
        acc = torch.zeros(h, dtype=torch.float32, device=x.device)
        for s in order:
            e = int(idx[t, s])
            gate = (wg[e].float() @ x[t].float()).clamp(max=swiglu_limit)
            up = (wu[e].float() @ x[t].float()).clamp(-swiglu_limit, swiglu_limit)
            z = F.silu(gate) * up
            z = z * float(w[t, s])
            zd = z.to(x.dtype).float()
            acc = acc + wd[e].float() @ zd
        out[t] = acc
    return out.to(x.dtype)


@pytest.mark.parametrize("m", [1, 2, 3, 8])
def test_grouped_expert_forward_matches_source_semantics(m):
    torch.manual_seed(m)
    cfg = _exact_cfg()
    e, h, i, k = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts_per_tok
    x = (torch.randn(m, h, device="cuda") * 6.0).to(torch.bfloat16)  # straddles +/- swiglu_limit
    wg = torch.randn(e, i, h, device="cuda") * 2.0
    wu = torch.randn(e, i, h, device="cuda") * 2.0
    wd = torch.randn(e, h, i, device="cuda") / (i ** 0.5)
    idx = torch.stack([torch.randperm(e, device="cuda")[:k] for _ in range(m)])
    wvals = torch.tensor([0.01, 0.1, 0.25, 0.5, 1.0, 1.5], device="cuda")
    w = wvals[torch.randint(0, len(wvals), (m, k), device="cuda")]
    got = grouped_expert_forward(
        x, w, idx, (wg.transpose(1, 2), wu.transpose(1, 2)), wd.transpose(1, 2), e, swiglu_limit=SWIGLU_LIMIT
    )
    want = _inline_expert_reference(x, idx, w, wg, wu, wd)
    diff = (got.float() - want.float()).abs().max().item()
    assert diff <= 2e-3, diff


def test_weight_before_down_not_after():
    """Q(r*x) != r*Q(x): weight applied pre-down (FP32) must match; post-down fails."""
    cfg = _exact_cfg()
    x = (torch.randn(2, cfg.hidden_size, device="cuda") * 6.0).to(torch.bfloat16)
    e, i, h = 4, cfg.moe_intermediate_size, cfg.hidden_size
    wg = torch.randn(e, i, h, device="cuda") * 2.0
    wu = torch.randn(e, i, h, device="cuda") * 2.0
    wd = torch.randn(e, h, i, device="cuda")
    idx = torch.tensor([[1, 3], [0, 2]], device="cuda")
    w = torch.tensor([[0.25, 1.5], [0.5, 0.01]], device="cuda")
    got = grouped_expert_forward(
        x, w, idx, (wg.transpose(1, 2), wu.transpose(1, 2)), wd.transpose(1, 2), e, swiglu_limit=SWIGLU_LIMIT
    ).float()
    # post-down variant: down on unweighted z, scale output afterwards (the phase-2.7 bug)
    post = torch.zeros_like(got)
    for t in range(2):
        acc = torch.zeros(h, device="cuda")
        for s in range(2):
            eid = int(idx[t, s])
            gate = (wg[eid] @ x[t].float()).clamp(max=SWIGLU_LIMIT)
            up = (wu[eid] @ x[t].float()).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            z = F.silu(gate) * up
            yd = wd[eid] @ z.to(x.dtype).float()
            acc = acc + float(w[t, s]) * yd
        post[t] = acc
    assert not torch.allclose(got, post, atol=1e-3), "weight-before-down must differ from after"

    # but the true pre-weight reference must match
    want = _inline_expert_reference(x, idx, w, wg, wu, wd)
    assert (got - want.float()).abs().max().item() <= 2e-3


def test_hash_router_real_scores_and_scaling():
    torch.manual_seed(0)
    cfg = _exact_cfg()
    r = HashRouter(cfg, 0).cuda()
    with torch.no_grad():
        r.weight.copy_(torch.randn_like(r.weight))
    vocab = 100
    r.tid2eid = torch.rand(cfg.n_routed_experts, device="cuda").argsort().reshape(vocab // 25, 4) % cfg.n_routed_experts
    r.tid2eid = torch.randint(0, cfg.n_routed_experts, (vocab, 4), device="cuda")
    x = torch.randn(7, cfg.hidden_size, device="cuda")
    tids = torch.arange(0, 49, 7, device="cuda")
    w, idx = r(x, tids)
    # expected: gathered original scores, row-normalized (sqrtsoftplus != softmax), x1.5
    original = _router_scores(F.linear(x.float(), r.weight.float()), cfg.scoring_func)
    got_exp = original.gather(-1, idx)
    got_exp = got_exp / got_exp.sum(-1, keepdim=True).clamp_min(1e-9)
    got_exp = got_exp * cfg.routed_scaling_factor
    assert w.dtype == torch.float32
    assert torch.allclose(w, got_exp, atol=1e-6), (w - got_exp).abs().max()
    assert idx.tolist() == r.tid2eid[tids].tolist()


def test_hash_router_missing_tid2eid_raises_under_exact():
    prev = os.environ.get("MOSAICVM_EXACT_TARGET", "1")
    os.environ["MOSAICVM_EXACT_TARGET"] = "1"
    cfg = _exact_cfg()
    r = HashRouter(cfg, 1).cuda()
    x = torch.randn(3, cfg.hidden_size, device="cuda")
    with pytest.raises(ExactContractError):
        r(x, torch.arange(3, device="cuda"))
    os.environ["MOSAICVM_EXACT_TARGET"] = "0"
    try:
        w, idx = r(x, torch.arange(3, device="cuda"))  # salted fallback allowed off-exact
        assert idx.shape[0] == 3
    finally:
        os.environ["MOSAICVM_EXACT_TARGET"] = prev


def test_validate_exact_v4_config_synthetic_pass_and_fail():
    meta = {"activation": "dynamic E4M3", "scale_fmt": "UE8M0", "expert_dtype": "FP4 E2M1"}
    validate_exact_v4_config(V4Config(), meta)
    # inference-config spelling also accepted
    raw = {
        "n_layers": 43, "num_hash_layers": 3, "n_activated_experts": 6,
        "score_func": "sqrtsoftplus", "route_scale": 1.5, "swiglu_limit": 10.0,
    }
    validate_exact_v4_config(raw, meta)
    bad = dict(meta, scale_fmt="BF16")
    with pytest.raises(ExactContractError):
        validate_exact_v4_config(V4Config(), bad)
    with pytest.raises(ExactContractError):
        validate_exact_v4_config({"n_layers": 42}, meta)


class _FakeSlice:
    def __init__(self, dt, shape):
        self._dt, self._shape = dt, shape

    def get_dtype(self):
        return self._dt

    def get_shape(self):
        return self._shape


def test_validate_exact_v4_tensors_wrong_layout_raises(tmp_path, monkeypatch):
    import mosaicvm.runtime.streaming_model as sm

    class FakeShards:
        shards = {
            "layers.0.ffn.experts.w1.weight": "x.safetensors",
            "layers.0.ffn.experts.w1.scale": "x.safetensors",
            "layers.0.ffn.gate.tid2eid": "x.safetensors",
        }

    class FakeLoader:
        shards = FakeShards()
        root = tmp_path
        cfg = V4Config()

    (tmp_path / "x.safetensors").write_bytes(b"")

    class FakeOpen:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_slice(self, name):
            # wrong layout: bf16 full-K weight
            return _FakeSlice(torch.bfloat16, (2048, 4096))

    monkeypatch.setattr(sm, "_safe_open", FakeOpen)
    with pytest.raises(ExactContractError):
        sm.validate_exact_v4_tensors(FakeLoader(), "layers.0.", 0)
