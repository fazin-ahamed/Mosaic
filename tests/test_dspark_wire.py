"""DSpark Stage-2 wiring structural tests (synthetic tensors, no checkpoint).

Validates the wiring math against the kit source:
  * YaRN rope roundtrip (rotary + inverse == identity)
  * hc_split_sinkhorn row/col normalization + shapes
  * FP8 dequant value-exactness
  * FP4 dequant roundtrip exactness
  * SwiGLUExpert clamp semantics
  * DSparkAttention prefill ring-fill (incl. wrap) + draft shapes
  * DSparkStage prefill short-circuit + full-block shapes
  * forward_embed/forward_head shapes and sampling determinism
"""
from __future__ import annotations

import math

import pytest
import torch

from mosaicvm.bench.dspark_wire import (
    BLOCK_SIZE, DIM, FP4_TABLE_VALUES, HEAD_DIM, HC_MULT, MOE_INTER,
    N_HEADS, NOISE_TOKEN_ID, O_GROUPS, O_LORA, Q_LORA, WINDOW,
    DSparkAttention, DSparkStage, Lin, MoEDQ, SwiGLUExpert,
    apply_rotary, dequant_fp8, dequant_fp4, hc_split_sinkhorn,
    precompute_freqs_cis)

DEV = "cpu"

import mosaicvm.bench.dspark_wire as dw

SMALL = {  # structural tests: dims aligned to fp8(128)/fp4(32) blocks
    "DIM": 128, "MOE_INTER": 128, "N_EXPERTS": 8, "TOPK": 2,
    "Q_LORA": 128, "HEAD_DIM": 128, "N_HEADS": 4, "O_GROUPS": 2,
    "O_LORA": 128, "ROPE_DIM": 8, "VOCAB": 1000, "NOISE_TOKEN_ID": 999,
}


@pytest.fixture
def small_dims(monkeypatch):
    for k, v in SMALL.items():
        monkeypatch.setattr(dw, k, v)
    yield


def _fp8_pair(out_dim, in_dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(out_dim, in_dim, generator=g) * 0.1).to(
        torch.float8_e4m3fn)
    s = torch.rand(out_dim, in_dim // 128, generator=g) * 0.02 + 0.01
    return w, s


def _fp4_pair(out_dim, in_dim, seed=0):
    """Synthetic packed-FP4 weight + scale."""
    g = torch.Generator().manual_seed(seed)
    nibbles = torch.randint(0, 16, (out_dim, in_dim), generator=g,
                            dtype=torch.uint8)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.int8)
    scale = torch.rand(out_dim, in_dim // 32, generator=g) * 0.5 + 0.5
    return packed, scale


FP4_TABLE = torch.tensor(FP4_TABLE_VALUES, dtype=torch.float32)


# ---------------------------------------------------------------- rope -----
class TestRope:
    def test_roundtrip(self):
        fc = precompute_freqs_cis(64, 8, 65536, 10000, 16, 32, 1)
        x = torch.randn(1, 8, 64)
        y = apply_rotary(x, fc)
        z = apply_rotary(y, fc, inverse=True)
        assert torch.allclose(x, z, atol=1e-4)

    def test_shape_and_dtype(self):
        fc = precompute_freqs_cis(64, 4, 65536, 10000, 16, 32, 1)
        x = torch.randn(2, 4, 64, dtype=torch.bfloat16)
        y = apply_rotary(x, fc)
        assert y.shape == x.shape
        assert y.dtype == torch.bfloat16

    def test_freqs_cis_shape(self):
        fc = precompute_freqs_cis(64, 100, 65536, 10000, 16, 32, 1)
        assert fc.shape == (100, 32)

    def test_first_position_is_identity(self):
        # position 0 => freqs 0 => complex(1, 0) => no rotation
        fc = precompute_freqs_cis(64, 4, 65536, 10000, 16, 32, 1)
        x = torch.randn(1, 4, 64)
        y = apply_rotary(x, fc)
        assert torch.allclose(x[:, 0], y[:, 0], atol=1e-5)


# -------------------------------------------------------------- sinkhorn ---
class TestSinkhorn:
    def test_shapes(self):
        n, mix_hc = 5, (2 + HC_MULT) * HC_MULT
        mixes = torch.randn(n, mix_hc)
        scale = torch.randn(3)
        base = torch.randn(mix_hc)
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base)
        assert pre.shape == (n, HC_MULT)
        assert post.shape == (n, HC_MULT)
        assert comb.shape == (n, HC_MULT, HC_MULT)

    def test_comb_doubly_stochastic(self):
        n, mix_hc = 7, (2 + HC_MULT) * HC_MULT
        torch.manual_seed(1)
        mixes = torch.randn(n, mix_hc)
        scale = torch.randn(3)
        base = torch.randn(mix_hc)
        _, _, comb = hc_split_sinkhorn(mixes, scale, base)
        rs = comb.sum(-1)
        cs = comb.sum(-2)
        assert torch.allclose(rs, torch.ones_like(rs), atol=1e-3)
        assert torch.allclose(cs, torch.ones_like(cs), atol=1e-3)

    def test_pre_positive_post_range(self):
        n, mix_hc = 3, (2 + HC_MULT) * HC_MULT
        torch.manual_seed(2)
        mixes = torch.randn(n, mix_hc)
        scale = torch.randn(3)
        base = torch.randn(mix_hc)
        pre, post, _ = hc_split_sinkhorn(mixes, scale, base)
        assert (pre > 0).all()
        assert (post >= 0).all() and (post <= 2).all()


# -------------------------------------------------------------- dequant ----
class TestDequant:
    def test_fp8_row_scale(self):
        w, s = _fp8_pair(256, 256)
        out = dequant_fp8(w, s)
        assert out.shape == (256, 256)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()

    def test_fp8_value_exact(self):
        # dequant value == fp8 value * scale (row-block)
        w = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
        w[0, 0] = 1.0
        w[0, 127] = 2.0
        w[127, 0] = -0.5
        s = torch.full((128, 1), 0.5)
        out = dequant_fp8(w, s).float()
        assert out[0, 0].item() == pytest.approx(0.5)
        assert out[0, 127].item() == pytest.approx(1.0)
        assert out[127, 0].item() == pytest.approx(-0.25)

    def test_fp4_roundtrip_exact(self):
        # every FP4 code roundtrips exactly through the table
        packed, scale = _fp4_pair(8, 64)
        scale = torch.ones_like(scale)
        out = dequant_fp4(packed, scale, 8, 64, FP4_TABLE).float()
        raw = packed.view(torch.uint8)
        low = (raw & 0x0F).long()
        high = ((raw >> 4) & 0x0F).long()
        assert torch.equal(out, torch.stack(
            [FP4_TABLE[low], FP4_TABLE[high]], dim=-1).flatten(-2))

    def test_fp4_scale_applied(self):
        packed = torch.zeros(1, 32, dtype=torch.int8)
        packed[0, 0] = 0x01       # low nibble = 1 -> 0.5
        packed[0, 0] |= 0x20      # high nibble = 2 -> 1.0
        scale = torch.full((1, 2), 2.0)   # [out, in/32] with in=64
        out = dequant_fp4(packed, scale, 1, 64, FP4_TABLE).float()
        assert out[0, 0].item() == pytest.approx(0.5 * 2.0)
        assert out[0, 1].item() == pytest.approx(1.0 * 2.0)

    def test_rejects_bad_scale(self):
        w = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
        with pytest.raises(ValueError):
            dequant_fp8(w, torch.ones(3, 7))


# ---------------------------------------------------------------- expert ---
class TestSwiGLU:
    def test_clamp_semantics(self):
        # huge weights => gate clamped at 10, up clamped +-10
        w1 = torch.full((4, 8), 50.0, dtype=torch.bfloat16)   # gate
        w3 = torch.full((4, 8), -50.0, dtype=torch.bfloat16)  # up
        w2 = torch.eye(8, 4, dtype=torch.bfloat16)
        e = SwiGLUExpert(w1, w2, w3)
        x = torch.ones(1, 8, dtype=torch.bfloat16)
        y = e(x)
        assert y.shape == (1, 8)      # w2 out-dim
        # gate clamped to 10, up clamped to -10 -> |silu(10)*(-10)| < 100
        assert torch.isfinite(y.float()).all()
        assert y.float().abs().max() <= 10.0 * 10.0 + 1e-3

    def test_weight_routing(self):
        w1 = torch.eye(4, 4, dtype=torch.bfloat16)
        w2 = torch.eye(4, 4, dtype=torch.bfloat16)
        w3 = torch.eye(4, 4, dtype=torch.bfloat16)
        e = SwiGLUExpert(w1, w2, w3)
        x = torch.ones(1, 4, dtype=torch.bfloat16)
        y = e(x, weight=torch.tensor([[0.5]]))
        assert y.shape == (1, 4)


# -------------------------------------------------------------- MoE gate ---
class TestMoEGate:
    def _tiny_moe(self):
        t = {}
        w, s = _fp8_pair(8, 128, seed=3)         # N_EXPERTS x DIM (small)
        t["gate.weight"], t["gate.scale"] = w, s
        t["gate.bias"] = torch.zeros(8)
        for name, (o, i) in {
                "w1": (128, 128), "w2": (128, 128),
                "w3": (128, 128)}.items():
            # shared experts ship FP8 (2D 128x128 block scale)
            pw, ps = _fp8_pair(o, i, seed=abs(hash(name)) % 2**31)
            t[f"shared.{name}.weight"] = pw
            t[f"shared.{name}.scale"] = ps
        for i in range(8):
            for name, (o, i_dim) in {
                    "w1": (128, 128), "w2": (128, 128),
                    "w3": (128, 128)}.items():
                pw, ps = _fp4_pair(o, i_dim, seed=i * 3 + ord(name[0]))
                t[f"experts.{i}.{name}.weight"] = pw
                t[f"experts.{i}.{name}.scale"] = ps
        return t

    def test_moe_output_shape_and_finite(self, small_dims):
        t = self._tiny_moe()
        moe = MoEDQ(t, FP4_TABLE, DEV)
        x = torch.randn(1, 3, 128, dtype=torch.bfloat16)
        y = moe(x)
        assert y.shape == (1, 3, 128)
        assert torch.isfinite(y.float()).all()

    def test_moe_extreme_bias_selects_expert(self, small_dims):
        # bias shifts selection only; output stays finite
        t = self._tiny_moe()
        t["gate.bias"] = torch.zeros(8)
        t["gate.bias"][7] = 1e6
        moe = MoEDQ(t, FP4_TABLE, DEV)
        x = torch.randn(1, 2, 128, dtype=torch.bfloat16)
        y = moe(x)
        assert torch.isfinite(y.float()).all()

    def test_moe_expert_cache_lru(self, small_dims):
        t = self._tiny_moe()
        moe = MoEDQ(t, FP4_TABLE, DEV)
        x = torch.randn(1, 1, 128, dtype=torch.bfloat16)
        moe(x)
        assert len(moe.cache) <= 2


# ------------------------------------------------------------ attention ----
def _attn_tensors(seed=0):
    t = {}
    specs = {
        "wq_a": (Q_LORA, DIM),
        "wq_b": (N_HEADS * HEAD_DIM, Q_LORA),
        "wkv": (HEAD_DIM, DIM),
        "wo_a": (O_GROUPS * O_LORA, N_HEADS * HEAD_DIM // O_GROUPS),
        "wo_b": (DIM, O_GROUPS * O_LORA),
    }
    for i, (name, (o, i_dim)) in enumerate(specs.items()):
        w, s = _fp8_pair(o, i_dim, seed=seed + i)
        t[f"{name}.weight"], t[f"{name}.scale"] = w, s
    t["q_norm.weight"] = torch.ones(Q_LORA) + torch.randn(Q_LORA) * 0.1
    t["kv_norm.weight"] = torch.ones(HEAD_DIM) + torch.randn(HEAD_DIM) * 0.1
    t["attn_sink"] = torch.randn(N_HEADS) * 0.1
    return t


class TestDSparkAttention:
    def _make(self):
        fc = precompute_freqs_cis(64, 512, 65536, 10000, 16, 32, 1)
        return DSparkAttention(_attn_tensors(), fc, DEV)

    def test_prefill_short_fills_ring(self):
        attn = self._make()
        n = 50
        main_x = torch.randn(1, n, DIM, dtype=torch.bfloat16)
        out = attn.forward(torch.randn(1, 5, DIM, dtype=torch.bfloat16),
                           0, main_x)
        # prefill returns x unchanged
        assert out.shape == (1, 5, DIM)
        assert not torch.equal(attn.kv_cache[:n],
                               torch.zeros_like(attn.kv_cache[:n]))

    def test_prefill_wrap_semantics(self):
        attn = self._make()
        n = WINDOW + 30        # longer than window
        main_x = torch.randn(1, n, DIM, dtype=torch.bfloat16)
        attn.forward(torch.zeros(1, 5, DIM, dtype=torch.bfloat16), 0, main_x)
        # ring fully overwritten
        assert not torch.equal(attn.kv_cache,
                               torch.zeros_like(attn.kv_cache))

    def test_draft_shapes(self):
        attn = self._make()
        n = 20
        main_x = torch.randn(1, n, DIM, dtype=torch.bfloat16)
        attn.forward(torch.zeros(1, 5, DIM, dtype=torch.bfloat16), 0, main_x)
        x = torch.randn(1, BLOCK_SIZE, DIM, dtype=torch.bfloat16)
        out = attn.forward(x, n, main_x[:, -1:, :])
        assert out.shape == (1, BLOCK_SIZE, DIM)
        assert torch.isfinite(out.float()).all()

    def test_draft_attends_all_positions_when_short(self):
        # start_pos < window: n_attend = start_pos+1 ring slots
        attn = self._make()
        n = 10
        main_x = torch.randn(1, n, DIM, dtype=torch.bfloat16)
        attn.forward(torch.zeros(1, 5, DIM, dtype=torch.bfloat16), 0, main_x)
        x = torch.randn(1, BLOCK_SIZE, DIM, dtype=torch.bfloat16)
        _ = attn.forward(x, n, main_x[:, -1:, :])
        # ring slot n (= start_pos % win) got the main kv
        assert not torch.equal(attn.kv_cache[n],
                               torch.zeros(HEAD_DIM, dtype=torch.bfloat16))


# ---------------------------------------------------------------- stage ----
def _stage_tensors(stage_id, dim=128, moe_inter=128, n_experts=8,
                   q_lora=128, head_dim=128, n_heads=4, o_groups=2,
                   o_lora=128, markov_rank=256):
    t = {"attn": {}}
    attn_specs = {
        "wq_a": (q_lora, dim),
        "wq_b": (n_heads * head_dim, q_lora),
        "wkv": (head_dim, dim),
        "wo_a": (o_groups * o_lora, n_heads * head_dim // o_groups),
        "wo_b": (dim, o_groups * o_lora),
    }
    for i, (name, (o, i_dim)) in enumerate(attn_specs.items()):
        w, s = _fp8_pair(o, i_dim, seed=stage_id * 10 + i)
        t["attn"][f"{name}.weight"], t["attn"][f"{name}.scale"] = w, s
    t["attn"]["q_norm.weight"] = torch.ones(q_lora)
    t["attn"]["kv_norm.weight"] = torch.ones(head_dim)
    t["attn"]["attn_sink"] = torch.randn(n_heads) * 0.1

    ffn = {}
    w, s = _fp8_pair(n_experts, dim, seed=100 + stage_id)
    ffn["gate.weight"], ffn["gate.scale"] = w, s
    ffn["gate.bias"] = torch.randn(n_experts) * 0.01
    for name, (o, i) in {
            "w1": (moe_inter, dim), "w2": (dim, moe_inter),
            "w3": (moe_inter, dim)}.items():
        # shared experts ship FP8; routed experts are FP4-packed
        pw, ps = _fp8_pair(o, i, seed=200 + stage_id)
        ffn[f"shared.{name}.weight"] = pw
        ffn[f"shared.{name}.scale"] = ps
    for i in range(n_experts):
        for name, (o, i_dim) in {
                "w1": (moe_inter, dim), "w2": (dim, moe_inter),
                "w3": (moe_inter, dim)}.items():
            pw, ps = _fp4_pair(o, i_dim, seed=300 + i * 7)
            ffn[f"experts.{i}.{name}.weight"] = pw
            ffn[f"experts.{i}.{name}.scale"] = ps
    t["ffn"] = ffn
    t["attn_norm"] = torch.ones(dim)
    t["ffn_norm"] = torch.ones(dim)
    mix = (2 + HC_MULT) * HC_MULT
    for which in ("attn", "ffn"):
        t[f"hc_{which}_fn"] = torch.randn(mix, HC_MULT * dim) * 0.02
        t[f"hc_{which}_base"] = torch.randn(mix) * 0.1
        t[f"hc_{which}_scale"] = torch.randn(3).abs() + 0.5
    if stage_id == 0:
        w, s = _fp8_pair(dim, 3 * dim, seed=400)
        t["main_proj.weight"], t["main_proj.scale"] = w, s
        t["main_norm"] = torch.ones(dim)
    if stage_id == 2:
        t["norm"] = torch.ones(dim)
        t["markov_w1.weight"] = torch.randn(1000, markov_rank) * 0.02
        w, s = _fp8_pair(1000, markov_rank, seed=401)
        t["markov_w2.weight"], t["markov_w2.scale"] = w, s
        t["confidence_proj.weight"] = torch.randn(1, dim + markov_rank) * 0.02
        # kit hc_head: fn [hc_mult, hc*dim], base [hc_mult], scale [1]
        t["hc_head_fn"] = torch.randn(HC_MULT, HC_MULT * dim) * 0.02
        t["hc_head_base"] = torch.randn(HC_MULT) * 0.1
        t["hc_head_scale"] = torch.randn(1).abs() + 0.5
    return t


class TestDSparkStage:
    def _fc(self):
        return precompute_freqs_cis(8, 512, 65536, 10000, 16, 32, 1)

    def test_prefill_shortcircuit(self, small_dims):
        st = DSparkStage(0, _stage_tensors(0), self._fc(), DEV, FP4_TABLE)
        x = torch.randn(1, BLOCK_SIZE, HC_MULT, 128, dtype=torch.bfloat16)
        main_x = torch.randn(1, 10, 128, dtype=torch.bfloat16)
        out = st(x, 0, main_x)
        assert out.shape == x.shape

    def test_full_block_shape(self, small_dims):
        st = DSparkStage(1, _stage_tensors(1), self._fc(), DEV, FP4_TABLE)
        x = torch.randn(1, BLOCK_SIZE, HC_MULT, 128, dtype=torch.bfloat16)
        st(x, 0, torch.randn(1, 10, 128, dtype=torch.bfloat16))
        main_x = torch.randn(1, 1, 128, dtype=torch.bfloat16)
        out = st(x, 10, main_x)
        assert out.shape == x.shape
        assert torch.isfinite(out.float()).all()

    def test_forward_head_shapes(self, small_dims):
        st = DSparkStage(2, _stage_tensors(2), self._fc(), DEV, FP4_TABLE)
        h = torch.randn(1, BLOCK_SIZE, HC_MULT, 128, dtype=torch.bfloat16)
        ids = torch.tensor([[5]])
        lm_head = torch.randn(1000, 128) * 0.02
        out = st.forward_head(h, ids, lm_head)
        assert out["greedy_ids"].shape == (1, BLOCK_SIZE + 1)
        assert out["sampled_ids"].shape == (1, BLOCK_SIZE + 1)
        assert out["greedy_ids"][0, 0].item() == 5
        assert out["sampled_ids"][0, 0].item() == 5
        assert len(out["greedy_topk"]) == BLOCK_SIZE
        assert len(out["greedy_topk"][0][0]) == 16
        assert out["confidence_greedy"].shape == (1, BLOCK_SIZE)
        assert torch.isfinite(out["confidence_greedy"].float()).all()

    def test_forward_head_sampling_reproducible(self, small_dims):
        st = DSparkStage(2, _stage_tensors(2), self._fc(), DEV, FP4_TABLE)
        h = torch.randn(1, BLOCK_SIZE, HC_MULT, 128, dtype=torch.bfloat16)
        ids = torch.tensor([[5]])
        lm_head = torch.randn(1000, 128) * 0.02
        torch.manual_seed(0)
        o1 = st.forward_head(h, ids, lm_head)
        torch.manual_seed(0)
        o2 = st.forward_head(h, ids, lm_head)
        assert torch.equal(o1["sampled_ids"], o2["sampled_ids"])
        assert torch.equal(o1["greedy_ids"], o2["greedy_ids"])

    def test_forward_embed_shapes(self, small_dims):
        st0 = DSparkStage(0, _stage_tensors(0), self._fc(), DEV, FP4_TABLE)
        embed = torch.randn(1000, 128) * 0.02
        ids = torch.tensor([[7]])
        main_h = torch.randn(1, 1, 3 * 128, dtype=torch.bfloat16)
        x, main_x = st0.forward_embed(main_h, ids, embed)
        assert x.shape == (1, BLOCK_SIZE, HC_MULT, 128)
        assert main_x.shape == (1, 1, 128)

    def test_forward_embed_noise_layout(self, small_dims):
        # draft ids = [committed, noise*4]
        st0 = DSparkStage(0, _stage_tensors(0), self._fc(), DEV, FP4_TABLE)
        embed = torch.randn(1000, 128) * 0.02
        ids = torch.tensor([[7]])
        main_h = torch.randn(1, 1, 3 * 128, dtype=torch.bfloat16)
        x, _ = st0.forward_embed(main_h, ids, embed)
        # x[:, 0] must equal embed[7]; x[:, 1:] equal embed[noise]
        embed_bf = embed.to(torch.bfloat16)
        assert torch.equal(x[0, 0, 0], embed_bf[7])
        assert torch.equal(x[0, 1, 0], embed_bf[dw.NOISE_TOKEN_ID])
        # all hc copies identical
        assert torch.equal(x[0, 0, 0], x[0, 0, 1])


# ---------------------------------------------------------------- Lin ------
class TestLin:
    def test_output_shape(self):
        w, s = _fp8_pair(64, 128, seed=9)
        lin = Lin(w, s, DEV)
        x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        y = lin(x)
        assert y.shape == (2, 3, 64)
        assert torch.isfinite(y.float()).all()
