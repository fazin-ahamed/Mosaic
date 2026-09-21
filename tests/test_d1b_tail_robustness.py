"""CPU regression for D1-B boundary accounting and nested state reset."""
import pytest
import torch

from mosaicvm.bench.d1b_tail_robustness import prefix_metrics, reset_decode_state


def test_prefix_horizons_exclude_root_and_include_last_decision():
    for k in (1, 8, 16, 32):
        assert prefix_metrics([True] * k, k)['E_A'] == k + 1
        first_miss = prefix_metrics([False] + [True] * (k - 1), k)
        assert first_miss['E_A'] == 1  # exact root must not inflate this
        assert first_miss['Sk'] == [0] * k
        last_miss = prefix_metrics([True] * (k - 1) + [False], k)
        assert last_miss['E_A'] == k
        assert last_miss['Sk'] == [1] * (k - 1) + [0]
        assert last_miss['E_A'] == 1 + sum(last_miss['Sk'])
    with pytest.raises(ValueError):
        prefix_metrics([True] * 31, 32)


def test_reset_reaches_both_compressors_and_preserves_constants():
    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.compressor = torch.nn.Module()
    model.attn.indexer = torch.nn.Module()
    model.attn.indexer.compressor = torch.nn.Module()
    for module in model.modules():
        for name in ('kv_cache', 'kv_state', 'score_state', 'freqs_cis'):
            module.register_buffer(name, torch.full((4,), 7.0))
    # Distinct previous-chain poison must produce identical initial state.
    states = []
    for poison in (123.0, -456.0):
        for name, buf in model.named_buffers():
            if not name.endswith('freqs_cis'):
                buf.fill_(poison)
        reset_decode_state(model)
        states.append({name: buf.clone() for name, buf in model.named_buffers()})
    for name, buf in states[0].items():
        assert torch.equal(buf, states[1][name])
        expected = 7 if name.endswith('freqs_cis') else (
            float('-inf') if name.endswith('score_state') else 0)
        assert (buf == expected).all()


def test_prompt_ci_does_not_invent_precision_for_one_prompt():
    from mosaicvm.bench.d1b_tail_robustness import prompt_mean_ci
    assert prompt_mean_ci([33]) is None
    assert prompt_mean_ci([33, 33, 33]) == [33, 33]
    low, high = prompt_mean_ci([1, 9, 17, 33])
    assert 1 <= low < high <= 33
