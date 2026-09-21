import pytest
from mosaicvm.bench.stage1_record import prompt_splits, capture_span


def test_prompt_split_and_future_state_boundary():
    spec = {'splits': {'train': ['a'], 'val': ['b']},
            'prompts': [{'category': 'a', 'prompt': 'hello'},
                        {'category': 'b', 'prompt': 'different'}]}
    assert prompt_splits(spec) == {'a': 'train', 'b': 'val'}
    # K8 requires root + eight future states, with a terminal next-token label.
    start, end = capture_span(5, 9)
    tokens = list(range(14))
    assert start == 4 and end == 13
    assert len(tokens[start + 1:end + 1]) == 9
    assert tokens[end] == 13  # last captured input state's label exists
    spec['prompts'][1]['prompt'] = ' HELLO '
    with pytest.raises(ValueError, match='Duplicate prompt'):
        prompt_splits(spec)
    spec['splits']['val'].append('a')
    with pytest.raises(ValueError, match='multiple splits'):
        prompt_splits(spec)
    with pytest.raises(ValueError):
        capture_span(0, 9)
