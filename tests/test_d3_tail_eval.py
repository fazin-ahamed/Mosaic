import pytest
import torch

from mosaicvm.bench.d3_tail_eval import score


def test_exact_root_excluded_terminal_scored_and_first_reject_stops():
    tokens = [0, 1, 2, 3, 4, 5]
    logits = torch.zeros(5, 20)
    # S=2: row 1 is the exact root; uncertain rows 2..4 predict 3..5.
    logits[1, 19] = 10  # Deliberately wrong root must not enter this score.
    for row, label in ((2, 3), (3, 4), (4, 5)):
        logits[row, label] = 10
    assert score(logits, tokens, 2, [1, 3])["3"]["E_A"] == 4
    logits[3, 19] = 20
    result = score(logits, tokens, 2, [3])["3"]
    assert result["E_A"] == 2
    assert result["Sk"] == [1, 0, 0]
    assert result["topk_true_path_conditioned"]["2"] == 1
    with pytest.raises(ValueError, match="terminal"):
        score(logits, tokens[:-1], 2, [3])
