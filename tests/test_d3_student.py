import hashlib

import pytest
import torch

from mosaicvm.bench.d3_student import fit, load_captures, predict, residual_metrics


def test_train_only_future_alignment_and_recursive_no_oracle(tmp_path):
    torch.set_num_threads(2)
    spec = {"splits": {"train": ["a"], "val": ["b"], "test": ["c"]},
            "prompts": [{"category": cat, "prompt": cat} for cat in "abc"]}
    for cat, split in (("a", "train"), ("b", "val"), ("c", "test")):
        h = torch.arange(6).view(6, 1, 1).expand(-1, 4, 4096).to(torch.bfloat16)
        torch.save({"category": cat, "split": split, "S": 2,
                    "tokens": list(range(7)), "gen": list(range(2, 7)),
                    "h40": h, "h42": h, "states": {40: h[1:]},
                    "state_position_start": 1, "prompt": cat,
                    "prompt_sha256": hashlib.sha256(cat.encode()).hexdigest(),
                    "teacher_class": "SOURCE_CLASS_TORCH_SHIM",
                    "script_sha256": "test", "snapshot_revision": "test"},
                   tmp_path / f"{cat}.pt")
    captures = load_captures(tmp_path, spec, ["a", "b"], horizon=4)
    with pytest.raises(ValueError, match="Train/val"):
        load_captures(tmp_path, spec, horizon=4)
    with pytest.raises(ValueError, match="train captures only"):
        fit(captures, horizon=4)
    model = fit(captures[:1], horizon=4, rank=1)
    assert model["training_pairs_by_horizon"] == [4, 3, 2, 1]
    root = captures[1]["h40"][1]
    predicted = predict(model, root, 4)
    chunked = predict(model, root, 4, chunk=2)
    assert predicted.shape == (4, 4, 4096)
    assert predicted.dtype == torch.bfloat16
    assert torch.equal(predicted, captures[1]["h40"][2:6])
    assert torch.equal(chunked, predicted)
    # Changing held-out future data does not change either deployment predictor.
    captures[1]["h40"][2:].fill_(900)
    assert torch.equal(predict(model, root, 4, chunk=2), chunked)
    assert residual_metrics(predicted, predicted)["relative_l2_by_horizon"] == [0] * 4
    c = torch.load(tmp_path / "b.pt", weights_only=True)
    c["tokens"].pop()  # A missing terminal target label must be rejected.
    torch.save(c, tmp_path / "b.pt")
    with pytest.raises(ValueError, match="alignment"):
        load_captures(tmp_path, spec, ["a", "b"], horizon=4)
