"""Tests for models.bert helper functions."""

import numpy as np
import pytest
from transformers import EvalPrediction

from greek_stylometer.models.bert import (
    _compute_metrics,
    _passages_to_dataset,
    _softmax,
    _split_dataset,
)
from greek_stylometer.models.config import TrainConfig
from greek_stylometer.schemas import Passage


def _make_passage(author_id: str, text: str = "test") -> Passage:
    return Passage(
        text=text,
        author=author_id,
        author_id=author_id,
        work_id="w001",
        passage_idx=0,
        source="test",
    )


class TestPassagesToDataset:
    def test_binary_labels(self):
        passages = [
            _make_passage("tlg0057"),
            _make_passage("tlg0099"),
            _make_passage("tlg0057"),
        ]
        ds = _passages_to_dataset(lambda: (p for p in passages), "tlg0057")
        assert ds["label"] == [1, 0, 1]

    def test_preserves_text_and_author(self):
        passages = [_make_passage("tlg0057", text="hello")]
        ds = _passages_to_dataset(lambda: (p for p in passages), "tlg0057")
        assert ds["text"] == ["hello"]
        assert ds["author"] == ["tlg0057"]


class TestSplitDataset:
    def test_split_sizes(self):
        passages = [_make_passage(f"a{i}") for i in range(100)]
        ds = _passages_to_dataset(lambda: (p for p in passages), "a0")
        config = TrainConfig(train_ratio=0.8, dev_ratio=0.1, seed=42)
        train, dev, test = _split_dataset(ds, config)
        assert len(train) == 80
        assert len(dev) == 10
        assert len(test) == 10

    def test_deterministic_with_seed(self):
        passages = [_make_passage(f"a{i}") for i in range(50)]
        ds = _passages_to_dataset(lambda: (p for p in passages), "a0")
        config = TrainConfig(seed=42)
        train1, _, _ = _split_dataset(ds, config)
        train2, _, _ = _split_dataset(ds, config)
        assert train1["author"] == train2["author"]


class TestSoftmax:
    def test_sums_to_one(self):
        logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        probs = _softmax(logits)
        np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0])

    def test_numerically_stable(self):
        logits = np.array([[1000.0, 1001.0]])
        probs = _softmax(logits)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(probs.sum(axis=-1), [1.0])


class TestComputeMetrics:
    def test_perfect_predictions(self):
        logits = np.array([[0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
        labels = np.array([1, 0, 1])
        result = _compute_metrics(EvalPrediction(predictions=logits, label_ids=labels))
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_wrong_predictions(self):
        logits = np.array([[10.0, 0.0], [0.0, 10.0]])
        labels = np.array([1, 0])
        result = _compute_metrics(EvalPrediction(predictions=logits, label_ids=labels))
        assert result["f1"] == pytest.approx(0.0)
