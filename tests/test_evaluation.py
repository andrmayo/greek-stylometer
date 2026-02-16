"""Tests for analysis.evaluation."""

import json
from pathlib import Path

from greek_stylometer.analysis.evaluation import evaluate
from greek_stylometer.schemas import Prediction


def _make_prediction(author: str, label: int, predicted: int, confidence: float = 0.9) -> Prediction:
    return Prediction(
        text="test",
        author=author,
        label=label,
        predicted=predicted,
        confidence=confidence,
        split="test",
        passage_idx=0,
    )


def _write_predictions(path: Path, predictions: list[Prediction]) -> None:
    with open(path, "w") as f:
        for p in predictions:
            f.write(p.to_json() + "\n")


def test_perfect_predictions(tmp_path: Path):
    preds = [
        _make_prediction("Galen", 1, 1, 0.95),
        _make_prediction("Galen", 1, 1, 0.90),
        _make_prediction("Other", 0, 0, 0.85),
    ]
    path = tmp_path / "preds.jsonl"
    _write_predictions(path, preds)

    summary = evaluate(path)
    assert "accuracy=1.000" in summary
    assert "f1=1.000" in summary


def test_per_author_breakdown(tmp_path: Path):
    preds = [
        _make_prediction("Galen", 1, 1),
        _make_prediction("Clement", 0, 0),
        _make_prediction("Clement", 0, 1),  # wrong
    ]
    path = tmp_path / "preds.jsonl"
    _write_predictions(path, preds)

    summary = evaluate(path)
    assert "Galen" in summary
    assert "Clement" in summary


def test_writes_json_output(tmp_path: Path):
    preds = [
        _make_prediction("Galen", 1, 1),
        _make_prediction("Other", 0, 0),
    ]
    path = tmp_path / "preds.jsonl"
    _write_predictions(path, preds)

    out = tmp_path / "eval.json"
    evaluate(path, out)

    result = json.loads(out.read_text())
    assert result["total"] == 2
    assert "overall" in result
    assert "per_author" in result


def test_empty_file(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    summary = evaluate(path)
    assert "No predictions found" in summary
