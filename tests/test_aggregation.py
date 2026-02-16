"""Tests for analysis.aggregation."""

from pathlib import Path

from greek_stylometer.analysis.aggregation import aggregate_predictions
from greek_stylometer.schemas import Prediction, WorkPrediction


def _make_pred(
    author_id: str, work_id: str, predicted: int, confidence: float,
    author: str = "Galen", label: int = 1,
) -> Prediction:
    return Prediction(
        text="test",
        author=author,
        label=label,
        predicted=predicted,
        confidence=confidence,
        split="full",
        passage_idx=0,
        author_id=author_id,
        work_id=work_id,
    )


def _write_preds(path: Path, preds: list[Prediction]) -> None:
    with open(path, "w") as f:
        for p in preds:
            f.write(p.to_json() + "\n")


def test_single_chunk_work(tmp_path: Path):
    preds = [_make_pred("tlg0057", "tlg001", predicted=1, confidence=0.9)]
    _write_preds(tmp_path / "preds.jsonl", preds)

    out = tmp_path / "works.jsonl"
    summary = aggregate_predictions(tmp_path / "preds.jsonl", out)

    results = [WorkPrediction.from_json(l) for l in out.read_text().strip().split("\n")]
    assert len(results) == 1
    assert results[0].predicted == 1
    assert results[0].confidence == 0.9
    assert results[0].n_chunks == 1
    assert "1 works" in summary


def test_multi_chunk_averaging(tmp_path: Path):
    """Three chunks for one work: two positive (0.8, 0.9), one negative (0.7).
    Positive-class confidences: 0.8, 0.9, 0.3 → mean 0.667 → predicted 1."""
    preds = [
        _make_pred("tlg0057", "tlg001", predicted=1, confidence=0.8),
        _make_pred("tlg0057", "tlg001", predicted=1, confidence=0.9),
        _make_pred("tlg0057", "tlg001", predicted=0, confidence=0.7),
    ]
    _write_preds(tmp_path / "preds.jsonl", preds)

    out = tmp_path / "works.jsonl"
    aggregate_predictions(tmp_path / "preds.jsonl", out)

    results = [WorkPrediction.from_json(l) for l in out.read_text().strip().split("\n")]
    assert len(results) == 1
    assert results[0].predicted == 1
    assert results[0].n_chunks == 3
    assert 0.6 < results[0].confidence < 0.7


def test_negative_work(tmp_path: Path):
    """All chunks predict negative with high confidence → work is negative."""
    preds = [
        _make_pred("tlg0555", "tlg001", predicted=0, confidence=0.95, author="Other", label=0),
        _make_pred("tlg0555", "tlg001", predicted=0, confidence=0.85, author="Other", label=0),
    ]
    _write_preds(tmp_path / "preds.jsonl", preds)

    out = tmp_path / "works.jsonl"
    aggregate_predictions(tmp_path / "preds.jsonl", out)

    results = [WorkPrediction.from_json(l) for l in out.read_text().strip().split("\n")]
    assert len(results) == 1
    assert results[0].predicted == 0
    assert results[0].confidence > 0.85


def test_multiple_works(tmp_path: Path):
    preds = [
        _make_pred("tlg0057", "tlg001", predicted=1, confidence=0.9),
        _make_pred("tlg0057", "tlg002", predicted=1, confidence=0.8),
        _make_pred("tlg0555", "tlg001", predicted=0, confidence=0.9, author="Other", label=0),
    ]
    _write_preds(tmp_path / "preds.jsonl", preds)

    out = tmp_path / "works.jsonl"
    summary = aggregate_predictions(tmp_path / "preds.jsonl", out)

    results = [WorkPrediction.from_json(l) for l in out.read_text().strip().split("\n")]
    assert len(results) == 3
    assert "3 works" in summary


def test_empty_file(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    out = tmp_path / "works.jsonl"
    summary = aggregate_predictions(path, out)
    assert "No predictions found" in summary
