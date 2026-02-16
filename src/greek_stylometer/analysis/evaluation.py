"""Evaluation metrics for predictions JSONL.

Reads a predictions file and computes overall and per-author
accuracy, precision, recall, and F1.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

from greek_stylometer.schemas import Prediction

logger = logging.getLogger(__name__)


def _read_predictions(path: Path) -> list[Prediction]:
    """Read predictions from a JSONL file."""
    predictions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(Prediction.from_json(line))
    return predictions


def _format_summary(result: dict) -> str:
    """Format evaluation results as a human-readable summary."""
    lines = [f"Total: {result['total']} predictions"]
    overall = result["overall"]
    if overall:
        lines.append(
            f"Overall: accuracy={overall['accuracy']:.3f}  "
            f"precision={overall['precision']:.3f}  "
            f"recall={overall['recall']:.3f}  "
            f"f1={overall['f1']:.3f}"
        )
    for author, m in result.get("per_author", {}).items():
        lines.append(
            f"  {author}: n={m['count']}  "
            f"accuracy={m['accuracy']:.3f}  "
            f"avg_confidence={m['avg_confidence']:.3f}"
        )
    return "\n".join(lines)


def evaluate(
    predictions_path: Path,
    output_path: Path | None = None,
) -> str:
    """Compute evaluation metrics from a predictions JSONL file.

    Args:
        predictions_path: Path to predictions JSONL.
        output_path: If given, write detailed results as JSON.

    Returns:
        A formatted summary string.
    """
    predictions = _read_predictions(predictions_path)
    if not predictions:
        logger.warning("No predictions found in %s", predictions_path)
        return "No predictions found."

    labels = [p.label for p in predictions]
    predicted = [p.predicted for p in predictions]

    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary"
    )
    accuracy = sum(l == p for l, p in zip(labels, predicted)) / len(labels)

    overall = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    # Per-author breakdown
    by_author: dict[str, list[Prediction]] = defaultdict(list)
    for pred in predictions:
        by_author[pred.author].append(pred)

    per_author = {}
    for author, author_preds in sorted(by_author.items()):
        a_labels = [p.label for p in author_preds]
        a_predicted = [p.predicted for p in author_preds]
        a_accuracy = sum(l == p for l, p in zip(a_labels, a_predicted)) / len(a_labels)
        a_avg_conf = sum(p.confidence for p in author_preds) / len(author_preds)

        per_author[author] = {
            "count": len(author_preds),
            "accuracy": float(a_accuracy),
            "avg_confidence": float(a_avg_conf),
        }

    result = {
        "overall": overall,
        "per_author": per_author,
        "total": len(predictions),
    }

    logger.info(
        "Evaluated %d predictions: accuracy=%.3f, f1=%.3f",
        len(predictions),
        accuracy,
        f1,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Wrote evaluation to %s", output_path)

    return _format_summary(result)
