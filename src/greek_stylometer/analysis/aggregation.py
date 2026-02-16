"""Aggregate chunk-level predictions into work-level predictions.

Groups predictions by (author_id, work_id), averages confidence
scores across chunks, and derives a single predicted label per work.
"""

from collections import defaultdict
from pathlib import Path

from greek_stylometer.schemas import Prediction, WorkPrediction


def aggregate_predictions(
    predictions_path: Path,
    output_path: Path,
) -> str:
    """Read chunk predictions and write work-level predictions.

    For each work, averages the confidence of chunks that predicted
    class 1 (positive). The work-level prediction is 1 if the mean
    positive-class confidence exceeds 0.5, else 0.

    Args:
        predictions_path: Input chunk-level predictions JSONL.
        output_path: Output work-level predictions JSONL.

    Returns:
        A formatted summary string.
    """
    # Group predictions by work
    works: dict[tuple[str, str], list[Prediction]] = defaultdict(list)
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred = Prediction.from_json(line)
            works[(pred.author_id, pred.work_id)].append(pred)

    if not works:
        return "No predictions found."

    # Aggregate and write
    results: list[WorkPrediction] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for (author_id, work_id), chunks in sorted(works.items()):
            # For each chunk: confidence is for the predicted class.
            # Convert to positive-class confidence for averaging.
            pos_confidences = []
            for c in chunks:
                if c.predicted == 1:
                    pos_confidences.append(c.confidence)
                else:
                    pos_confidences.append(1.0 - c.confidence)

            mean_pos = sum(pos_confidences) / len(pos_confidences)
            predicted = 1 if mean_pos > 0.5 else 0
            confidence = mean_pos if predicted == 1 else 1.0 - mean_pos

            wp = WorkPrediction(
                author=chunks[0].author,
                author_id=author_id,
                work_id=work_id,
                predicted=predicted,
                confidence=confidence,
                n_chunks=len(chunks),
                label=chunks[0].label,
            )
            results.append(wp)
            f.write(wp.to_json() + "\n")

    return _format_summary(results)


def _format_summary(results: list[WorkPrediction]) -> str:
    """Format work-level predictions as a human-readable summary."""
    lines = [f"Aggregated {len(results)} works:"]
    lines.append("")

    for wp in results:
        verdict = "POSITIVE" if wp.predicted == 1 else "NEGATIVE"
        correct = "correct" if wp.predicted == wp.label else "WRONG"
        lines.append(
            f"  {wp.author_id}/{wp.work_id}: {verdict} "
            f"(confidence={wp.confidence:.3f}, chunks={wp.n_chunks}, {correct})"
        )

    # Overall accuracy
    n_correct = sum(1 for wp in results if wp.predicted == wp.label)
    lines.append("")
    lines.append(f"Work-level accuracy: {n_correct}/{len(results)}")

    return "\n".join(lines)
