"""LIME explanations for binary authorship attribution.

Generates local interpretable explanations for model predictions
using LIME (Local Interpretable Model-agnostic Explanations).
Outputs explanations as JSONL and optionally as HTML files.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from greek_stylometer.schemas import Explanation, FeatureWeight, Prediction

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


def _make_classifier_fn(model_dir: Path, max_length: int = 512):
    """Build a classifier function for LIME from a saved model.

    Returns a callable that takes a list of strings and returns
    an (n, 2) numpy array of class probabilities.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    def classifier_fn(texts: list[str]) -> np.ndarray:
        inputs = tokenizer(
            texts, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=-1).numpy()

    return classifier_fn


def explain(
    predictions_path: Path,
    model_dir: Path,
    output_path: Path,
    num_features: int = 10,
    num_samples: int = 5000,
    class_names: tuple[str, str] = ("non-target", "target"),
    max_length: int = 512,
    html_dir: Path | None = None,
    seed: int = 1234,
) -> int:
    """Generate LIME explanations for predictions.

    Args:
        predictions_path: Predictions JSONL to explain.
        model_dir: Directory containing the saved model.
        output_path: Path to write explanations JSONL.
        num_features: Number of top features per explanation.
        num_samples: Number of perturbed samples LIME generates.
        class_names: Display names for (negative, positive) classes.
        max_length: Max token length for the model.
        html_dir: If given, save HTML visualizations here.
        seed: Random seed for LIME.

    Returns:
        Number of explanations written.
    """
    predictions = _read_predictions(predictions_path)
    if not predictions:
        logger.warning("No predictions found in %s", predictions_path)
        return 0

    logger.info("Explaining %d predictions with LIME", len(predictions))

    classifier_fn = _make_classifier_fn(model_dir, max_length)
    explainer = LimeTextExplainer(
        class_names=list(class_names), random_state=seed
    )

    if html_dir is not None:
        html_dir.mkdir(parents=True, exist_ok=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            exp = explainer.explain_instance(
                pred.text,
                classifier_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=[0, 1],
            )

            # Extract top features for the predicted class
            feature_weights = [
                FeatureWeight(token=token, weight=weight)
                for token, weight in exp.as_list(label=pred.predicted)
            ]

            explanation = Explanation(
                passage_idx=pred.passage_idx,
                text=pred.text,
                predicted=pred.predicted,
                confidence=pred.confidence,
                top_features=feature_weights,
            )
            f.write(explanation.to_json() + "\n")

            if html_dir is not None:
                exp.save_to_file(str(html_dir / f"explanation_{pred.passage_idx}.html"))

            count += 1
            if count % 10 == 0:
                logger.info("Explained %d / %d", count, len(predictions))

    logger.info("Wrote %d explanations to %s", count, output_path)
    return count
