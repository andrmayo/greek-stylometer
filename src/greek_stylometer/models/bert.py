"""BERT fine-tuning and inference for binary authorship attribution.

Reads corpus JSONL, creates binary labels (target author → 1, all
others → 0), and trains using the HuggingFace Trainer API.  Produces
a saved model directory and test-set predictions in JSONL.
"""

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from greek_stylometer.corpus.jsonl import read_corpus
from greek_stylometer.models.config import TrainConfig
from greek_stylometer.schemas import Prediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passages_to_dataset(
    passages: list, positive_author_id: str
) -> Dataset:
    """Convert Passage records to a HuggingFace Dataset with binary labels."""
    return Dataset.from_dict(
        {
            "text": [p.text for p in passages],
            "label": [1 if p.author_id == positive_author_id else 0 for p in passages],
            "author": [p.author for p in passages],
            "passage_idx": [p.passage_idx for p in passages],
            "author_id": [p.author_id for p in passages],
            "work_id": [p.work_id for p in passages],
        }
    )


def _split_dataset(
    dataset: Dataset, config: TrainConfig
) -> tuple[Dataset, Dataset, Dataset]:
    """Shuffle and split into train / dev / test."""
    shuffled = dataset.shuffle(seed=config.seed)
    n = len(shuffled)
    train_end = round(n * config.train_ratio)
    dev_end = train_end + round(n * config.dev_ratio)

    train_ds = shuffled.select(range(train_end))
    dev_ds = shuffled.select(range(train_end, dev_end))
    test_ds = shuffled.select(range(dev_end, n))
    return train_ds, dev_ds, test_ds


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Precision / recall / F1 for binary classification."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _write_predictions(
    trainer: Trainer,
    tokenized_dataset: Dataset,
    raw_dataset: Dataset,
    split: str,
    output_path: Path,
) -> int:
    """Run prediction and write results to JSONL. Returns count."""
    # cast needed: datasets.Dataset doesn't match the generic
    # Dataset[Unknown] that the transformers Trainer stubs expect.
    result = trainer.predict(cast(Any, tokenized_dataset))
    logits = result.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted_labels = np.argmax(logits, axis=-1)
    probs = _softmax(logits)
    confidences = probs.max(axis=-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(raw_dataset)):
            pred = Prediction(
                text=raw_dataset[i]["text"],
                author=raw_dataset[i]["author"],
                label=raw_dataset[i]["label"],
                predicted=int(predicted_labels[i]),
                confidence=float(confidences[i]),
                split=split,
                passage_idx=raw_dataset[i]["passage_idx"],
                author_id=raw_dataset[i]["author_id"],
                work_id=raw_dataset[i]["work_id"],
            )
            f.write(pred.to_json() + "\n")
            count += 1

    logger.info("Wrote %d predictions to %s", count, output_path)
    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    corpus_path: Path,
    output_dir: Path,
    positive_author_id: str,
    cfg: TrainConfig = TrainConfig(),
) -> Path:
    """Train a binary BERT classifier.

    Reads a corpus JSONL, creates binary labels (positive_author_id → 1,
    all others → 0), trains with HuggingFace Trainer, saves the model,
    and writes test-set predictions to ``output_dir/test_predictions.jsonl``.

    Args:
        corpus_path: Input corpus JSONL file.
        output_dir: Directory to save model, checkpoints, and predictions.
        positive_author_id: Author ID for the positive class (e.g. "tlg0057").
        cfg: Training configuration. Uses TrainConfig defaults if not provided.

    Returns:
        Path to the saved model directory.
    """

    # Load data
    passages = list(read_corpus(corpus_path))
    logger.info("Loaded %d passages from %s", len(passages), corpus_path)

    dataset = _passages_to_dataset(passages, positive_author_id)
    pos_count = sum(1 for label in dataset["label"] if label == 1)
    logger.info(
        "Labels: %d positive (%s), %d negative",
        pos_count,
        positive_author_id,
        len(dataset) - pos_count,
    )

    train_ds, dev_ds, test_ds = _split_dataset(dataset, cfg)
    logger.info(
        "Split: %d train, %d dev, %d test", len(train_ds), len(dev_ds), len(test_ds)
    )

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2
    )

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=cfg.max_length
        )

    train_tok = train_ds.map(tokenize, batched=True)
    dev_tok = dev_ds.map(tokenize, batched=True)
    test_tok = test_ds.map(tokenize, batched=True)

    # Train
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_epochs,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    logger.info("Saved model to %s", model_dir)

    # Evaluate on test set and write predictions
    predictions_path = output_dir / "test_predictions.jsonl"
    _write_predictions(trainer, test_tok, test_ds, "test", predictions_path)

    return model_dir


def predict(
    corpus_path: Path,
    model_dir: Path,
    output_path: Path,
    positive_author_id: str,
    max_length: int = 512,
    batch_size: int = 64,
) -> int:
    """Run inference with a saved model on a corpus JSONL.

    Args:
        corpus_path: Input corpus JSONL file.
        model_dir: Directory containing the saved model.
        output_path: Path to write predictions JSONL.
        positive_author_id: Author ID for the positive class.
        max_length: Maximum token length for tokenization.
        batch_size: Batch size for inference.

    Returns:
        Number of predictions written.
    """
    passages = list(read_corpus(corpus_path))
    dataset = _passages_to_dataset(passages, positive_author_id)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=max_length
        )

    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="/tmp/greek-stylometer-predict",
        per_device_eval_batch_size=batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    return _write_predictions(trainer, tokenized, dataset, "full", output_path)
