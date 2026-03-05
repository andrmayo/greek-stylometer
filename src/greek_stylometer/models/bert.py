"""BERT fine-tuning and inference for binary authorship attribution.

Reads corpus JSONL, creates binary labels (target author → 1, all
others → 0), and trains using the HuggingFace Trainer API.  Produces
a saved model directory and test-set predictions in JSONL.
"""

import copy
import dataclasses
import json
import logging
import math
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from shutil import rmtree
from typing import Any, Callable, Iterator, cast

import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import PredictionOutput

from greek_stylometer.corpus.jsonl import read_corpus, stream_passage_to_dict
from greek_stylometer.models.config import TrainConfig
from greek_stylometer.schemas import MonteCarloAggregation, Passage, Prediction

logger = logging.getLogger(__name__)


TRAIN_SET_LOSS_FILENAME = "train_set_loss.jsonl"
DEV_SET_EVAL_FILENAME = "dev_set_evals.jsonl"
TEST_SET_EVAL_FILENAME = "test_set_evals.jsonl"
TEST_SET_PREDICTION_FILENAME = "test_predictions.jsonl"
TEST_SET_PREDICTION_METRICS_FILENAME = "test_metrics.jsonl"
MC_CV_STATS_FILENAME = "mc_cv_stats.json"
MC_CV_FINAL_MODEL_DIR = "mc_cv_final_model/"

# ---------------------------------------------------------------------------
# Training Callbacks
# ---------------------------------------------------------------------------


class TestEvalCallback(TrainerCallback):
    def __init__(
        self,
        test_dataset: Dataset,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "test",
    ):
        self.test_dataset = test_dataset
        self.ignore_keys = ignore_keys
        self.metric_key_prefix = metric_key_prefix
        self.history: list[dict[str, int | float | None]] = []
        self._evaluating: bool = False
        self.trainer: Trainer | None = None

    def _evaluate_on_testset(
        self,
        trainer: Trainer,
    ) -> dict[str, float]:
        return trainer.evaluate(
            cast(Any, self.test_dataset),
            self.ignore_keys,
            self.metric_key_prefix,
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        _, _, _ = args, control, kwargs
        # Recursion guard
        if self._evaluating:
            return
        if not isinstance(self.trainer, Trainer):
            logger.warning(
                "skipping test set eval: for TestEvalCallback to work, use callback.trainer = trainer, where trainer is the trainer object with eval set up"
            )
            return

        self._evaluating = True
        try:
            test_metrics = self._evaluate_on_testset(self.trainer)
            self.history.append(
                {"step": state.global_step, "epoch": state.epoch, **test_metrics}
            )
        finally:
            self._evaluating = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PassageFactory = Callable[[], Iterator[Passage]]


def path_to_dataset(path: Path, positive_author_id: str) -> Dataset:
    factory: PassageFactory = partial(read_corpus, path)
    return _passages_to_dataset(factory, positive_author_id)


def _passages_to_dataset(
    passage_factory: PassageFactory, positive_author_id: str
) -> Dataset:
    """Convert Passage records to a HuggingFace Dataset with binary labels."""

    def gen(pf: PassageFactory, pos_auth_id: str):
        passages = pf()  # Created within worker for Dataset multiprocessing
        yield from stream_passage_to_dict(passages, pos_auth_id)

    return cast(
        Dataset,
        Dataset.from_generator(
            gen, gen_kwargs={"pf": passage_factory, "pos_auth_id": positive_author_id}
        ),
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
    dev_ds = shuffled.select(range(train_end, dev_end)) if dev_end > train_end else shuffled.select([])
    test_ds = shuffled.select(range(dev_end, n)) if n > dev_end else shuffled.select([])
    return train_ds, dev_ds, test_ds


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Precision / recall / F1 for binary classification."""
    # Note: TrainerArguments must have include_for_metrics=["loss"]
    logits, labels, *_ = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    # loss gets computed automatically by trainer
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _write_predictions(
    trainer: Trainer,
    tokenized_dataset: Dataset,
    raw_dataset: Dataset,
    split: str,
    output_path: Path,
) -> tuple[int, PredictionOutput]:
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
    return count, result


def _write_train_dev_test_history(
    trainer: Trainer, test_eval_callback: TestEvalCallback, output_dir: Path
):
    with open(output_dir / TRAIN_SET_LOSS_FILENAME, "w") as f:
        train_loss_logs = [
            x
            for x in trainer.state.log_history
            if "loss" in x and not any(k.startswith("eval_") for k in x)
        ]
        for line in train_loss_logs:
            json.dump(line, f)
            f.write("\n")

    with open(output_dir / DEV_SET_EVAL_FILENAME, "w") as f:
        dev_evals = [
            x
            for x in trainer.state.log_history
            if any(k.startswith("eval_") for k in x)
        ]
        for line in dev_evals:
            json.dump(line, f)
            f.write("\n")

    with open(output_dir / TEST_SET_EVAL_FILENAME, "w") as f:
        for line in test_eval_callback.history:
            json.dump(line, f)
            f.write("\n")


def _get_mc_cv_metrics(output_paths: list[Path]) -> MonteCarloAggregation:
    """Create a MonteCarloAggregation object and write it as json file to aggregation_path."""
    avg_best_step = None
    avg_best_epoch = None
    best_aggregate_step = None
    best_aggregate_epoch = None
    avg_test_loss_for_best_step = None
    avg_test_F1_for_best_step = None
    avg_test_precision_for_best_step = None
    avg_test_recall_for_best_step = None
    avg_dev_loss_for_best_step = None
    avg_dev_F1_for_best_step = None
    avg_dev_precision_for_best_step = None
    avg_dev_recall_for_best_step = None

    best_per_run_test_steps = []
    best_per_run_test_epochs = []
    best_agg_test_losses = []
    best_agg_test_F1s = []
    best_agg_dev_losses = []
    best_agg_dev_F1s = []

    dev_loss_over_runs = defaultdict(float)

    n_outputs = len(output_paths)

    for op in output_paths:
        # Aggregate metrics from test set eval from the run's best model
        with open(op / TEST_SET_PREDICTION_METRICS_FILENAME, "r") as f:
            output_dict = json.load(f)
            avg_best_step = (
                avg_best_step + output_dict["step"]
                if avg_best_step is not None
                else output_dict["step"]
            )
            avg_best_epoch = (
                avg_best_epoch + output_dict["epoch"]
                if avg_best_epoch is not None
                else output_dict["epoch"]
            )
            best_per_run_test_steps.append(output_dict["step"])
            best_per_run_test_epochs.append(output_dict["epoch"])
            # Aggregate metrics from dev set evals for each checkpoint
            with open(op / DEV_SET_EVAL_FILENAME, "r") as f:
                for line in f:
                    output_dict = json.loads(line)
                    dev_loss_over_runs[output_dict["step"]] += output_dict["eval_loss"]

    best_agg_loss = float("inf")
    for k, v in dev_loss_over_runs.items():
        if v < best_agg_loss:
            best_aggregate_step = k
            best_agg_loss = v

    map_steps_to_epochs = {}
    # Now use best_aggregate_step to figure out what the avg metrics at this step are
    for op in output_paths:
        with open(op / TEST_SET_EVAL_FILENAME, "r") as f:
            for line in f:
                output_dict = json.loads(line)
                map_steps_to_epochs[output_dict["step"]] = output_dict["epoch"]
        with open(op / TEST_SET_EVAL_FILENAME, "r") as f:
            for line in f:
                output_dict = json.loads(line)
                if output_dict["step"] == best_aggregate_step:
                    avg_test_loss_for_best_step = (
                        avg_test_loss_for_best_step + output_dict["test_loss"]
                        if avg_test_loss_for_best_step is not None
                        else output_dict["test_loss"]
                    )
                    avg_test_F1_for_best_step = (
                        avg_test_F1_for_best_step + output_dict["test_f1"]
                        if avg_test_F1_for_best_step is not None
                        else output_dict["test_f1"]
                    )
                    avg_test_precision_for_best_step = (
                        avg_test_precision_for_best_step + output_dict["test_precision"]
                        if avg_test_precision_for_best_step is not None
                        else output_dict["test_precision"]
                    )
                    avg_test_recall_for_best_step = (
                        avg_test_recall_for_best_step + output_dict["test_recall"]
                        if avg_test_recall_for_best_step is not None
                        else output_dict["test_recall"]
                    )
                    best_agg_test_losses.append(output_dict["test_loss"])
                    best_agg_test_F1s.append(output_dict["test_f1"])
                    break
        with open(op / DEV_SET_EVAL_FILENAME, "r") as f:
            for line in f:
                output_dict = json.loads(line)
                if output_dict["step"] == best_aggregate_step:
                    avg_dev_loss_for_best_step = (
                        avg_dev_loss_for_best_step + output_dict["eval_loss"]
                        if avg_dev_loss_for_best_step is not None
                        else output_dict["eval_loss"]
                    )
                    avg_dev_F1_for_best_step = (
                        avg_dev_F1_for_best_step + output_dict["eval_f1"]
                        if avg_dev_F1_for_best_step is not None
                        else output_dict["eval_f1"]
                    )
                    avg_dev_precision_for_best_step = (
                        avg_dev_precision_for_best_step + output_dict["eval_precision"]
                        if avg_dev_precision_for_best_step is not None
                        else output_dict["eval_precision"]
                    )
                    avg_dev_recall_for_best_step = (
                        avg_dev_recall_for_best_step + output_dict["eval_recall"]
                        if avg_dev_recall_for_best_step is not None
                        else output_dict["eval_recall"]
                    )
                    best_agg_dev_losses.append(output_dict["eval_loss"])
                    best_agg_dev_F1s.append(output_dict["eval_f1"])

    best_aggregate_epoch = map_steps_to_epochs.get(best_aggregate_step)

    def get_average_with_fallback(summation: float | int | None, count: int) -> float:
        return summation / count if summation is not None else -1.0

    avg_best_step = get_average_with_fallback(avg_best_step, n_outputs)
    avg_best_epoch = get_average_with_fallback(avg_best_epoch, n_outputs)
    avg_test_loss_for_best_step = get_average_with_fallback(
        avg_test_loss_for_best_step, n_outputs
    )
    avg_test_F1_for_best_step = get_average_with_fallback(
        avg_test_F1_for_best_step, n_outputs
    )
    avg_test_precision_for_best_step = get_average_with_fallback(
        avg_test_precision_for_best_step, n_outputs
    )
    avg_test_recall_for_best_step = get_average_with_fallback(
        avg_test_recall_for_best_step, n_outputs
    )
    avg_dev_loss_for_best_step = get_average_with_fallback(
        avg_dev_loss_for_best_step, n_outputs
    )
    avg_dev_F1_for_best_step = get_average_with_fallback(
        avg_dev_F1_for_best_step, n_outputs
    )
    avg_dev_precision_for_best_step = get_average_with_fallback(
        avg_dev_precision_for_best_step, n_outputs
    )
    avg_dev_recall_for_best_step = get_average_with_fallback(
        avg_dev_recall_for_best_step, n_outputs
    )

    return MonteCarloAggregation(
        avg_best_step=int(avg_best_step),
        avg_best_epoch=int(avg_best_epoch),
        best_aggregate_step=int(best_aggregate_step)
        if best_aggregate_step is not None
        else -1,
        best_aggregate_epoch=int(best_aggregate_epoch)
        if best_aggregate_epoch is not None
        else -1,
        avg_dev_loss_for_best_step=avg_dev_loss_for_best_step,
        avg_dev_F1_for_best_step=avg_dev_F1_for_best_step,
        avg_dev_precision_for_best_step=avg_dev_precision_for_best_step,
        avg_dev_recall_for_best_step=avg_dev_recall_for_best_step,
        avg_test_loss_for_best_step=avg_test_loss_for_best_step,
        avg_test_F1_for_best_step=avg_test_F1_for_best_step,
        avg_test_precision_for_best_step=avg_test_precision_for_best_step,
        avg_test_recall_for_best_step=avg_test_recall_for_best_step,
        best_agg_dev_losses=best_agg_dev_losses,
        best_agg_dev_F1s=best_agg_dev_F1s,
        best_per_run_test_steps=best_per_run_test_steps,
        best_per_run_test_epochs=best_per_run_test_epochs,
        best_agg_test_losses=best_agg_test_losses,
        best_agg_test_F1s=best_agg_test_F1s,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    corpus_path: Path,
    output_dir: Path,
    positive_author_id: str,
    cfg: TrainConfig | None = None,
    exclude_work: str | None = None,
    monte_carlo_validation: bool = False,
    max_steps: int | None = None,
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
        exclude_work: useful for training a model to classify a specific document in training corpus
        monte_carlo_validation: controls use of TestEvalCallback

    Returns:
        Path to the saved model directory.
    """

    if cfg is None:
        cfg = TrainConfig()

    # Load data
    dataset = path_to_dataset(corpus_path, positive_author_id)
    if exclude_work:
        dataset = dataset.filter(
            lambda row: f"{row['author_id']}.{row['work_id']}" != exclude_work
        )
    logger.info("Loaded %d passages from %s", len(dataset), corpus_path)
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
        return tokenizer(examples["text"], truncation=True, max_length=cfg.max_length)

    train_tok = train_ds.map(tokenize, batched=True)
    dev_tok = dev_ds.map(tokenize, batched=True)
    test_tok = test_ds.map(tokenize, batched=True)

    # Train
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"

    # logging_dir is deprecated in transformers v5+; use env var instead
    os.environ["TENSORBOARD_LOGGING_DIR"] = (
        str(cfg.train_log_dir) if cfg.train_log_dir else str(output_dir / "logs")
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_epochs,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        save_total_limit=3,
        include_for_metrics=["loss"],
        seed=cfg.seed,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
    )

    # this overrides num_train_epochs
    if max_steps:
        training_args.max_steps = max_steps

    callbacks = []
    test_eval_callback = None
    if monte_carlo_validation:
        # NOTE: trainer needs to be added as test_eval_callback.trainer
        test_eval_callback = TestEvalCallback(test_tok)
        callbacks.append(test_eval_callback)
    elif cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )

    if test_eval_callback:
        test_eval_callback.trainer = trainer

    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    logger.info("Saved model to %s", model_dir)

    # grab history from test_eval_callback if in use
    # and also make sure dev metrics are accessible for run
    if test_eval_callback:
        _write_train_dev_test_history(trainer, test_eval_callback, output_dir)

    if (cfg.train_ratio + cfg.dev_ratio) >= 1.0:
        logging.info("No test set configured, so skipping final test set eval")
        return model_dir

    # Evaluate on test set and write predictions
    predictions_path = output_dir / TEST_SET_PREDICTION_FILENAME
    _, test_outputs = _write_predictions(
        trainer, test_tok, test_ds, "test", predictions_path
    )

    test_metrics_path = output_dir / TEST_SET_PREDICTION_METRICS_FILENAME

    if test_outputs.predictions is not None and test_outputs.label_ids is not None:
        test_loss = (
            test_outputs.metrics.get("test_loss") if test_outputs.metrics else None
        )
        test_outputs = EvalPrediction(
            predictions=test_outputs.predictions,
            label_ids=test_outputs.label_ids,
        )

        test_metrics = _compute_metrics(test_outputs)
        steps_per_epoch = math.ceil(
            len(train_tok)
            / (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
            )
        )
        with open(test_metrics_path, "w") as f:
            json.dump(
                {
                    "step": trainer.state.best_global_step,
                    "epoch": trainer.state.best_global_step / steps_per_epoch
                    if isinstance(trainer.state.best_global_step, int)
                    else -1,
                    "test_loss": test_loss,
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                },
                f,
            )
    else:
        logging.warning("skipping metrics on test set")

    return model_dir


def monte_carlo_train(
    corpus_path: Path,
    output_dir: Path,
    positive_author_id: str,
    num_runs: int = 5,
    train_full_model: bool = True,
    cfg: TrainConfig | None = None,
    exclude_work: str | None = None,
) -> Path | None:
    """
    Function to orchestrate training runs for monte-carlo cross validation.
    If train_full_model is set to True (default), cross-validation results are used
    to choose best step/epoch and a model is trained on the full dataset.

    Args:
        corpus_path: Input corpus JSONL file.
        output_dir: Directory to save model, checkpoints, and predictions.
        positive_author_id: Author ID for the positive class (e.g. "tlg0057").
        num_runs: number of training runs with different data partitions
        train_full_model: contols whether to use MC-CV to determine best step/epoch, and train model on all data to best step
        cfg: Training configuration. Uses TrainConfig defaults if not provided.
        exclude_work: useful for training a model to classify a specific document in training corpus

    Returns:
        Path to the saved model directory if train_full_model is set to True, else returns None
    """

    cfg = copy.deepcopy(cfg) if cfg else TrainConfig()

    base_log_dir = cfg.train_log_dir
    output_paths = []
    for i in range(num_runs):
        logger.info(f"Running Monte-Carlo CV with seed {cfg.seed}")
        run_path = output_dir / f"mc-cv-run-{i + 1}"
        if base_log_dir:
            cfg.train_log_dir = base_log_dir / f"mc-cv-run-{i + 1}"
        output_paths.append(run_path)
        model_dir = train(
            corpus_path,
            run_path,
            positive_author_id,
            cfg,
            exclude_work,
            monte_carlo_validation=True,
        )
        logger.info(f"deleting MC-CV model from {model_dir}")
        rmtree(model_dir)
        checkpoints_dir = run_path / "checkpoints"
        if checkpoints_dir.exists():
            logger.info(f"deleting MC-CV checkpoints from {checkpoints_dir}")
            rmtree(checkpoints_dir)
        cfg.seed += 1

    mc_aggregation = _get_mc_cv_metrics(output_paths)
    with open(output_dir / MC_CV_STATS_FILENAME, "w") as f:
        json.dump(dataclasses.asdict(mc_aggregation), f)

    for k, v in dataclasses.asdict(mc_aggregation).items():
        if v is None or v == -1:
            logging.warning(
                f"Metrics for Monte-Carlo CV not properly generated for {k}: {v}"
            )
    logging.info(
        f"Best aggregate step from cross-validation is {mc_aggregation.best_aggregate_step} = epoch {mc_aggregation.best_aggregate_epoch}"
    )

    msg = f"""\
    with:
        test set F1 averaged over runs at this step/epoch: {mc_aggregation.avg_test_F1_for_best_step}
        test set precision averaged over runs at this step/epoch: {mc_aggregation.avg_test_precision_for_best_step}
        test set recall averaged over runs at this step/epoch: {mc_aggregation.avg_test_recall_for_best_step}
        test set loss averaged over runs at this step/epoch: {mc_aggregation.avg_test_loss_for_best_step}
        dev set F1 averaged over runs at this step/epoch: {mc_aggregation.avg_dev_F1_for_best_step}
        dev set precision averaged over runs at this step/epoch: {mc_aggregation.avg_dev_precision_for_best_step}
        dev set recall averaged over runs at this step/epoch: {mc_aggregation.avg_dev_recall_for_best_step}
        dev set loss averaged over runs at this step/epoch: {mc_aggregation.avg_dev_loss_for_best_step}
    """
    logging.info(msg)

    if not train_full_model:
        logging.info("Finishing Monte-Carlo CV without training full model")
        return

    if not (
        isinstance(mc_aggregation.best_aggregate_step, int)
        and mc_aggregation.best_aggregate_step > 0
    ):
        logger.error(
            f"Can't train full model based on Monte-Carlo CV, since {mc_aggregation.best_aggregate_step=}"
        )
        return

    logging.info(
        "Training full model based on aggregate best step from Monte-Carlo CV ..."
    )

    cfg.early_stopping_patience = 0
    cfg.eval_strategy = "no"
    cfg.save_strategy = "no"
    cfg.load_best_model_at_end = False

    # set partition to put all data in training set
    cfg.train_ratio = 1.0
    cfg.dev_ratio = 0.0

    final_model_dir = train(
        corpus_path,
        output_dir / MC_CV_FINAL_MODEL_DIR,
        positive_author_id,
        cfg,
        exclude_work,
        max_steps=mc_aggregation.best_aggregate_step,
    )

    return final_model_dir


def predict(
    corpus_path: Path,
    model_dir: Path,
    output_path: Path,
    positive_author_id: str,
    max_length: int = 512,
    batch_size: int = 64,
) -> tuple[int, PredictionOutput]:
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
    dataset = path_to_dataset(corpus_path, positive_author_id)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

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
