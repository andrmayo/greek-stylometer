"""Training configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """
    Hyperparameters for BERT fine-tuning.
    """

    model_name: str = "pranaydeeps/Ancient-Greek-BERT"
    max_length: int = 512
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 6
    eval_steps: int = 25
    save_steps: int = 100
    seed: int = 12345
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    train_log_dir: Path | None = None  # None for same dir as general logging
