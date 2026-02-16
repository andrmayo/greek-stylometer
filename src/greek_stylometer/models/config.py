"""Training configuration."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyperparameters for BERT fine-tuning.

    Defaults match the original Galen attribution experiments
    (pranaydeeps/Ancient-Greek-BERT, binary classification).
    """

    model_name: str = "pranaydeeps/Ancient-Greek-BERT"
    max_length: int = 512
    learning_rate: float = 2e-5
    train_batch_size: int = 64
    eval_batch_size: int = 64
    num_epochs: int = 6
    eval_steps: int = 25
    save_steps: int = 100
    seed: int = 12345
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
