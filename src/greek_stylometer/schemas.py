"""Data schemas for JSONL interchange format.

Every stage of the pipeline reads and writes JSONL using these schemas.
They also serve as the cross-language interchange format with Common Lisp.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Self


@dataclass
class Passage:
    """A single text passage from a corpus."""

    text: str
    author: str
    author_id: str
    work_id: str
    passage_idx: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Self:
        return cls(**json.loads(line))


@dataclass
class Prediction:
    """A model prediction on a passage."""

    text: str
    author: str
    label: int
    predicted: int
    confidence: float
    split: str
    passage_idx: int
    author_id: str = ""
    work_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Self:
        d = json.loads(line)
        d.setdefault("author_id", "")
        d.setdefault("work_id", "")
        return cls(**d)


@dataclass
class WorkPrediction:
    """An aggregated prediction for an entire work."""

    author: str
    author_id: str
    work_id: str
    predicted: int
    confidence: float
    n_chunks: int
    label: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Self:
        return cls(**json.loads(line))


@dataclass
class FeatureWeight:
    """A single token's contribution to a prediction."""

    token: str
    weight: float


@dataclass
class Explanation:
    """A LIME explanation for a single passage."""

    passage_idx: int
    text: str
    predicted: int
    confidence: float
    top_features: list[FeatureWeight]

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Self:
        d = json.loads(line)
        d["top_features"] = [FeatureWeight(**fw) for fw in d["top_features"]]
        return cls(**d)


@dataclass
class ActivationManifestEntry:
    """Metadata for exported model activations.

    Actual tensor data lives in .npy files; this JSONL record
    holds the metadata and file references so Lisp can find them.
    """

    passage_idx: int
    layer: int
    cls_embedding_file: str
    token_embeddings_file: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Self:
        return cls(**json.loads(line))
