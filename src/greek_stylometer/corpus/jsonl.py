"""JSONL read/write utilities for Passage records."""

from pathlib import Path
from typing import Generator, Iterable, Iterator, cast

from greek_stylometer.schemas import Passage


def write_corpus(passages: Iterable[Passage], path: Path) -> int:
    """Write passages to a JSONL file. Returns the number of passages written."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for passage in passages:
            f.write(passage.to_json() + "\n")
            count += 1
    return count


def read_corpus(path: Path) -> Iterator[Passage]:
    """Lazily read passages from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Passage.from_json(line)


def get_training_dict(
    passage: Passage,
    positive_author_id: str,
) -> dict[str, str | int | list[str | int]]:
    return {
        "text": passage.text,
        "label": 1 if passage.author_id == positive_author_id else 0,
        "author": passage.author,
        "passage_idx": passage.passage_idx,
        "author_id": passage.author_id,
        "work_id": passage.work_id,
    }


def stream_passage_to_dict(
    passages: Iterator[Passage], positive_author_id: str
) -> Generator[dict[str, str | int]]:
    for p in passages:
        yield cast(dict[str, str | int], get_training_dict(p, positive_author_id))
