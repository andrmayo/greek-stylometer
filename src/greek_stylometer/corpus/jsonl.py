"""JSONL read/write utilities for Passage records."""

from collections.abc import Iterable, Iterator
from pathlib import Path

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
