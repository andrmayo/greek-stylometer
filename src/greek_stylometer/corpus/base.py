"""Base protocol for corpus readers."""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from greek_stylometer.schemas import Passage


class CorpusReader(Protocol):
    """Protocol for reading a corpus into Passage records.

    Implementations should yield Passage objects from a given source path.
    Each implementation handles a specific corpus format (e.g. First1KGreek
    TEI XML, plain text directories, etc.).
    """

    def read(self, source: Path) -> Iterator[Passage]: ...
