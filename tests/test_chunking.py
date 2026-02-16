"""Tests for tokenizer-aware chunking."""

from greek_stylometer.preprocessing.chunking import chunk_passages
from greek_stylometer.schemas import Passage


class FakeTokenizer:
    """Minimal tokenizer stub that splits on whitespace."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return list(range(len(text.split())))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        # We can't recover real text from fake IDs, so just return
        # a placeholder of the right "word count"
        return " ".join(f"w{i}" for i in ids)


def _make_passage(
    text: str, author_id: str = "tlg0057", work_id: str = "tlg001"
) -> Passage:
    return Passage(
        text=text,
        author="Galen",
        author_id=author_id,
        work_id=work_id,
        passage_idx=0,
        source="test",
    )


def test_short_passages_combined():
    """Two short passages from the same work become one chunk."""
    passages = [
        _make_passage("a b c"),
        _make_passage("d e f"),
    ]
    chunks = list(chunk_passages(iter(passages), FakeTokenizer(), max_tokens=10))
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_tokens"] == 6


def test_long_passage_split():
    """A passage longer than max_tokens is split into multiple chunks."""
    text = " ".join(f"word{i}" for i in range(20))
    passages = [_make_passage(text)]
    chunks = list(chunk_passages(iter(passages), FakeTokenizer(), max_tokens=8))
    assert len(chunks) == 3  # 8 + 8 + 4
    assert chunks[0].metadata["chunk_tokens"] == 8
    assert chunks[1].metadata["chunk_tokens"] == 8
    assert chunks[2].metadata["chunk_tokens"] == 4


def test_overlap():
    """Chunks overlap by the specified amount."""
    text = " ".join(f"w{i}" for i in range(10))
    passages = [_make_passage(text)]
    chunks = list(
        chunk_passages(iter(passages), FakeTokenizer(), max_tokens=6, overlap=2)
    )
    # stride = 4: chunks at [0:6], [4:10], [8:10] = 3 chunks
    assert len(chunks) == 3
    assert chunks[0].metadata["chunk_tokens"] == 6
    assert chunks[1].metadata["chunk_tokens"] == 6
    assert chunks[2].metadata["chunk_tokens"] == 2


def test_different_works_not_merged():
    """Passages from different works are chunked separately."""
    passages = [
        _make_passage("a b c", work_id="tlg001"),
        _make_passage("d e f", work_id="tlg002"),
    ]
    chunks = list(chunk_passages(iter(passages), FakeTokenizer(), max_tokens=10))
    assert len(chunks) == 2
    assert chunks[0].work_id == "tlg001"
    assert chunks[1].work_id == "tlg002"


def test_metadata_preserved():
    """Author info and source are carried through to chunks."""
    passages = [_make_passage("a b c d e")]
    chunks = list(chunk_passages(iter(passages), FakeTokenizer(), max_tokens=10))
    assert chunks[0].author == "Galen"
    assert chunks[0].author_id == "tlg0057"
    assert chunks[0].source == "test"
    assert chunks[0].metadata["chunked"] is True
