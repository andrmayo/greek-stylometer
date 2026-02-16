"""Tokenizer-aware text chunking.

Concatenates passages within each work, then splits into chunks
of a target token length. Produces a new set of Passage records
with uniform size, suitable for transformer training.
"""

from collections.abc import Iterator
from itertools import groupby
from pathlib import Path

from greek_stylometer.corpus.jsonl import read_corpus, write_corpus
from greek_stylometer.schemas import Passage


def _work_key(passage: Passage) -> tuple[str, str]:
    """Group key: (author_id, work_id)."""
    return (passage.author_id, passage.work_id)


def chunk_passages(
    passages: Iterator[Passage],
    tokenizer,
    max_tokens: int = 512,
    overlap: int = 0,
) -> Iterator[Passage]:
    """Re-chunk passages to a uniform token length.

    Concatenates text from passages sharing the same (author_id, work_id),
    tokenizes the combined text, and yields new Passage records of
    exactly ``max_tokens`` tokens (except possibly the last chunk per work).

    Args:
        passages: Input passages, should be sorted by (author_id, work_id).
        tokenizer: A HuggingFace tokenizer (anything with an ``encode``
            method that returns token IDs, and a ``decode`` method).
        max_tokens: Target chunk size in tokens.
        overlap: Number of tokens to overlap between consecutive chunks.
            Use 0 for training (avoids data leakage), >0 for inference
            (sliding window).
    """
    stride = max_tokens - overlap

    for (author_id, work_id), group in groupby(passages, key=_work_key):
        group_list = list(group)
        if not group_list:
            continue

        # Take metadata from the first passage in the group
        first = group_list[0]

        combined_text = " ".join(p.text for p in group_list)
        token_ids = tokenizer.encode(combined_text, add_special_tokens=False)

        chunk_idx = 0
        start = 0
        while start < len(token_ids):
            chunk_ids = token_ids[start : start + max_tokens]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

            yield Passage(
                text=chunk_text,
                author=first.author,
                author_id=author_id,
                work_id=work_id,
                passage_idx=chunk_idx,
                source=first.source,
                metadata={
                    **first.metadata,
                    "chunk_tokens": len(chunk_ids),
                    "chunked": True,
                },
            )
            chunk_idx += 1
            start += stride


def chunk_corpus_file(
    input_path: Path,
    output_path: Path,
    tokenizer_name: str,
    max_tokens: int = 512,
    overlap: int = 0,
) -> int:
    """Read a corpus JSONL, re-chunk, and write to a new JSONL.

    Args:
        input_path: Path to input corpus JSONL.
        output_path: Path to write chunked JSONL.
        tokenizer_name: HuggingFace tokenizer name or path.
        max_tokens: Target chunk size in tokens.
        overlap: Token overlap between consecutive chunks.

    Returns the number of chunks written.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    passages = read_corpus(input_path)
    chunks = chunk_passages(passages, tokenizer, max_tokens, overlap)
    return write_corpus(chunks, output_path)
