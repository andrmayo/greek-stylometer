"""Activation and embedding export from BERT models.

Registers forward hooks on specified layers, runs the model over a
corpus, and writes CLS embeddings (and optionally full token embeddings)
to .npy files.  A JSONL manifest maps passage indices to file paths,
enabling downstream consumption from both Python and Common Lisp.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from greek_stylometer.corpus.jsonl import read_corpus
from greek_stylometer.schemas import ActivationManifestEntry

logger = logging.getLogger(__name__)


def export_activations(
    corpus_path: Path,
    model_dir: Path,
    output_dir: Path,
    layers: list[int] | None = None,
    include_token_embeddings: bool = False,
    max_length: int = 512,
    batch_size: int = 32,
) -> int:
    """Export model activations for a corpus to .npy files.

    Args:
        corpus_path: Input corpus JSONL file.
        model_dir: Directory containing the saved model.
        output_dir: Directory to write .npy files and manifest JSONL.
        layers: Which BERT layers to export (0-indexed). Defaults to
            the last layer only.
        include_token_embeddings: If True, also save per-token embeddings
            (larger files).
        max_length: Maximum token length for tokenization.
        batch_size: Batch size for inference.

    Returns:
        Number of manifest entries written.
    """
    passages = list(read_corpus(corpus_path))
    logger.info("Loaded %d passages from %s", len(passages), corpus_path)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, output_hidden_states=True
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    if layers is None:
        layers = [num_layers - 1]
    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(
                f"Layer {layer} out of range for model with {num_layers} layers"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    npy_dir = output_dir / "activations"
    npy_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "activations.jsonl"

    count = 0
    with torch.no_grad(), open(manifest_path, "w", encoding="utf-8") as manifest:
        for start in range(0, len(passages), batch_size):
            batch_passages = passages[start : start + batch_size]
            texts = [p.text for p in batch_passages]

            inputs = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )

            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden)

            for i, passage in enumerate(batch_passages):
                for layer in layers:
                    # +1 because hidden_states[0] is the embedding layer output
                    layer_output = hidden_states[layer + 1][i]

                    # CLS embedding (first token)
                    cls_emb = layer_output[0].numpy()
                    cls_file = f"{passage.passage_idx}_layer{layer}_cls.npy"
                    np.save(npy_dir / cls_file, cls_emb)

                    token_file = None
                    if include_token_embeddings:
                        token_emb = layer_output.numpy()
                        token_file = f"{passage.passage_idx}_layer{layer}_tokens.npy"
                        np.save(npy_dir / token_file, token_emb)

                    entry = ActivationManifestEntry(
                        passage_idx=passage.passage_idx,
                        layer=layer,
                        cls_embedding_file=f"activations/{cls_file}",
                        token_embeddings_file=(
                            f"activations/{token_file}" if token_file else None
                        ),
                    )
                    manifest.write(entry.to_json() + "\n")
                    count += 1

            if start % (batch_size * 10) == 0 and start > 0:
                logger.info("Exported %d / %d passages", start, len(passages))

    logger.info("Wrote %d manifest entries to %s", count, manifest_path)
    return count
