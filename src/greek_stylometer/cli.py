"""CLI entry points for greek-stylometer.

Thin layer: parse arguments, delegate to module functions, print results.
"""

import logging

from enum import Enum
from pathlib import Path
from typing import Annotated, assert_never

import typer

from greek_stylometer.models.config import TrainConfig

_TRAIN_DEFAULTS = TrainConfig()

app = typer.Typer(
    help="Authorship attribution and interpretability toolkit for Ancient Greek texts."
)


class Source(str, Enum):
    first1kgreek = "first1kgreek"


@app.callback()
def configure(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging.")
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


@app.command()
def ingest(
    input_paths: Annotated[
        list[Path],
        typer.Option(
            "--input", exists=True, help="Author directory to ingest (can be repeated)."
        ),
    ],
    output: Annotated[Path, typer.Option(help="Output JSONL file path.")],
    source: Annotated[
        Source, typer.Option(help="Corpus format to ingest.")
    ] = Source.first1kgreek,
    keep_section_numbers: Annotated[
        bool, typer.Option(help="Keep section numbers in passage text.")
    ] = False,
    no_iota_adscript: Annotated[
        bool, typer.Option(help="Disable iota subscript → adscript conversion.")
    ] = False,
) -> None:
    """Ingest a corpus into JSONL format."""
    if source == Source.first1kgreek:
        from greek_stylometer.corpus.first1kgreek import ingest as do_ingest

        count = do_ingest(
            input_paths,
            output,
            strip_section_numbers=not keep_section_numbers,
            iota_adscript=not no_iota_adscript,
        )
    else:
        assert_never(source)

    typer.echo(f"Wrote {count} passages to {output}")


@app.command()
def chunk(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, help="Input corpus JSONL file.")
    ],
    output: Annotated[Path, typer.Option(help="Output chunked JSONL file.")],
    tokenizer: Annotated[str, typer.Option(help="HuggingFace tokenizer name or path.")],
    max_tokens: Annotated[int, typer.Option(help="Target chunk size in tokens.")] = 512,
    overlap: Annotated[
        int,
        typer.Option(
            help="Token overlap between chunks. 0 for training, >0 for sliding window."
        ),
    ] = 0,
) -> None:
    """Re-chunk a corpus JSONL to uniform token lengths."""
    from greek_stylometer.preprocessing.chunking import chunk_corpus_file

    typer.echo(f"Loading tokenizer: {tokenizer}")
    count = chunk_corpus_file(input_path, output, tokenizer, max_tokens, overlap)
    typer.echo(f"Wrote {count} chunks to {output}")


@app.command()
def train(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, help="Input corpus JSONL file.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Output directory for model and predictions.")
    ],
    positive_author: Annotated[
        str, typer.Option(help="Author ID for the positive class (e.g. tlg0057).")
    ],
    model_name: Annotated[
        str, typer.Option(help="HuggingFace model name or path.")
    ] = _TRAIN_DEFAULTS.model_name,
    max_length: Annotated[
        int, typer.Option(help="Maximum token length.")
    ] = _TRAIN_DEFAULTS.max_length,
    learning_rate: Annotated[
        float, typer.Option(help="Learning rate.")
    ] = _TRAIN_DEFAULTS.learning_rate,
    train_batch_size: Annotated[
        int, typer.Option(help="Training batch size.")
    ] = _TRAIN_DEFAULTS.train_batch_size,
    num_epochs: Annotated[
        int, typer.Option(help="Number of training epochs.")
    ] = _TRAIN_DEFAULTS.num_epochs,
    exclude_work: Annotated[
        str | None,
        typer.Option(
            help="tlg-format id of author + work to exclude from training, e.g. tlg0001.tlg001"
        ),
    ] = None,
    train_log_dir: Annotated[
        Path | None,
        typer.Option(
            help="Specify dir for training logs (general logging set up in callback)"
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = _TRAIN_DEFAULTS.seed,
) -> None:
    """Train a binary BERT classifier on a corpus JSONL."""
    from greek_stylometer.models.bert import train as do_train

    cfg = TrainConfig(
        model_name=model_name,
        max_length=max_length,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        num_epochs=num_epochs,
        seed=seed,
        train_log_dir=train_log_dir,
    )
    typer.echo(f"Training {cfg.model_name} on {input_path}")
    model_dir = do_train(input_path, output_dir, positive_author, cfg, exclude_work)
    typer.echo(f"Model saved to {model_dir}")


@app.command()
def predict(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, help="Input corpus JSONL file.")
    ],
    model_dir: Annotated[
        Path, typer.Option(exists=True, help="Directory containing the saved model.")
    ],
    output: Annotated[Path, typer.Option(help="Output predictions JSONL file.")],
    positive_author: Annotated[
        str, typer.Option(help="Author ID for the positive class (e.g. tlg0057).")
    ],
    max_length: Annotated[int, typer.Option(help="Maximum token length.")] = 512,
    batch_size: Annotated[int, typer.Option(help="Batch size for inference.")] = 64,
) -> None:
    """Run inference with a saved model on a corpus JSONL."""
    from greek_stylometer.models.bert import predict as do_predict

    typer.echo(f"Predicting with model from {model_dir}")
    count = do_predict(
        input_path, model_dir, output, positive_author, max_length, batch_size
    )
    typer.echo(f"Wrote {count} predictions to {output}")


@app.command()
def evaluate(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, help="Predictions JSONL file.")
    ],
    output: Annotated[
        Path | None, typer.Option(help="Output JSON file for results.")
    ] = None,
) -> None:
    """Compute evaluation metrics from a predictions JSONL."""
    from greek_stylometer.analysis.evaluation import evaluate as do_evaluate

    summary = do_evaluate(input_path, output)
    typer.echo(summary)


@app.command()
def explain(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, help="Predictions JSONL file to explain."),
    ],
    model_dir: Annotated[
        Path, typer.Option(exists=True, help="Directory containing the saved model.")
    ],
    output: Annotated[Path, typer.Option(help="Output explanations JSONL file.")],
    num_features: Annotated[
        int, typer.Option(help="Number of top features per explanation.")
    ] = 10,
    num_samples: Annotated[
        int, typer.Option(help="Number of perturbed samples for LIME.")
    ] = 5000,
    html_dir: Annotated[
        Path | None, typer.Option(help="Directory to save HTML visualizations.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed for LIME.")] = 1234,
) -> None:
    """Generate LIME explanations for model predictions."""
    from greek_stylometer.analysis.lime import explain as do_explain

    typer.echo(f"Generating LIME explanations for {input_path}")
    count = do_explain(
        input_path,
        model_dir,
        output,
        num_features,
        num_samples,
        html_dir=html_dir,
        seed=seed,
    )
    typer.echo(f"Wrote {count} explanations to {output}")


@app.command(name="export-activations")
def export_activations(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, help="Input corpus JSONL file.")
    ],
    model_dir: Annotated[
        Path, typer.Option(exists=True, help="Directory containing the saved model.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Output directory for .npy files and manifest.")
    ],
    layers: Annotated[
        list[int] | None,
        typer.Option(
            help="Layer indices to export (can be repeated). Defaults to last layer."
        ),
    ] = None,
    include_token_embeddings: Annotated[
        bool, typer.Option(help="Also export per-token embeddings.")
    ] = False,
    max_length: Annotated[int, typer.Option(help="Maximum token length.")] = 512,
    batch_size: Annotated[int, typer.Option(help="Batch size for inference.")] = 32,
) -> None:
    """Export model activations to .npy files for downstream analysis."""
    from greek_stylometer.models.export import export_activations as do_export

    typer.echo(f"Exporting activations from {model_dir}")
    count = do_export(
        input_path,
        model_dir,
        output_dir,
        layers,
        include_token_embeddings,
        max_length,
        batch_size,
    )
    typer.echo(f"Wrote {count} activation entries to {output_dir}")


@app.command()
def classify(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, help="Input corpus JSONL (whole works)."),
    ],
    model_dir: Annotated[
        Path, typer.Option(exists=True, help="Directory containing the saved model.")
    ],
    output: Annotated[Path, typer.Option(help="Output work-level predictions JSONL.")],
    positive_author: Annotated[
        str, typer.Option(help="Author ID for the positive class (e.g. tlg0057).")
    ],
    max_tokens: Annotated[int, typer.Option(help="Chunk size in tokens.")] = 512,
    overlap: Annotated[
        int | None,
        typer.Option(help="Token overlap between chunks. Defaults to max-tokens / 4."),
    ] = None,
    max_length: Annotated[
        int, typer.Option(help="Maximum token length for model inference.")
    ] = 512,
    batch_size: Annotated[int, typer.Option(help="Batch size for inference.")] = 64,
) -> None:
    """Classify whole works: chunk with overlap, predict, aggregate."""
    import tempfile

    from greek_stylometer.analysis.aggregation import aggregate_predictions
    from greek_stylometer.models.bert import predict as do_predict
    from greek_stylometer.preprocessing.chunking import chunk_corpus_file

    effective_overlap = overlap if overlap is not None else max_tokens // 4

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        chunked_path = tmp / "chunked.jsonl"
        predictions_path = tmp / "chunk_predictions.jsonl"

        typer.echo(
            f"Chunking with max_tokens={max_tokens}, overlap={effective_overlap}"
        )
        n_chunks = chunk_corpus_file(
            input_path,
            chunked_path,
            str(model_dir),
            max_tokens=max_tokens,
            overlap=effective_overlap,
        )
        typer.echo(f"  {n_chunks} chunks")

        typer.echo("Running inference on chunks")
        n_preds = do_predict(
            chunked_path,
            model_dir,
            predictions_path,
            positive_author,
            max_length,
            batch_size,
        )
        typer.echo(f"  {n_preds} predictions")

        typer.echo("Aggregating to work level")
        summary = aggregate_predictions(predictions_path, output)

    typer.echo(summary)


def main() -> None:
    app()
