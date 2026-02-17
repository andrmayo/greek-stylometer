# greek-stylometer

Authorship attribution and interpretability toolkit for Ancient Greek texts.

Classifies passages using fine-tuned BERT models, then investigates _why_ the
model makes the decisions it does — via LIME in Python and sparse autoencoders
in Common Lisp. All pipeline stages communicate through JSONL, keeping the
Python and Lisp sides decoupled.

## Install

Requires Python 3.13+. Managed with [uv](https://docs.astral.sh/uv/).

```
uv sync              # install all dependencies
uv sync --group dev  # + pytest, pyright, jupyter
```

## Usage

### Ingest

Multiple `--input` flags can be passed to ingest several authors into one file.

Convert First1KGreek TEI XML into corpus JSONL:

```
greek-stylometer ingest \
    --input ../stylometry/First1KGreek/data/tlg0057 \
    --input ../stylometry/First1KGreek/data/tlg0530 \
    --input ../stylometry/First1KGreek/data/tlg0555 \
    --input ../stylometry/First1KGreek/data/tlg0081 \
    --input ../stylometry/First1KGreek/data/tlg0564 \
    --input ../stylometry/First1KGreek/data/tlg0544 \
    --input ../stylometry/First1KGreek/data/tlg0565 \
    --output corpus.jsonl
```

The above ingests Galen, pseudo-Galen, Clement of Alexandria, Dionysius of
Halicarnassus, Rufus of Ephesus, Sextus Empiricus, and Soranus (assuming that
the `First1KGreek/` directory is in a `stylometry/` directory parallel to the
project repo).

### Chunk

Re-chunk a corpus to uniform token lengths (for training):

```

greek-stylometer chunk \
 --input corpus.jsonl \
 --output corpus-chunked.jsonl \
 --tokenizer pranaydeeps/Ancient-Greek-BERT \
 --max-tokens 512

```

### Train

Train a binary BERT classifier (target author vs. all others):

```

greek-stylometer train \
 --input corpus-chunked.jsonl \
 --output-dir runs/galen \
 --positive-author tlg0057 \
 --model-name pranaydeeps/Ancient-Greek-BERT

```

Saves the model to `runs/galen/model/` and writes test-set predictions to
`runs/galen/test_predictions.jsonl`.

### Predict

Run inference with a saved model:

```

greek-stylometer predict \
 --input corpus.jsonl \
 --model-dir runs/galen/model \
 --output predictions.jsonl \
 --positive-author tlg0057

```

### Evaluate

Compute per-author accuracy, precision, recall, and F1:

```

greek-stylometer evaluate --input predictions.jsonl greek-stylometer evaluate
--input predictions.jsonl --output eval.json

```

### Explain

Generate LIME explanations for model predictions:

```

greek-stylometer explain \
 --input predictions.jsonl \
 --model-dir runs/galen/model \
 --output explanations.jsonl \
 --html-dir lime-html/

```

### Export activations

Extract CLS/token embeddings for downstream analysis (e.g. sparse autoencoders):

```

greek-stylometer export-activations \
 --input corpus.jsonl \
 --model-dir runs/galen/model \
 --output-dir activations/ \
 --layers 11

```

Writes `.npy` files with a JSONL manifest (`activations/activations.jsonl`).

## Project structure

```

src/greek_stylometer/ schemas.py # JSONL data schemas (Passage, Prediction,
Explanation, ...) cli.py # CLI entry points (typer) corpus/ # raw data -> JSONL
ingestion cleaning.py # TEI XML cleaning pipeline first1kgreek.py # First1KGreek
corpus reader jsonl.py # JSONL read/write base.py # CorpusReader protocol
preprocessing/ # text normalization and chunking normalize.py # accent
stripping, section number removal chunking.py # tokenizer-aware uniform chunking
models/ # BERT fine-tuning, inference, activation export config.py # TrainConfig
dataclass bert.py # train() and predict() export.py # activation/embedding
extraction -> npy + JSONL analysis/ # interpretability and evaluation
evaluation.py # per-author metrics from predictions JSONL lime.py # LIME
explanations -> JSONL + HTML

```

```

```
