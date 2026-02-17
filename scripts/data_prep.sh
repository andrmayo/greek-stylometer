#!/usr/bin/env bash

# assuming this script is in scripts/ dir
project_dir="$(cd "$(dirname "$0")/.." && pwd)"

# swap this out with name of parallel directory with data
data_dir="${project_dir}/../stylometry"

greek-stylometer ingest \
  --input "${data_dir}"/First1KGreek/data/tlg0057 \
  --input "${data_dir}"/First1KGreek/data/tlg0530 \
  --input "${data_dir}"/First1KGreek/data/tlg0555 \
  --input "${data_dir}"/First1KGreek/data/tlg0081 \
  --input "${data_dir}"/First1KGreek/data/tlg0564 \
  --input "${data_dir}"/First1KGreek/data/tlg0544 \
  --input "${data_dir}"/First1KGreek/data/tlg0565 \
  --output "${project_dir}"/corpus.jsonl

# Chunk for training (no overlap)
# Note that the sequence length warning is harmless
greek-stylometer chunk \
  --input "${project_dir}"/corpus.jsonl \
  --output "${project_dir}"/chunked.jsonl \
  --tokenizer pranaydeeps/Ancient-Greek-BERT
