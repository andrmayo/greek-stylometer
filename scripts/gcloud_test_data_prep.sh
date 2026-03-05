#!/usr/bin/env bash
set -euo pipefail

# assuming this script is in scripts/ dir
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
data_dir_name="stylometrics-rough"

BUCKET="gs://greek-stylometer-data"

if [[ ! -e "${project_dir}"/corpus.jsonl || ! -e "${project_dir}"/chunked.jsonl ]]; then
  echo 'Either corpus.jsonl or chunked.jsonl is missing, so generating both files before generating test file data file'

  # swap this out with name of parallel directory with data
  data_dir="${project_dir}"/../"${data_dir_name}"

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
fi

echo "copying 10% of lines randomly from chunked.jsonl to test_chunked.jsonl"
awk 'BEGIN { srand() } rand() < 0.1' "${project_dir}/chunked.jsonl" >"${project_dir}/test_chunked.jsonl"

# Upload test training data to GCS
echo "Uploading test_chunked.jsonl to GCS..."
gcloud storage cp "${project_dir}/test_chunked.jsonl" "$BUCKET"/
