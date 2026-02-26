#!/usr/bin/env bash
set -euo pipefail

# Run from project root
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

BUCKET="gs://greek-stylometer-data"
IMAGE="us-central1-docker.pkg.dev/greek-stylometer/greek-stylometer/train:latest"

# Create tensorboard instance if needed
TB_RESOURCE=$(gcloud ai tensorboards list --region=us-central1 --format="value(name)" --limit=1)
if [ -z "$TB_RESOURCE" ]; then
  gcloud ai tensorboards create --display-name="stylometry-tensorboard" --region=us-central1
  TB_RESOURCE=$(gcloud ai tensorboards list --region=us-central1 --format="value(name)" --limit=1)
fi

# Submit training job
echo "Submitting Vertex AI training job..."
JOB_ID=$(
  gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=galen-finetune \
    --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri="$IMAGE" \
    --args=train,--input,"$BUCKET"/chunked.jsonl,--output-dir,"$BUCKET"/output,--positive-author,tlg0057 \
    --format="value(name)"
)

echo "Job submitted. Waiting for logs directory to appear..."
LOGS_DIR="$BUCKET/output/logs/"
while ! gcloud storage ls "$LOGS_DIR" &>/dev/null; do
  sleep 30
done

echo "Logs detected. Starting TensorBoard uploader..."
tb-gcp-uploader --tensorboard_resource_name="$TB_RESOURCE" --experiment_name=galen-finetune --logdir="$LOGS_DIR" &
TB_PID=$!
trap 'kill $TB_PID 2>/dev/null' EXIT

echo "Monitor TensorBoard with:"
echo "  gcloud ai tensorboards open --region=us-central1 --tensorboard=$TB_RESOURCE"

gcloud ai custom-jobs stream-logs "$JOB_ID" --region=us-central1
