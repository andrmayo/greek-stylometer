#!/usr/bin/env bash
set -euo pipefail

# Run from project root
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

BUCKET="gs://greek-stylometer-data"
IMAGE="us-central1-docker.pkg.dev/greek-stylometer/greek-stylometer/train:latest"

# Upload training data to GCS
echo "Uploading chunked.jsonl to GCS..."
gcloud storage cp chunked.jsonl "$BUCKET"/

# Build and push Docker image
echo "Building Docker image..."
docker build -t "$IMAGE" .

echo "Pushing Docker image..."
docker push "$IMAGE"

# Submit training job
echo "Submitting Vertex AI training job..."
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=galen-finetune \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri="$IMAGE" \
  --args=train,--input,"$BUCKET"/chunked.jsonl,--output-dir,"$BUCKET"/output,--positive-author,tlg0057

echo "Job submitted. Monitor with:"
echo "  gcloud ai custom-jobs list --region=us-central1"
echo "  gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1"
