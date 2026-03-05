#!/usr/bin/env bash
set -euo pipefail

# Run from project root
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

BUCKET="gs://greek-stylometer-data"
IMAGE="us-central1-docker.pkg.dev/greek-stylometer/greek-stylometer/train:latest"

# Build and push Docker image
echo "Building Docker image..."
docker build -t "$IMAGE" .

echo "Pushing Docker image..."
docker push "$IMAGE"

echo "Submitting Vertex AI monte-carlo val and train job..."

JOB_ID=$(
  gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=galen-mccv-finetune \
    --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri="$IMAGE" \
    --args=monte-carlo,--input,"$BUCKET"/chunked.jsonl,--output-dir,"$BUCKET"/output/mccv,--positive-author,tlg0057,--num-runs,5 \
    --format="value(name)"
)

echo "Job submitted: $JOB_ID"
echo ""
echo "After job completes, view TensorBoard with:"
echo "  tb-gcp-uploader --tensorboard_resource_name=\$(gcloud ai tensorboards list --region=us-central1 --format='value(name)' --limit=1) --experiment_name=galen-mccv-finetune --logdir=$BUCKET/output/mccv/"
echo ""
gcloud ai custom-jobs stream-logs "$JOB_ID" --region=us-central1
