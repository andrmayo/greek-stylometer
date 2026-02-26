#!/usr/bin/env bash

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

BUCKET="gs://greek-stylometer-data"

TB_RESOURCE=$(gcloud ai tensorboards list --region=us-central1 --format="value(name)" --limit=1)
if [ -z "$TB_RESOURCE" ]; then
  gcloud ai tensorboards create --display-name="stylometry-tensorboard" --region=us-central1
  TB_RESOURCE=$(gcloud ai tensorboards list --region=us-central1 --format="value(name)" --limit=1)
fi

tb-gcp-uploader --tensorboard_resource_name="$TB_RESOURCE" --experiment_name=galen-finetune --one_shot=true --logdir=gs://greek-stylometer-data/output/logs/

echo "Monitor tensorboard outputs with:"
echo "  gcloud ai tensorboards open --region=us-central1 --tensorboard="$TB_RESOURCE""
