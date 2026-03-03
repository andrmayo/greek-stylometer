#!/usr/bin/env bash

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

PROJECT_NUMBER="782738894010"
REGION="us-central1"
EXPERIMENT_NAME="galen-finetune"

if [ "$1" = "--delete" ]; then
  TB_TO_DELETE=$(gcloud ai tensorboards list --region="$REGION" --format="value(name)" --limit=1)
  if [ -z "$TB_TO_DELETE" ]; then
    echo "No tensorboard instances found."
    exit 1
  fi
  gcloud ai tensorboards delete "$TB_TO_DELETE" --region="$REGION"
  # Clean up metadata contexts and artifacts left behind by tb-gcp-uploader
  METADATA_BASE="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_NUMBER}/locations/${REGION}/metadataStores/default"
  TOKEN=$(gcloud auth print-access-token)
  # Delete all contexts with IDs starting with experiment name
  CONTEXTS=$(curl -s -H "Authorization: Bearer $TOKEN" "${METADATA_BASE}/contexts" | python3 -c "import sys,json; [print(c['name'].split('/')[-1]) for c in json.load(sys.stdin).get('contexts',[]) if c['name'].split('/')[-1].startswith('${EXPERIMENT_NAME}')]" 2>/dev/null)
  for ctx in $CONTEXTS; do
    echo "Deleting metadata context: $ctx"
    curl -s -X DELETE -H "Authorization: Bearer $TOKEN" "${METADATA_BASE}/contexts/$ctx" > /dev/null
  done
  # Delete all artifacts with IDs starting with experiment name
  ARTIFACTS=$(curl -s -H "Authorization: Bearer $TOKEN" "${METADATA_BASE}/artifacts" | python3 -c "import sys,json; [print(a['name'].split('/')[-1]) for a in json.load(sys.stdin).get('artifacts',[]) if a['name'].split('/')[-1].startswith('${EXPERIMENT_NAME}')]" 2>/dev/null)
  for art in $ARTIFACTS; do
    echo "Deleting metadata artifact: $art"
    curl -s -X DELETE -H "Authorization: Bearer $TOKEN" "${METADATA_BASE}/artifacts/$art" > /dev/null
  done
  exit
fi

LOGDIR="${1:-}"
if [ -z "$LOGDIR" ]; then
  echo "greek-stylometer-data bucket:"
  echo $(gcloud storage ls gs://greek-stylometer-data)
  read -rp "Enter GCS path for TensorBoard logs: " LOGDIR
fi
if [[ "$LOGDIR" != gs://greek-stylometry-data/* ]]; then
  LOGDIR="gs://greek-stylometer-data/$LOGDIR"
fi

TB_RESOURCE=$(gcloud ai tensorboards list --region=us-central1 --format="value(name)" --limit=1)
if [ -z "$TB_RESOURCE" ]; then
  gcloud ai tensorboards create --display-name="stylometry-tensorboard" --region=us-central1
  TB_RESOURCE=$(gcloud ai tensorboards list --region=us-central1 --format="value(name)" --limit=1)
fi

tb-gcp-uploader --tensorboard_resource_name="$TB_RESOURCE" --experiment_name=galen-finetune --one_shot=true --logdir="$LOGDIR"

echo "Monitor tensorboard outputs with:"
echo "  gcloud ai tensorboards open --region=us-central1 --tensorboard=$TB_RESOURCE"
