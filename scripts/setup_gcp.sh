#!/usr/bin/env bash
# One-command GCP setup for DocuSense AI.
#
# Usage:
#   ./scripts/setup_gcp.sh <PROJECT_ID> [REGION]
#
# Creates/enables everything the app needs: APIs, GCS bucket, Pub/Sub
# topic + subscription, Artifact Registry repo, and a runtime service
# account with least-privilege roles.

set -euo pipefail

PROJECT_ID="${1:?Usage: ./scripts/setup_gcp.sh <PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"

BUCKET="${PROJECT_ID}-docusense-documents"
TOPIC="docusense-ingest"
SUBSCRIPTION="docusense-ingest-sub"
REPO="docusense"
SA_NAME="docusense-run"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "==> Using project: ${PROJECT_ID} (region: ${REGION})"
gcloud config set project "${PROJECT_ID}"

echo "==> Enabling required APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  pubsub.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com

echo "==> Creating GCS bucket gs://${BUCKET}..."
gcloud storage buckets create "gs://${BUCKET}" \
  --location="${REGION}" --uniform-bucket-level-access \
  2>/dev/null || echo "    (bucket already exists, skipping)"

echo "==> Creating Pub/Sub topic + subscription..."
gcloud pubsub topics create "${TOPIC}" \
  2>/dev/null || echo "    (topic already exists, skipping)"
gcloud pubsub subscriptions create "${SUBSCRIPTION}" \
  --topic="${TOPIC}" --ack-deadline=600 \
  2>/dev/null || echo "    (subscription already exists, skipping)"

echo "==> Creating Artifact Registry repo '${REPO}'..."
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${REGION}" \
  2>/dev/null || echo "    (repo already exists, skipping)"

echo "==> Creating runtime service account ${SA_EMAIL}..."
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="DocuSense Cloud Run runtime" \
  2>/dev/null || echo "    (service account already exists, skipping)"

echo "==> Granting roles to the service account..."
for ROLE in roles/aiplatform.user roles/storage.objectAdmin \
            roles/pubsub.publisher roles/pubsub.subscriber; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" --role="${ROLE}" \
    --condition=None --quiet >/dev/null
  echo "    granted ${ROLE}"
done

cat <<EOF

✅ GCP setup complete.

Next steps:
  1. Deploy the backend:
       gcloud builds submit --config cloudbuild.yaml .
     or directly:
       gcloud run deploy docusense-api \\
         --source . --region ${REGION} --allow-unauthenticated \\
         --service-account ${SA_EMAIL} --memory 1Gi \\
         --set-env-vars GCP_PROJECT_ID=${PROJECT_ID},GCP_LOCATION=${REGION},GCS_BUCKET=${BUCKET},VERTEX_AI_MOCK=false

  2. (Optional, production ingestion) point a push subscription at the service:
       gcloud pubsub subscriptions create docusense-ingest-push \\
         --topic ${TOPIC} \\
         --push-endpoint "https://<your-cloud-run-url>/pubsub/push"

  3. Run the UI locally against the deployed backend:
       BACKEND_URL=https://<your-cloud-run-url> streamlit run ui/streamlit_app.py
EOF
