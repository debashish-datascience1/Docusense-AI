# 📄 DocuSense AI

A GCP-native **agentic RAG system**: upload PDF/text documents, ask questions in
natural language, get streaming answers from **Gemini 1.5 Flash** grounded in
your documents — with source citations and confidence scores.

> 💡 **Zero-GCP local mode:** set `VERTEX_AI_MOCK=true` and the whole stack
> (API, UI, ingestion, retrieval, tests) runs on your laptop with no Google
> Cloud account at all. See [Local development](#3-local-development-no-gcp-needed).

---

## 1. Architecture

```
                ┌────────────────────────────────────────────────────────────┐
                │                        Google Cloud                        │
                │                                                            │
 ┌───────────┐  │  ┌─────────────────┐         ┌────────────────────────┐    │
 │ Streamlit │  │  │   Cloud Run     │ publish │       Pub/Sub          │    │
 │    UI     │──┼─▶│  FastAPI API    │────────▶│  docusense-ingest      │    │
 │ (chat +   │  │  │                 │         └───────────┬────────────┘    │
 │  upload)  │◀─┼──│ /ingest /ask    │                     │ push/pull       │
 └───────────┘  │  │ /job /documents │◀────────────────────┘                 │
   streaming    │  └───┬─────────┬───┘   POST /pubsub/push                   │
   NDJSON       │      │         │                                           │
                │      ▼         ▼                                           │
                │  ┌────────┐ ┌──────────────────────────────┐               │
                │  │  GCS   │ │         Vertex AI            │               │
                │  │ bucket │ │  text-embedding-004 (embed)  │               │
                │  │ (docs, │ │  gemini-1.5-flash (generate) │               │
                │  │  jobs) │ │  Matching Engine (optional)  │               │
                │  └────────┘ └──────────────────────────────┘               │
                └────────────────────────────────────────────────────────────┘
                       │
                       ▼
                ┌──────────────┐
                │ FAISS index  │   local default; swap to Matching Engine
                │ (local disk) │   with VECTOR_BACKEND=matching_engine
                └──────────────┘
```

**Data flow**

1. **Ingest** — `POST /ingest` uploads the raw file to GCS, records a job, and
   publishes a Pub/Sub message.
2. **Process** — the Pub/Sub subscriber (or push endpoint) downloads the file,
   extracts text, chunks it (512 tokens, 50 overlap), embeds chunks with
   `text-embedding-004`, and upserts vectors into FAISS / Matching Engine.
3. **Ask** — `POST /ask` embeds the question, retrieves the top-k chunks,
   builds a grounded prompt, and streams Gemini 1.5 Flash's answer back as
   NDJSON (sources first, then tokens).

### What each GCP service does here

| Service | Role in DocuSense |
|---|---|
| **Vertex AI — Gemini 1.5 Flash** | Generates the final answer, grounded in retrieved chunks |
| **Vertex AI — text-embedding-004** | Turns chunks and questions into 768-dim vectors |
| **Vertex AI — Matching Engine** | Optional managed vector search for production scale (replaces local FAISS) |
| **Cloud Storage** | Persists raw documents, ingestion-job state, and (for Matching Engine) chunk payloads |
| **Pub/Sub** | Decouples upload from processing so ingestion is async and retryable |
| **Cloud Run** | Serverless hosting for the FastAPI backend (scales to zero) |
| **Cloud Build** | CI: runs tests, builds the image, deploys to Cloud Run (`cloudbuild.yaml`) |
| **Artifact Registry** | Stores the built container images |

---

## 2. GCP free-tier setup (step by step)

Everything below fits the GCP free tier / new-account credits for light usage.

1. **Create an account & project**
   - Sign up at <https://console.cloud.google.com> (new accounts get $300 credits).
   - Create a project, note its ID (e.g. `docusense-demo-123`).
2. **Install the gcloud CLI** — <https://cloud.google.com/sdk/docs/install>, then:
   ```bash
   gcloud auth login
   gcloud auth application-default login   # local credentials for the SDK
   ```
3. **Enable billing** on the project (required for Vertex AI, still covered by credits).
4. **Run the one-command setup script** — enables APIs and creates the bucket,
   Pub/Sub topic + subscription, Artifact Registry repo, and a least-privilege
   runtime service account:
   ```bash
   ./scripts/setup_gcp.sh <PROJECT_ID> us-central1
   ```
5. **Cost notes**
   - Gemini 1.5 Flash and text-embedding-004 are pay-per-use and very cheap at
     demo scale; Cloud Run scales to zero; Pub/Sub and GCS have generous free tiers.
   - **Matching Engine is NOT free** (always-on endpoint). Keep the default
     `VECTOR_BACKEND=faiss` unless you need managed vector search.

---

## 3. Local development (no GCP needed!)

Mock mode replaces every GCP dependency: embeddings become deterministic
seeded vectors, Gemini returns `Mock answer: <question>`, Pub/Sub becomes an
in-memory queue, and GCS writes to `/tmp/docusense/storage`.

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the backend in mock mode
VERTEX_AI_MOCK=true uvicorn app.main:app --reload --port 8080

# 3. In another terminal, start the UI
BACKEND_URL=http://localhost:8080 streamlit run ui/streamlit_app.py

# 4. Run the tests (also fully offline)
pytest tests/ -v
```

Open <http://localhost:8501>, upload a PDF or text file from the sidebar, and
chat with it. Answers will be mocked, but the full pipeline — upload → GCS →
Pub/Sub → chunk → embed → FAISS → retrieve → stream — actually runs.

To run locally **against real GCP** instead (after `setup_gcp.sh`):

```bash
export VERTEX_AI_MOCK=false
export GCP_PROJECT_ID=<PROJECT_ID>
export GCS_BUCKET=<PROJECT_ID>-docusense-documents
export ENABLE_PULL_SUBSCRIBER=true   # process Pub/Sub jobs in-process
uvicorn app.main:app --port 8080
```

---

## 4. Deploy to Cloud Run

**Option A — one-off deploy from source:**

```bash
gcloud run deploy docusense-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --service-account docusense-run@<PROJECT_ID>.iam.gserviceaccount.com \
  --memory 1Gi \
  --set-env-vars GCP_PROJECT_ID=<PROJECT_ID>,GCP_LOCATION=us-central1,GCS_BUCKET=<PROJECT_ID>-docusense-documents,VERTEX_AI_MOCK=false
```

**Option B — CI via Cloud Build** (tests → build → push → deploy):

```bash
gcloud builds submit --config cloudbuild.yaml .
```

**Production ingestion:** point a Pub/Sub *push* subscription at the deployed
service so Cloud Run instances are woken to process jobs:

```bash
gcloud pubsub subscriptions create docusense-ingest-push \
  --topic docusense-ingest \
  --push-endpoint "https://<cloud-run-url>/pubsub/push"
```

Then run the UI anywhere with `BACKEND_URL=https://<cloud-run-url>`.

> ⚠️ Cloud Run note: the default FAISS backend keeps its index on instance-local
> disk, which is ephemeral and per-instance. Fine for demos (set
> `--max-instances 1`); for production switch to
> `VECTOR_BACKEND=matching_engine` after creating a streaming-update Matching
> Engine index + endpoint, and set `MATCHING_ENGINE_INDEX_ID`,
> `MATCHING_ENGINE_ENDPOINT_ID`, `MATCHING_ENGINE_DEPLOYED_INDEX_ID`.

---

## 5. API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Multipart file upload (`.pdf` `.txt` `.md`, ≤20 MB) → `{job_id}` |
| `GET` | `/job/{job_id}` | Job status: `queued` → `processing` → `done`/`failed` |
| `POST` | `/ask` | `{"question": "...", "top_k": 5, "stream": true}` → NDJSON stream (`sources`, `token`×N, `done`); `"stream": false` → plain JSON |
| `GET` | `/documents` | All ingested documents with status + chunk counts |
| `GET` | `/health` | Health check (used by the container HEALTHCHECK) |
| `POST` | `/pubsub/push` | Pub/Sub push delivery target (production ingestion) |

## 6. Configuration

All settings are env vars (see [app/config.py](app/config.py)). Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `VERTEX_AI_MOCK` | `false` | `true` = fully local, zero GCP |
| `GCP_PROJECT_ID` / `GCP_LOCATION` | — / `us-central1` | Vertex AI + GCS + Pub/Sub project |
| `VECTOR_BACKEND` | `faiss` | `faiss` or `matching_engine` |
| `GCS_BUCKET` | `docusense-documents` | Document/job storage bucket |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `512` / `50` | Chunking parameters |
| `BACKEND_URL` | `http://localhost:8080` | Where the Streamlit UI finds the API |

## 7. Project layout

```
docusense-ai/
├── app/
│   ├── main.py              # FastAPI app + endpoints
│   ├── vertex_client.py     # Vertex AI wrapper (embeddings + Gemini, mockable)
│   ├── vector_store.py      # FAISS + Matching Engine adapter
│   ├── pubsub_handler.py    # Pub/Sub publisher + subscriber (mockable)
│   ├── gcs_handler.py       # Cloud Storage upload/download (mockable)
│   ├── rag_pipeline.py      # Ingest → chunk → embed → index → retrieve → generate
│   └── config.py            # Settings from env vars
├── ui/streamlit_app.py      # Chat UI with upload + streaming answers + citations
├── scripts/setup_gcp.sh     # One-command GCP project setup
├── tests/test_rag_pipeline.py  # Offline test suite (mock mode)
├── Dockerfile               # python:3.11-slim, non-root, /health healthcheck
├── cloudbuild.yaml          # CI: test → build → push → deploy
└── requirements.txt
```

## 8. Screenshots

> 🖼️ *Placeholder — add screenshots here:*
>
> ![Upload & ingestion](docs/screenshots/upload.png)
> ![Chat with citations](docs/screenshots/chat.png)
