"""Application settings, loaded from environment variables (or a .env file).

Every knob in the system lives here so that the same codebase runs in three
modes without code changes:

  1. Fully local mock mode  -> VERTEX_AI_MOCK=true   (zero GCP setup needed)
  2. Local dev against GCP  -> VERTEX_AI_MOCK=false + ADC credentials
  3. Cloud Run production   -> env vars set at deploy time
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- GCP project ------------------------------------------------------
    gcp_project_id: str = "your-gcp-project"
    gcp_location: str = "us-central1"

    # --- Mock mode: run everything locally with zero GCP dependencies ------
    vertex_ai_mock: bool = False

    # --- Vertex AI models ---------------------------------------------------
    embedding_model: str = "text-embedding-004"
    generation_model: str = "gemini-1.5-flash"

    # --- Vector store -------------------------------------------------------
    # "faiss" (local index, default) or "matching_engine" (Vertex AI, prod)
    vector_backend: str = "faiss"
    faiss_index_dir: str = "/tmp/docusense/index"
    matching_engine_index_id: str = ""
    matching_engine_endpoint_id: str = ""
    matching_engine_deployed_index_id: str = ""

    # --- Cloud Storage ------------------------------------------------------
    gcs_bucket: str = "docusense-documents"
    # Used instead of GCS when VERTEX_AI_MOCK=true
    local_storage_dir: str = "/tmp/docusense/storage"

    # --- Pub/Sub --------------------------------------------------------------
    pubsub_topic: str = "docusense-ingest"
    pubsub_subscription: str = "docusense-ingest-sub"
    # Start a background streaming-pull subscriber inside the API process.
    # On Cloud Run prefer the push endpoint (POST /pubsub/push) instead.
    enable_pull_subscriber: bool = False

    # --- Chunking (sizes are in tokens, approximated as ~4 chars/token) -----
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # --- Retrieval -----------------------------------------------------------
    top_k: int = 5

    # --- Serving ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    # Where the Streamlit UI finds the FastAPI backend
    backend_url: str = "http://localhost:8080"


@lru_cache
def get_settings() -> Settings:
    return Settings()
