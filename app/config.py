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

    # --- Google AI Studio API key (free tier, no billing/GCP project) ------
    # When set (and not in mock mode), Gemini + embeddings are called via
    # this key instead of Vertex AI, and GCS/PubSub fall back to local
    # equivalents. Get a key at https://aistudio.google.com/apikey
    gemini_api_key: str = ""

    # --- Vertex AI models ---------------------------------------------------
    embedding_model: str = "text-embedding-004"
    generation_model: str = "gemini-1.5-flash"

    # --- AI Studio models -----------------------------------------------------
    # The free Gemini API retired text-embedding-004 (Jan 2026) and the 1.5
    # series; these are their supported successors on that endpoint.
    ai_studio_embedding_model: str = "gemini-embedding-001"
    ai_studio_generation_model: str = "gemini-2.5-flash"

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

    @property
    def use_local_infra(self) -> bool:
        """True when GCS/PubSub should be replaced by local equivalents.

        Mock mode and AI-Studio-key mode both run without a GCP project.
        """
        return self.vertex_ai_mock or bool(self.gemini_api_key)

    @property
    def ai_backend(self) -> str:
        if self.vertex_ai_mock:
            return "mock"
        if self.gemini_api_key:
            return "ai_studio"
        return "vertex"


@lru_cache
def get_settings() -> Settings:
    return Settings()
