"""Application settings loaded from environment variables or a .env file.

Create a .env file in the project root (it is git-ignored) and put your keys
there — never pass secrets on the command line or export them into shell history.
See .env.example for every available setting.

Run modes (selected automatically from which keys are present):
  VERTEX_AI_MOCK=true          → mock   (zero setup)
  GROQ_API_KEY=<key>           → groq   (free tier, local embeddings)
  GROQ_API_KEY + GEMINI_API_KEY → groq  (free tier, Gemini embeddings)
  GEMINI_API_KEY=<key>         → ai_studio (free tier, full Gemini stack)
  (none of the above)          → vertex (production, GCP billing required)
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- GCP project (only needed for vertex backend) --------------------
    gcp_project_id: str = "your-gcp-project"
    gcp_location: str = "us-central1"

    # --- Mock mode -------------------------------------------------------
    vertex_ai_mock: bool = False

    # --- API keys (put these in .env, never on the command line) ---------
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # --- Vertex AI models ------------------------------------------------
    embedding_model: str = "text-embedding-004"
    generation_model: str = "gemini-1.5-flash"

    # --- AI Studio models (different generations than Vertex AI) ---------
    ai_studio_embedding_model: str = "gemini-embedding-001"
    ai_studio_generation_model: str = "gemini-2.5-flash"

    # --- Groq models -----------------------------------------------------
    groq_model: str = "llama-3.3-70b-versatile"
    # sentence-transformers model used for local embeddings in Groq-only mode
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # --- Vector store ----------------------------------------------------
    vector_backend: str = "faiss"
    faiss_index_dir: str = "/tmp/docusense/index"
    matching_engine_index_id: str = ""
    matching_engine_endpoint_id: str = ""
    matching_engine_deployed_index_id: str = ""

    # --- Cloud Storage ---------------------------------------------------
    gcs_bucket: str = "docusense-documents"
    local_storage_dir: str = "/tmp/docusense/storage"

    # --- Pub/Sub ---------------------------------------------------------
    pubsub_topic: str = "docusense-ingest"
    pubsub_subscription: str = "docusense-ingest-sub"
    enable_pull_subscriber: bool = False

    # --- Chunking --------------------------------------------------------
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # --- Retrieval -------------------------------------------------------
    top_k: int = 5

    # --- Serving ---------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    backend_url: str = "http://localhost:8080"

    # ------------------------------------------------------------------ #
    # Derived properties — read these, not the raw fields above           #
    # ------------------------------------------------------------------ #

    @property
    def ai_backend(self) -> str:
        """Which LLM provider handles generation."""
        if self.vertex_ai_mock:
            return "mock"
        if self.groq_api_key:
            return "groq"
        if self.gemini_api_key:
            return "ai_studio"
        return "vertex"

    @property
    def embedding_backend(self) -> str:
        """Which provider handles embeddings.

        Groq has no embedding API; fall back to Gemini if the key is present,
        otherwise use local sentence-transformers (no external call needed).
        """
        if self.vertex_ai_mock:
            return "mock"
        if self.gemini_api_key:
            return "ai_studio"   # prefer Gemini embeddings when available
        if self.groq_api_key:
            return "local"       # Groq-only → sentence-transformers locally
        return "vertex"

    @property
    def embedding_dim(self) -> int:
        """Vector dimensionality produced by the active embedding backend.

        ⚠️  Changing backends changes this value. Clear /tmp/docusense/index
        before switching so the FAISS index is rebuilt at the correct size.
        """
        return 384 if self.embedding_backend == "local" else 768

    @property
    def use_local_infra(self) -> bool:
        """True when GCS/PubSub should be replaced by local equivalents."""
        return self.ai_backend in ("mock", "groq") or bool(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
