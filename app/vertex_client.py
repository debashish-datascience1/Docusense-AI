"""LLM + embedding client supporting four backends.

Backend      Generation              Embeddings               Keys needed
---------    ---------------------   ----------------------   ----------------
mock         seeded fake text        seeded fake vectors      none
groq         Groq API (LLaMA etc.)   sentence-transformers    GROQ_API_KEY
             + optionally Gemini     or Gemini embeddings     + GEMINI_API_KEY
ai_studio    Gemini via AI Studio    gemini-embedding-001     GEMINI_API_KEY
vertex       Gemini via Vertex AI    text-embedding-004       GCP ADC creds

Backend selection is automatic based on which keys exist — see config.ai_backend
and config.embedding_backend. All SDK imports are lazy so unused backends never
import their dependencies.
"""

import functools
import hashlib
import logging
import random
import time
from typing import Generator

from app.config import get_settings

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 100

SYSTEM_PROMPT = (
    "You are DocuSense, a helpful assistant that answers questions using ONLY "
    "the document excerpts provided. If the excerpts do not contain the answer, "
    "say \"I could not find that in the uploaded documents.\" "
    "Be concise and mention the source filenames you relied on."
)

PROMPT_TEMPLATE = """DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

Answer using only the excerpts above."""


def retry_with_backoff(max_attempts: int = 4, base_delay: float = 1.0):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        break
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "%s failed (%s: %s), retrying in %.1fs",
                        fn.__name__, type(exc).__name__, exc, delay,
                    )
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


class VertexClient:
    """Multi-backend LLM + embedding client."""

    def __init__(self, project_id: str | None = None, location: str | None = None):
        settings = get_settings()
        self.backend = settings.ai_backend
        self.embedding_backend = settings.embedding_backend
        self.embedding_dim = settings.embedding_dim
        self.mock = self.backend == "mock"

        self.project_id = project_id or settings.gcp_project_id
        self.location = location or settings.gcp_location

        # Model names (generation)
        self.generation_model_name = {
            "mock":      "mock",
            "groq":      settings.groq_model,
            "ai_studio": settings.ai_studio_generation_model,
            "vertex":    settings.generation_model,
        }[self.backend]

        # Model names (embeddings)
        self.embedding_model_name = {
            "mock":      "mock",
            "local":     settings.local_embedding_model,
            "ai_studio": settings.ai_studio_embedding_model,
            "vertex":    settings.embedding_model,
        }[self.embedding_backend]

        self._embedding_model = None   # Vertex TextEmbeddingModel
        self._generative_model = None  # Vertex GenerativeModel
        self._genai_client = None      # AI Studio genai.Client
        self._groq_client = None       # Groq client
        self._st_model = None          # sentence-transformers model

        if self.backend == "groq":
            from groq import Groq
            self._groq_client = Groq(api_key=settings.groq_api_key)

        if self.embedding_backend == "ai_studio":
            from google import genai
            self._genai_client = genai.Client(api_key=settings.gemini_api_key)

        elif self.embedding_backend == "local":
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(settings.local_embedding_model)

        elif self.embedding_backend == "vertex":
            import vertexai
            vertexai.init(project=self.project_id, location=self.location)

        # Also init Vertex for generation when backend == vertex
        if self.backend == "vertex" and self.embedding_backend != "vertex":
            import vertexai
            vertexai.init(project=self.project_id, location=self.location)

    # ------------------------------------------------------------------ #
    # Embeddings                                                          #
    # ------------------------------------------------------------------ #

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.embedding_backend == "mock":
            return [self._mock_embedding(t) for t in texts]
        if self.embedding_backend == "local":
            return self._embed_local(texts)
        # ai_studio or vertex — batched
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start: start + _EMBED_BATCH_SIZE]))
        return vectors

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """sentence-transformers local inference; no API call, no rate limit."""
        embeddings = self._st_model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vec)) for vec in embeddings]

    @retry_with_backoff()
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if self.embedding_backend == "ai_studio":
            from google.genai import types
            result = self._genai_client.models.embed_content(
                model=self.embedding_model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.embedding_dim
                ),
            )
            return [list(emb.values) for emb in result.embeddings]
        # vertex
        model = self._get_vertex_embedding_model()
        return [emb.values for emb in model.get_embeddings(batch)]

    def _get_vertex_embedding_model(self):
        if self._embedding_model is None:
            from vertexai.language_models import TextEmbeddingModel
            self._embedding_model = TextEmbeddingModel.from_pretrained(
                self.embedding_model_name
            )
        return self._embedding_model

    def _mock_embedding(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding: same text always yields same vector."""
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.embedding_dim)]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    # ------------------------------------------------------------------ #
    # Generation                                                          #
    # ------------------------------------------------------------------ #

    @retry_with_backoff()
    def generate_answer(self, context: str, question: str) -> str:
        if self.mock:
            return f"Mock answer: {question}"
        if self.backend == "groq":
            resp = self._groq_client.chat.completions.create(
                model=self.generation_model_name,
                messages=self._build_messages(context, question),
            )
            return resp.choices[0].message.content
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        if self.backend == "ai_studio":
            return self._genai_client.models.generate_content(
                model=self.generation_model_name, contents=prompt
            ).text
        return self._get_vertex_generative_model().generate_content(prompt).text

    def stream_answer(self, context: str, question: str) -> Generator[str, None, None]:
        if self.mock:
            for word in f"Mock answer: {question}".split():
                yield word + " "
            return
        if self.backend == "groq":
            stream = self._start_groq_stream(context, question)
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    yield text
            return
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        if self.backend == "ai_studio":
            for chunk in self._start_ai_studio_stream(prompt):
                if chunk.text:
                    yield chunk.text
            return
        for chunk in self._start_vertex_stream(prompt):
            try:
                if chunk.text:
                    yield chunk.text
            except ValueError:
                continue

    @retry_with_backoff()
    def _start_groq_stream(self, context: str, question: str):
        return self._groq_client.chat.completions.create(
            model=self.generation_model_name,
            messages=self._build_messages(context, question),
            stream=True,
        )

    @retry_with_backoff()
    def _start_ai_studio_stream(self, prompt: str):
        return self._genai_client.models.generate_content_stream(
            model=self.generation_model_name, contents=prompt
        )

    @retry_with_backoff()
    def _start_vertex_stream(self, prompt: str):
        return self._get_vertex_generative_model().generate_content(
            prompt, stream=True
        )

    def _get_vertex_generative_model(self):
        if self._generative_model is None:
            from vertexai.generative_models import GenerativeModel
            self._generative_model = GenerativeModel(self.generation_model_name)
        return self._generative_model

    @staticmethod
    def _build_messages(context: str, question: str) -> list[dict]:
        """OpenAI-style chat messages for Groq (and future OpenAI-compat providers)."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(context=context, question=question),
            },
        ]
