"""Vertex AI wrapper: text-embedding-004 embeddings + Gemini 1.5 Flash generation.

When VERTEX_AI_MOCK=true this client returns deterministic fake responses
(seeded random embeddings, "Mock answer: ..." generations) so the entire
stack runs and is testable without any GCP credentials. All google-cloud
imports are lazy so mock mode never touches the GCP SDKs.
"""

import functools
import hashlib
import logging
import random
import time
from typing import Generator

from app.config import get_settings

logger = logging.getLogger(__name__)

# Output dimensionality of text-embedding-004
EMBEDDING_DIM = 768
# text-embedding-004 accepts up to 250 instances per request; stay well under
_EMBED_BATCH_SIZE = 100

PROMPT_TEMPLATE = """You are DocuSense, an assistant that answers questions using ONLY the document excerpts below.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

Answer using only the excerpts above. If the excerpts do not contain the answer, \
say "I could not find that in the uploaded documents." Be concise and mention the \
source filenames you relied on."""


def retry_with_backoff(max_attempts: int = 4, base_delay: float = 1.0):
    """Retry transient Vertex AI failures with exponential backoff (1s, 2s, 4s...)."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # Vertex raises many transient error types
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
    """Thin wrapper over the Vertex AI SDK with a fully-local mock mode."""

    def __init__(self, project_id: str | None = None, location: str | None = None):
        settings = get_settings()
        self.mock = settings.vertex_ai_mock
        self.project_id = project_id or settings.gcp_project_id
        self.location = location or settings.gcp_location
        self.embedding_model_name = settings.embedding_model
        self.generation_model_name = settings.generation_model
        self._embedding_model = None
        self._generative_model = None
        if not self.mock:
            import vertexai

            vertexai.init(project=self.project_id, location=self.location)

    # ------------------------------------------------------------------ #
    # Embeddings                                                          #
    # ------------------------------------------------------------------ #

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts with text-embedding-004 (768-dim vectors)."""
        if not texts:
            return []
        if self.mock:
            return [self._mock_embedding(t) for t in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start : start + _EMBED_BATCH_SIZE]))
        return vectors

    @retry_with_backoff()
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        model = self._get_embedding_model()
        return [emb.values for emb in model.get_embeddings(batch)]

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from vertexai.language_models import TextEmbeddingModel

            self._embedding_model = TextEmbeddingModel.from_pretrained(
                self.embedding_model_name
            )
        return self._embedding_model

    @staticmethod
    def _mock_embedding(text: str) -> list[float]:
        """Deterministic pseudo-embedding: same text always yields same vector."""
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    # ------------------------------------------------------------------ #
    # Generation (Gemini 1.5 Flash)                                       #
    # ------------------------------------------------------------------ #

    @retry_with_backoff()
    def generate_answer(self, context: str, question: str) -> str:
        """Answer a question grounded in the given context. Blocking version."""
        if self.mock:
            return f"Mock answer: {question}"
        model = self._get_generative_model()
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = model.generate_content(prompt)
        return response.text

    def stream_answer(self, context: str, question: str) -> Generator[str, None, None]:
        """Streaming version of generate_answer; yields text chunks as they arrive."""
        if self.mock:
            for word in f"Mock answer: {question}".split(" "):
                yield word + " "
            return
        model = self._get_generative_model()
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        for chunk in self._start_stream(model, prompt):
            # Chunks without candidates (e.g. final usage metadata) have no .text
            try:
                if chunk.text:
                    yield chunk.text
            except ValueError:
                continue

    @retry_with_backoff()
    def _start_stream(self, model, prompt: str):
        return model.generate_content(prompt, stream=True)

    def _get_generative_model(self):
        if self._generative_model is None:
            from vertexai.generative_models import GenerativeModel

            self._generative_model = GenerativeModel(self.generation_model_name)
        return self._generative_model
