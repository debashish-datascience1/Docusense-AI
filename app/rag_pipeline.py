"""End-to-end RAG pipeline: ingest -> chunk -> embed -> index -> retrieve -> generate.

Ingestion is asynchronous: ingest_document() uploads the raw file to GCS and
publishes a Pub/Sub message; process_ingestion_job() (invoked by the Pub/Sub
subscriber or the /pubsub/push endpoint) does the heavy lifting. Job state is
persisted as JSON in GCS so any instance can answer GET /job/{job_id}.
"""

import logging
import uuid
from functools import lru_cache
from io import BytesIO
from typing import Generator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.gcs_handler import GCSHandler
from app.pubsub_handler import PubSubHandler
from app.vector_store import VectorStore, get_vector_store
from app.vertex_client import VertexClient

logger = logging.getLogger(__name__)

# text-embedding-004 has no public tokenizer; ~4 chars/token is the standard
# approximation, applied to the configured token-based chunk sizes.
_CHARS_PER_TOKEN = 4


class RAGPipeline:
    def __init__(
        self,
        vertex_client: VertexClient | None = None,
        vector_store: VectorStore | None = None,
        gcs: GCSHandler | None = None,
        pubsub: PubSubHandler | None = None,
    ):
        settings = get_settings()
        self.vertex = vertex_client or VertexClient()
        self.store = vector_store or get_vector_store()
        self.gcs = gcs or GCSHandler()
        self.pubsub = pubsub or PubSubHandler()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size_tokens * _CHARS_PER_TOKEN,
            chunk_overlap=settings.chunk_overlap_tokens * _CHARS_PER_TOKEN,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------ #
    # Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def ingest_document(self, file_bytes: bytes, filename: str) -> str:
        """Upload a document and queue it for async processing. Returns job_id."""
        job_id = uuid.uuid4().hex
        self.gcs.upload_bytes(f"uploads/{job_id}/{filename}", file_bytes)
        self._write_job(job_id, status="queued", filename=filename)
        self.pubsub.publish_ingestion_job(job_id)
        logger.info("Queued ingestion job %s for %s", job_id, filename)
        return job_id

    def process_ingestion_job(self, job_id: str) -> None:
        """Download, chunk, embed and index a queued document (subscriber side)."""
        job = self.get_job_status(job_id)
        if job is None:
            logger.error("Unknown ingestion job %s", job_id)
            return
        filename = job["filename"]
        self._write_job(job_id, status="processing", filename=filename)
        try:
            raw = self.gcs.download_bytes(f"uploads/{job_id}/{filename}")
            text = extract_text(raw, filename)
            chunks = [c for c in self.splitter.split_text(text) if c.strip()]
            if not chunks:
                raise ValueError("No extractable text found in document")
            vectors = self.vertex.embed_text(chunks)
            ids = [f"{job_id}-{i}" for i in range(len(chunks))]
            metadatas = [
                {"text": chunk, "filename": filename, "job_id": job_id, "chunk": i}
                for i, chunk in enumerate(chunks)
            ]
            self.store.upsert(ids, vectors, metadatas)
            self._write_job(
                job_id, status="done", filename=filename, chunks=len(chunks)
            )
            logger.info("Job %s done: %d chunks indexed", job_id, len(chunks))
        except Exception as exc:
            logger.exception("Ingestion job %s failed", job_id)
            self._write_job(job_id, status="failed", filename=filename, error=str(exc))

    def _write_job(self, job_id: str, **fields) -> None:
        self.gcs.upload_json(f"jobs/{job_id}.json", {"job_id": job_id, **fields})

    def get_job_status(self, job_id: str) -> dict | None:
        path = f"jobs/{job_id}.json"
        if not self.gcs.exists(path):
            return None
        return self.gcs.download_json(path)

    def delete_document(self, job_id: str) -> dict | None:
        """Remove a document: its chunks from the index, its file and job record.

        Returns a summary dict, or None if the job_id is unknown.
        """
        job = self.get_job_status(job_id)
        if job is None:
            return None
        # Chunk ids are deterministic ({job_id}-{i}), so the job record is
        # enough to address every vector this document produced.
        ids = [f"{job_id}-{i}" for i in range(int(job.get("chunks", 0)))]
        removed = self.store.delete(ids) if ids else 0
        filename = job.get("filename", "")
        if filename:
            self.gcs.delete_file(f"uploads/{job_id}/{filename}")
        self.gcs.delete_file(f"jobs/{job_id}.json")
        logger.info("Deleted document %s (%s): %d chunks", job_id, filename, removed)
        return {"job_id": job_id, "filename": filename, "removed_chunks": removed}

    def list_documents(self) -> list[dict]:
        """All ingestion jobs with their status (the de-facto document registry)."""
        docs = []
        for path in self.gcs.list_files("jobs/"):
            try:
                docs.append(self.gcs.download_json(path))
            except Exception:
                logger.warning("Skipping unreadable job record %s", path)
        return docs

    # ------------------------------------------------------------------ #
    # Question answering                                                  #
    # ------------------------------------------------------------------ #

    def answer_question(self, question: str, top_k: int = 5) -> dict:
        """Retrieve top_k chunks and generate a grounded answer."""
        retrieval = self._retrieve(question, top_k)
        if retrieval is None:
            return {
                "answer": "No documents have been ingested yet. Upload a document first.",
                "sources": [],
                "confidence": 0.0,
            }
        context, sources, confidence = retrieval
        answer = self.vertex.generate_answer(context, question)
        return {"answer": answer, "sources": sources, "confidence": confidence}

    def stream_answer_events(
        self, question: str, top_k: int = 5
    ) -> Generator[dict, None, None]:
        """Streaming variant: yields a sources event, then token events, then done."""
        retrieval = self._retrieve(question, top_k)
        if retrieval is None:
            yield {"type": "sources", "sources": [], "confidence": 0.0}
            yield {
                "type": "token",
                "text": "No documents have been ingested yet. Upload a document first.",
            }
            yield {"type": "done"}
            return
        context, sources, confidence = retrieval
        yield {"type": "sources", "sources": sources, "confidence": confidence}
        for text in self.vertex.stream_answer(context, question):
            yield {"type": "token", "text": text}
        yield {"type": "done"}

    def _retrieve(
        self, question: str, top_k: int
    ) -> tuple[str, list[dict], float] | None:
        """Embed the question and fetch supporting chunks.

        Returns (context, sources, confidence), or None when the index is empty.
        """
        query_vector = self.vertex.embed_text([question])[0]
        results = self.store.search(query_vector, top_k=top_k)
        if not results:
            return None
        context = "\n\n---\n\n".join(
            f"[Source: {r.metadata.get('filename', 'unknown')}]\n{r.text}"
            for r in results
        )
        sources = [
            {
                "chunk_id": r.chunk_id,
                "filename": r.metadata.get("filename", "unknown"),
                "score": round(r.score, 4),
                "snippet": r.text[:200],
            }
            for r in results
        ]
        # Cosine similarity of retrieved chunks, clamped to [0, 1]
        confidence = round(max(0.0, sum(r.score for r in results) / len(results)), 4)
        return context, sources, confidence


def extract_text(raw: bytes, filename: str) -> str:
    """Extract plain text from PDF or text-like uploads."""
    if filename.lower().endswith(".pdf"):
        from PyPDF2 import PdfReader

        reader = PdfReader(BytesIO(raw))
        # PyPDF2 often emits a newline between every word; collapse runs of
        # whitespace per page so chunks and citation snippets read normally.
        pages = (
            " ".join((page.extract_text() or "").split()) for page in reader.pages
        )
        return "\n\n".join(p for p in pages if p)
    return raw.decode("utf-8", errors="replace")


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()
