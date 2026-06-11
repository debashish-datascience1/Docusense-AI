"""Tests for the DocuSense RAG pipeline. Run entirely in mock mode — no GCP needed.

    VERTEX_AI_MOCK=true pytest tests/ -v

(The fixture below forces mock mode regardless, so a bare `pytest` also works.)
"""

import os
import time

import pytest

# Must be set before any app module reads settings
os.environ["VERTEX_AI_MOCK"] = "true"

from app import config, pubsub_handler, rag_pipeline, vector_store  # noqa: E402
from app.vector_store import FaissVectorStore  # noqa: E402
from app.vertex_client import EMBEDDING_DIM, VertexClient  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Each test gets its own storage/index dirs and fresh singletons."""
    monkeypatch.setenv("VERTEX_AI_MOCK", "true")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "index"))
    config.get_settings.cache_clear()
    vector_store.get_vector_store.cache_clear()
    rag_pipeline.get_pipeline.cache_clear()
    pubsub_handler.reset_mock_queue()
    yield
    config.get_settings.cache_clear()
    vector_store.get_vector_store.cache_clear()
    rag_pipeline.get_pipeline.cache_clear()


# --------------------------------------------------------------------- #
# VertexClient (mock mode)                                                #
# --------------------------------------------------------------------- #


def test_mock_embeddings_are_deterministic():
    client = VertexClient()
    first = client.embed_text(["hello world", "another text"])
    second = client.embed_text(["hello world", "another text"])
    assert first == second
    assert len(first) == 2
    assert all(len(vec) == EMBEDDING_DIM for vec in first)
    assert first[0] != first[1]


def test_mock_generate_answer():
    client = VertexClient()
    assert client.generate_answer("ctx", "What is X?") == "Mock answer: What is X?"


def test_mock_stream_answer():
    client = VertexClient()
    streamed = "".join(client.stream_answer("ctx", "What is X?"))
    assert streamed.strip() == "Mock answer: What is X?"


# --------------------------------------------------------------------- #
# FAISS vector store                                                      #
# --------------------------------------------------------------------- #


def test_faiss_upsert_and_search(tmp_path):
    store = FaissVectorStore(index_dir=str(tmp_path / "faiss"))
    client = VertexClient()
    texts = ["the sky is blue", "grass is green", "the ocean is deep"]
    vectors = client.embed_text(texts)
    store.upsert(
        ids=[f"c{i}" for i in range(3)],
        vectors=vectors,
        metadatas=[{"text": t, "filename": "facts.txt"} for t in texts],
    )
    assert store.count() == 3

    # The exact text must be its own nearest neighbour (cosine sim == 1)
    results = store.search(client.embed_text(["grass is green"])[0], top_k=1)
    assert results[0].text == "grass is green"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[0].metadata["filename"] == "facts.txt"


def test_faiss_persistence(tmp_path):
    index_dir = str(tmp_path / "faiss")
    client = VertexClient()
    store = FaissVectorStore(index_dir=index_dir)
    store.upsert(["a"], client.embed_text(["persisted chunk"]), [{"text": "persisted chunk"}])

    reloaded = FaissVectorStore(index_dir=index_dir)
    assert reloaded.count() == 1
    results = reloaded.search(client.embed_text(["persisted chunk"])[0], top_k=1)
    assert results[0].text == "persisted chunk"


def test_faiss_empty_search(tmp_path):
    store = FaissVectorStore(index_dir=str(tmp_path / "faiss"))
    assert store.search([0.0] * EMBEDDING_DIM, top_k=5) == []


# --------------------------------------------------------------------- #
# Pipeline: ingest -> process -> answer                                   #
# --------------------------------------------------------------------- #


def _make_pipeline():
    return rag_pipeline.get_pipeline()


def test_ingest_creates_queued_job():
    pipeline = _make_pipeline()
    job_id = pipeline.ingest_document(b"Some document text.", "doc.txt")
    job = pipeline.get_job_status(job_id)
    # The mock subscriber thread may already have picked the job up
    assert job["status"] in ("queued", "processing", "done")
    assert job["filename"] == "doc.txt"


def test_full_ingest_and_answer_flow():
    pipeline = _make_pipeline()
    text = (
        "DocuSense is a retrieval augmented generation system. "
        "It stores document embeddings in a FAISS index. "
        "Questions are answered by Gemini 1.5 Flash grounded in retrieved chunks."
    )
    job_id = pipeline.ingest_document(text.encode(), "about.txt")
    pipeline.process_ingestion_job(job_id)  # synchronous, deterministic

    job = pipeline.get_job_status(job_id)
    assert job["status"] == "done"
    assert job["chunks"] >= 1

    result = pipeline.answer_question("What index does DocuSense use?")
    assert result["answer"] == "Mock answer: What index does DocuSense use?"
    assert result["sources"], "expected at least one source"
    assert result["sources"][0]["filename"] == "about.txt"
    assert 0.0 <= result["confidence"] <= 1.0


def test_answer_with_no_documents():
    pipeline = _make_pipeline()
    result = pipeline.answer_question("Anything there?")
    assert result["sources"] == []
    assert result["confidence"] == 0.0
    assert "No documents" in result["answer"]


def test_failed_job_records_error():
    pipeline = _make_pipeline()
    # Empty content -> "No extractable text" failure path
    job_id = pipeline.ingest_document(b"   ", "empty.txt")
    pipeline.process_ingestion_job(job_id)
    job = pipeline.get_job_status(job_id)
    assert job["status"] == "failed"
    assert "error" in job


def test_chunking_long_document():
    pipeline = _make_pipeline()
    long_text = ("Sentence number %d about testing. " * 500) % tuple(range(500))
    job_id = pipeline.ingest_document(long_text.encode(), "long.txt")
    pipeline.process_ingestion_job(job_id)
    job = pipeline.get_job_status(job_id)
    assert job["status"] == "done"
    assert job["chunks"] > 1, "a long document must produce multiple chunks"


# --------------------------------------------------------------------- #
# FastAPI endpoints                                                       #
# --------------------------------------------------------------------- #


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mock"] is True


def test_job_not_found(client):
    assert client.get("/job/nonexistent").status_code == 404


def test_ingest_rejects_bad_extension(client):
    response = client.post("/ingest", files={"file": ("evil.exe", b"binary")})
    assert response.status_code == 400


def test_api_ingest_then_ask(client):
    response = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"The launch code is in the blue folder.")},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # Mock Pub/Sub worker processes asynchronously; poll until done
    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/job/{job_id}").json()["status"]
        if status in ("done", "failed"):
            break
        time.sleep(0.05)
    assert status == "done"

    docs = client.get("/documents").json()["documents"]
    assert any(d["filename"] == "notes.txt" and d["status"] == "done" for d in docs)

    response = client.post(
        "/ask", json={"question": "Where is the launch code?", "stream": False}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Mock answer: Where is the launch code?"
    assert body["sources"]


def test_api_ask_streaming(client):
    # Index something first (synchronously, via the pipeline)
    pipeline = rag_pipeline.get_pipeline()
    job_id = pipeline.ingest_document(b"Streaming test document content.", "s.txt")
    pipeline.process_ingestion_job(job_id)

    import json as jsonlib

    with client.stream(
        "POST", "/ask", json={"question": "What is this?", "stream": True}
    ) as response:
        assert response.status_code == 200
        events = [jsonlib.loads(line) for line in response.iter_lines() if line.strip()]

    types = [e["type"] for e in events]
    assert types[0] == "sources"
    assert types[-1] == "done"
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer.strip() == "Mock answer: What is this?"
