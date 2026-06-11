"""FastAPI backend for DocuSense AI.

Endpoints:
  POST /ingest        upload a document, returns a job_id (async ingestion)
  GET  /job/{job_id}  ingestion job status
  POST /ask           question answering (NDJSON streaming by default)
  GET  /documents     list ingested documents
  GET  /health        health/liveness check
  POST /pubsub/push   Pub/Sub push delivery target (production ingestion)
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.pubsub_handler import PubSubHandler
from app.rag_pipeline import get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    pipeline = get_pipeline()
    # Mock mode always processes via the in-memory queue worker. In real mode
    # a pull subscriber is opt-in; Cloud Run should use /pubsub/push instead.
    if settings.vertex_ai_mock or settings.enable_pull_subscriber:
        pipeline.pubsub.start_subscriber(pipeline.process_ingestion_job)
    logger.info(
        "DocuSense API started (mock=%s, vector_backend=%s)",
        settings.vertex_ai_mock,
        settings.vector_backend,
    )
    yield


app = FastAPI(title="DocuSense AI", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = True


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "mock": settings.vertex_ai_mock,
        "vector_backend": settings.vector_backend,
    }


@app.post("/ingest")
async def ingest(file: UploadFile) -> dict:
    filename = file.filename or "upload.txt"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit")
    job_id = get_pipeline().ingest_document(data, filename)
    return {"job_id": job_id, "status": "queued", "filename": filename}


@app.get("/job/{job_id}")
def job_status(job_id: str) -> dict:
    job = get_pipeline().get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
    return job


@app.post("/ask")
def ask(request: AskRequest):
    pipeline = get_pipeline()
    if not request.stream:
        return pipeline.answer_question(request.question, top_k=request.top_k)

    def event_stream():
        for event in pipeline.stream_answer_events(
            request.question, top_k=request.top_k
        ):
            yield json.dumps(event) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/documents")
def documents() -> dict:
    return {"documents": get_pipeline().list_documents()}


@app.post("/pubsub/push", status_code=204)
async def pubsub_push(request: Request) -> None:
    """Pub/Sub push subscription target: processes one ingestion job."""
    envelope = await request.json()
    try:
        job_id = PubSubHandler.parse_push_message(envelope)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Bad Pub/Sub envelope: {exc}")
    get_pipeline().process_ingestion_job(job_id)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port)
