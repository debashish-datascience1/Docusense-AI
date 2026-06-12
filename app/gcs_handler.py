"""Google Cloud Storage handler for document and job-state persistence.

In mock mode (VERTEX_AI_MOCK=true) files are written under local_storage_dir
(default /tmp/docusense/storage) with the same path layout as the GCS bucket,
so the rest of the pipeline is identical in both modes.

Bucket layout:
  uploads/{job_id}/{filename}   raw uploaded documents
  jobs/{job_id}.json            ingestion job status records
  chunks/{chunk_id}.json        chunk payloads (Matching Engine backend only)
"""

import json
import logging
import uuid
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


class GCSHandler:
    def __init__(self, bucket_name: str | None = None):
        settings = get_settings()
        # Local storage in mock mode AND in AI-Studio-key mode (no GCP project)
        self.mock = settings.use_local_infra
        self.bucket_name = bucket_name or settings.gcs_bucket
        if self.mock:
            self._root = Path(settings.local_storage_dir)
            self._root.mkdir(parents=True, exist_ok=True)
            self._bucket = None
        else:
            from google.cloud import storage

            self._client = storage.Client(project=settings.gcp_project_id)
            self._bucket = self._client.bucket(self.bucket_name)

    # ------------------------------------------------------------------ #

    def upload_bytes(
        self, path: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        """Write bytes to gs://bucket/path (or the local mirror in mock mode)."""
        if self.mock:
            target = self._root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write mirroring GCS semantics: job-status files are
            # polled and rewritten by concurrent threads. The temp name must
            # be unique per writer or two simultaneous writers race on it.
            tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
            return f"local://{target}"
        blob = self._bucket.blob(path)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self.bucket_name}/{path}"

    def download_bytes(self, path: str) -> bytes:
        if self.mock:
            return (self._root / path).read_bytes()
        return self._bucket.blob(path).download_as_bytes()

    def delete_file(self, path: str) -> bool:
        """Delete an object; returns False if it didn't exist."""
        if self.mock:
            target = self._root / path
            if not target.exists():
                return False
            target.unlink()
            return True
        blob = self._bucket.blob(path)
        if not blob.exists():
            return False
        blob.delete()
        return True

    def exists(self, path: str) -> bool:
        if self.mock:
            return (self._root / path).exists()
        return self._bucket.blob(path).exists()

    def list_files(self, prefix: str = "") -> list[str]:
        """List object paths under a prefix, relative to the bucket root."""
        if self.mock:
            base = self._root / prefix
            if not base.exists():
                return []
            return sorted(
                str(p.relative_to(self._root))
                for p in base.rglob("*")
                if p.is_file() and p.suffix != ".tmp"
            )
        return sorted(blob.name for blob in self._bucket.list_blobs(prefix=prefix))

    # --- JSON convenience wrappers ------------------------------------- #

    def upload_json(self, path: str, payload: dict) -> str:
        return self.upload_bytes(
            path, json.dumps(payload).encode("utf-8"), content_type="application/json"
        )

    def download_json(self, path: str) -> dict:
        return json.loads(self.download_bytes(path).decode("utf-8"))
