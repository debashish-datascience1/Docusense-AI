"""Vector store abstraction: local FAISS index + Vertex AI Matching Engine adapter.

Switch backends with VECTOR_BACKEND=faiss|matching_engine. Both implement the
same upsert/search interface so the RAG pipeline doesn't care which one runs.

FAISS uses inner product over L2-normalized vectors, i.e. cosine similarity,
and persists the index + chunk metadata to disk between restarts.
"""

import abc
import json
import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    score: float  # cosine similarity in [-1, 1]
    text: str
    metadata: dict = field(default_factory=dict)


class VectorStore(abc.ABC):
    @abc.abstractmethod
    def upsert(
        self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]
    ) -> None:
        """Insert or replace chunks. Each metadata dict must include 'text'."""

    @abc.abstractmethod
    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return the top_k most similar chunks."""

    @abc.abstractmethod
    def count(self) -> int:
        """Number of chunks currently indexed."""


# ---------------------------------------------------------------------- #
# FAISS (local, default)                                                  #
# ---------------------------------------------------------------------- #


class FaissVectorStore(VectorStore):
    """Flat inner-product FAISS index persisted under faiss_index_dir."""

    def __init__(self, index_dir: str | None = None, dim: int = 768):
        settings = get_settings()
        self.dim = dim
        self.index_dir = Path(index_dir or settings.faiss_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.index_dir / "index.faiss"
        self._meta_path = self.index_dir / "metadata.json"
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._metadata: dict[str, dict] = {}
        self._index = self._load()

    def _load(self):
        import faiss

        if self._index_path.exists() and self._meta_path.exists():
            index = faiss.read_index(str(self._index_path))
            saved = json.loads(self._meta_path.read_text())
            self._ids = saved["ids"]
            self._metadata = saved["metadata"]
            logger.info("Loaded FAISS index with %d vectors", index.ntotal)
            return index
        return faiss.IndexFlatIP(self.dim)

    def _persist(self) -> None:
        import faiss

        faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(
            json.dumps({"ids": self._ids, "metadata": self._metadata})
        )

    def upsert(
        self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]
    ) -> None:
        import faiss

        if not ids:
            return
        matrix = np.array(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        with self._lock:
            # IndexFlatIP has no native delete; replacing an existing id would
            # require a rebuild. Chunk ids are content-unique uuids, so plain
            # append is correct here.
            self._index.add(matrix)
            self._ids.extend(ids)
            for chunk_id, meta in zip(ids, metadatas):
                self._metadata[chunk_id] = meta
            self._persist()

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        import faiss

        with self._lock:
            if self._index.ntotal == 0:
                return []
            query = np.array([vector], dtype="float32")
            faiss.normalize_L2(query)
            scores, indices = self._index.search(query, min(top_k, self._index.ntotal))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                chunk_id = self._ids[idx]
                meta = self._metadata.get(chunk_id, {})
                results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        score=float(score),
                        text=meta.get("text", ""),
                        metadata={k: v for k, v in meta.items() if k != "text"},
                    )
                )
            return results

    def count(self) -> int:
        return self._index.ntotal


# ---------------------------------------------------------------------- #
# Vertex AI Matching Engine (production)                                  #
# ---------------------------------------------------------------------- #


class MatchingEngineVectorStore(VectorStore):
    """Adapter for Vertex AI Matching Engine (Vector Search).

    Matching Engine stores only vectors, so chunk text/metadata is persisted
    to GCS (chunks/{chunk_id}.json) and re-hydrated at query time. Requires a
    deployed streaming-update index; see scripts/setup_gcp.sh and the README.
    """

    def __init__(self):
        from google.cloud import aiplatform

        settings = get_settings()
        if not (
            settings.matching_engine_index_id
            and settings.matching_engine_endpoint_id
            and settings.matching_engine_deployed_index_id
        ):
            raise ValueError(
                "VECTOR_BACKEND=matching_engine requires MATCHING_ENGINE_INDEX_ID, "
                "MATCHING_ENGINE_ENDPOINT_ID and MATCHING_ENGINE_DEPLOYED_INDEX_ID"
            )
        aiplatform.init(
            project=settings.gcp_project_id, location=settings.gcp_location
        )
        self._index = aiplatform.MatchingEngineIndex(settings.matching_engine_index_id)
        self._endpoint = aiplatform.MatchingEngineIndexEndpoint(
            settings.matching_engine_endpoint_id
        )
        self._deployed_index_id = settings.matching_engine_deployed_index_id
        # GCS holds the chunk payloads Matching Engine can't store
        from app.gcs_handler import GCSHandler

        self._gcs = GCSHandler()

    def upsert(
        self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]
    ) -> None:
        if not ids:
            return
        datapoints = [
            {"datapoint_id": chunk_id, "feature_vector": vector}
            for chunk_id, vector in zip(ids, vectors)
        ]
        self._index.upsert_datapoints(datapoints=datapoints)
        for chunk_id, meta in zip(ids, metadatas):
            self._gcs.upload_json(f"chunks/{chunk_id}.json", meta)

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        neighbors = self._endpoint.find_neighbors(
            deployed_index_id=self._deployed_index_id,
            queries=[vector],
            num_neighbors=top_k,
        )
        results = []
        for neighbor in neighbors[0] if neighbors else []:
            try:
                meta = self._gcs.download_json(f"chunks/{neighbor.id}.json")
            except Exception:
                logger.warning("No metadata found for chunk %s", neighbor.id)
                meta = {}
            results.append(
                SearchResult(
                    chunk_id=neighbor.id,
                    # find_neighbors returns distance; convert to similarity
                    score=1.0 - float(neighbor.distance),
                    text=meta.get("text", ""),
                    metadata={k: v for k, v in meta.items() if k != "text"},
                )
            )
        return results

    def count(self) -> int:
        # Matching Engine exposes no cheap count; -1 signals "unknown"
        return -1


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_backend == "matching_engine" and not settings.vertex_ai_mock:
        return MatchingEngineVectorStore()
    return FaissVectorStore()
