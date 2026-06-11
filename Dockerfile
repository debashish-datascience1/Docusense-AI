# DocuSense AI — FastAPI backend image for Cloud Run
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# curl is needed for the container health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/docusense

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY ui/ ui/

# Run as a non-root user; /tmp/docusense holds the FAISS index + mock storage
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /tmp/docusense \
    && chown -R appuser:appuser /tmp/docusense /srv/docusense
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8080}/health" || exit 1

# Cloud Run injects $PORT; default to 8080 for local docker run
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
