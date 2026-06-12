# DocuSense AI — Roadmap

Ideas for future iterations, roughly ordered by value-for-effort.

## Near term

- [x] Document deletion (API + UI)
- [x] CI on every push (GitHub Actions, mock-mode tests)
- [ ] **Multi-turn chat** — rewrite follow-up questions using conversation
  history before embedding, so "what about his education?" retrieves well.
- [ ] **Duplicate-upload detection** — hash file bytes on ingest and skip
  (or version) documents that are already indexed.
- [ ] **Deploy the Streamlit UI to Cloud Run** as a second service, so the
  whole app runs in the cloud (currently only the API has a Dockerfile).

## Retrieval quality

- [ ] **Hybrid search** — combine FAISS vector scores with BM25 keyword
  scores; resumes and technical docs benefit a lot from exact-term matches.
- [ ] **Reranking** — retrieve top-20, rerank to top-5 with a cross-encoder
  or a cheap LLM call before building the prompt.
- [ ] **Evaluation harness** — a golden set of (question, expected source)
  pairs and a script that reports retrieval hit-rate and answer faithfulness,
  so retrieval changes can be measured instead of eyeballed.
- [ ] **Smarter chunking** — heading-aware splitting for structured docs;
  OCR fallback (Document AI) for scanned PDFs that PyPDF2 can't read.

## Production hardening

- [ ] **API authentication** — require Google ID tokens (or API keys) on all
  endpoints; currently the deployed API is open if `--allow-unauthenticated`.
- [ ] **Job state in Firestore** instead of GCS JSON files — atomic updates,
  queries, and no list-then-read pattern for `/documents`.
- [ ] **Matching Engine end-to-end guide** — scripted creation of the
  streaming-update index + endpoint (it's the one piece `setup_gcp.sh`
  doesn't provision because of its always-on cost).
- [ ] **Observability** — structured JSON logs for Cloud Logging, plus
  latency/token-count metrics per pipeline stage.
- [ ] **Per-user document namespaces** — isolate uploads by authenticated
  user so the index isn't shared by everyone.

## Format support

- [ ] DOCX and HTML ingestion
- [ ] Tables in PDFs (extract as markdown so Gemini can read them)
