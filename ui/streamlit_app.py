"""Streamlit chat UI for DocuSense AI.

Sidebar: file upload + ingestion status + document list.
Main: chat with streaming answers and source citations.

Run with:  streamlit run ui/streamlit_app.py
Configure the backend with BACKEND_URL (default http://localhost:8080).
"""

import json
import os
import time

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="DocuSense AI", page_icon="📄", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, sources?, confidence?}]
if "jobs" not in st.session_state:
    st.session_state.jobs = {}  # job_id -> filename


def backend_ok() -> bool:
    try:
        return httpx.get(f"{BACKEND_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def fetch_documents() -> list[dict]:
    try:
        response = httpx.get(f"{BACKEND_URL}/documents", timeout=10)
        response.raise_for_status()
        return response.json()["documents"]
    except httpx.HTTPError:
        return []


# --------------------------------------------------------------------- #
# Sidebar: upload + status                                               #
# --------------------------------------------------------------------- #

with st.sidebar:
    st.title("📄 DocuSense AI")
    st.caption(f"Backend: {BACKEND_URL}")

    if not backend_ok():
        st.error("Backend unreachable. Start it with:\n`uvicorn app.main:app --port 8080`")
        st.stop()

    st.subheader("Upload documents")
    uploaded = st.file_uploader(
        "PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if uploaded and st.button("Ingest", type="primary", use_container_width=True):
        for file in uploaded:
            response = httpx.post(
                f"{BACKEND_URL}/ingest",
                files={"file": (file.name, file.getvalue())},
                timeout=60,
            )
            if response.status_code == 200:
                job = response.json()
                st.session_state.jobs[job["job_id"]] = file.name
                st.toast(f"Queued {file.name}")
            else:
                st.error(f"{file.name}: {response.json().get('detail', response.text)}")

        # Give async ingestion a moment, then poll each queued job briefly
        with st.spinner("Ingesting..."):
            deadline = time.time() + 30
            pending = set(st.session_state.jobs)
            while pending and time.time() < deadline:
                for job_id in list(pending):
                    status = httpx.get(f"{BACKEND_URL}/job/{job_id}", timeout=10)
                    if status.status_code == 200 and status.json()["status"] in (
                        "done",
                        "failed",
                    ):
                        pending.discard(job_id)
                if pending:
                    time.sleep(0.5)

    st.subheader("Documents")
    docs = fetch_documents()
    if not docs:
        st.caption("Nothing ingested yet.")
    for doc in docs:
        icon = {"done": "✅", "failed": "❌", "processing": "⏳", "queued": "🕐"}.get(
            doc.get("status", ""), "❓"
        )
        label = f"{icon} {doc.get('filename', '?')}"
        if doc.get("status") == "done":
            label += f" ({doc.get('chunks', '?')} chunks)"
        name_col, delete_col = st.columns([5, 1])
        name_col.write(label)
        if delete_col.button("🗑", key=f"delete-{doc['job_id']}", help="Delete document"):
            response = httpx.delete(
                f"{BACKEND_URL}/documents/{doc['job_id']}", timeout=30
            )
            if response.status_code == 200:
                st.toast(f"Deleted {doc.get('filename', '?')}")
            st.rerun()
        if doc.get("status") == "failed":
            st.caption(f"Error: {doc.get('error', 'unknown')}")

# --------------------------------------------------------------------- #
# Main: chat                                                              #
# --------------------------------------------------------------------- #

st.header("Ask your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(
                f"Sources ({len(message['sources'])}) — confidence {message['confidence']:.0%}"
            ):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['filename']}** (score {source['score']:.3f})\n\n"
                        f"> {source['snippet']}"
                    )

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources_box: dict = {"sources": [], "confidence": 0.0}

        def token_stream():
            with httpx.stream(
                "POST",
                f"{BACKEND_URL}/ask",
                json={"question": question, "stream": True},
                timeout=120,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event["type"] == "sources":
                        sources_box["sources"] = event["sources"]
                        sources_box["confidence"] = event["confidence"]
                    elif event["type"] == "token":
                        yield event["text"]

        answer = st.write_stream(token_stream())
        if sources_box["sources"]:
            with st.expander(
                f"Sources ({len(sources_box['sources'])}) — "
                f"confidence {sources_box['confidence']:.0%}"
            ):
                for source in sources_box["sources"]:
                    st.markdown(
                        f"**{source['filename']}** (score {source['score']:.3f})\n\n"
                        f"> {source['snippet']}"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources_box["sources"],
            "confidence": sources_box["confidence"],
        }
    )
