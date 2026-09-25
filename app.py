
"""
app.py

Streamlit UI for the generalized multimodal RAG engine.

Upload any PDF, index it (text + tables + vision-summarized images), then
ask questions about it and get grounded, cited answers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine

load_dotenv()

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_default(key: str) -> str:
    """Read from Streamlit Secrets first, then environment variables."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.getenv(key, "")


GROQ_API_KEY = get_default("GROQ_API_KEY")
PINECONE_API_KEY = get_default("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_default("PINECONE_INDEX_NAME") or "multimodal-rag"

keys_present = bool(GROQ_API_KEY and PINECONE_API_KEY)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models and connecting to Pinecone...")
def get_engine(
    groq_key: str,
    pinecone_key: str,
    index_name: str,
) -> RAGEngine:
    return RAGEngine(
        groq_api_key=groq_key,
        pinecone_api_key=pinecone_key,
        index_name=index_name,
        image_dir=Path(tempfile.gettempdir()) / "rag_extracted_images",
    )


# ---------------------------------------------------------------------------
# Sidebar - document upload and session controls
# ---------------------------------------------------------------------------

st.sidebar.header("Document")

if not keys_present:
    st.sidebar.error(
        "API keys are missing. Configure GROQ_API_KEY and "
        "PINECONE_API_KEY in your .env file or Streamlit Secrets."
    )

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    disabled=not keys_present,
)

process_clicked = st.sidebar.button(
    "Process document",
    disabled=not (keys_present and uploaded_file is not None),
)

st.sidebar.divider()

if st.sidebar.button(
    "Clear session",
    disabled="namespace" not in st.session_state,
):
    if st.session_state.get("namespace"):
        get_engine(
            GROQ_API_KEY,
            PINECONE_API_KEY,
            PINECONE_INDEX_NAME,
        ).delete_namespace(st.session_state["namespace"])

    for key in ("retriever", "namespace", "source_name", "chat_history"):
        st.session_state.pop(key, None)

    st.rerun()

st.sidebar.caption(
    "Clears the chat and deletes this document's vectors from Pinecone."
)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("📄 Document Q&A")

st.caption(
    "Upload any PDF — reports, tenders, manuals, specs — and ask questions "
    "about its text, tables, and diagrams."
)

if not keys_present:
    st.warning(
        "API keys are missing. Please configure them in your .env file "
        "or Streamlit Secrets."
    )
    st.stop()

engine = get_engine(
    GROQ_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Process a newly uploaded document
# ---------------------------------------------------------------------------

if process_clicked and uploaded_file is not None:
    tmp_path = Path(tempfile.gettempdir()) / uploaded_file.name
    tmp_path.write_bytes(uploaded_file.getvalue())

    progress_bar = st.sidebar.progress(0.0, text="Starting...")

    def progress_callback(page: int, total: int) -> None:
        progress_bar.progress(
            page / total,
            text=f"Processing page {page}/{total}",
        )

    try:
        with st.spinner(
            "Reading and indexing the document — this can take a minute..."
        ):
            namespace, retriever, stats = engine.ingest(
                tmp_path,
                source_name=uploaded_file.name,
                progress_callback=progress_callback,
            )

        st.session_state.retriever = retriever
        st.session_state.namespace = namespace
        st.session_state.source_name = uploaded_file.name
        st.session_state.chat_history = []

        progress_bar.empty()

        if stats["skipped"]:
            st.sidebar.info(
                f"'{uploaded_file.name}' was already indexed — reusing it."
            )
        else:
            st.sidebar.success(
                f"Indexed {stats['num_chunks']} chunks from "
                f"'{uploaded_file.name}'."
            )
            st.sidebar.write(stats["modality_counts"])

    except Exception as e:
        progress_bar.empty()
        st.sidebar.error(f"Failed to process document: {e}")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

has_document = "retriever" in st.session_state

if has_document:
    st.caption(
        f"Currently indexed: **{st.session_state['source_name']}**"
    )
else:
    st.info(
        "Upload a PDF in the sidebar and click **Process document** "
        "to get started."
    )


# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write(
                        f"Page {source['page']} — {source['modality']}"
                    )

        if msg.get("images"):
            for col, img_path in zip(
                st.columns(len(msg["images"])),
                msg["images"],
            ):
                col.image(img_path)


question = st.chat_input(
    "Ask a question about the document...",
    disabled=not has_document,
)

if question:
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = engine.ask(
                    question,
                    st.session_state.retriever,
                )
            except Exception as e:
                result = {
                    "answer": f"Something went wrong answering that: {e}",
                    "sources": [],
                    "image_paths": [],
                }

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.write(
                        f"Page {source['page']} — {source['modality']}"
                    )

        if result["image_paths"]:
            for col, img_path in zip(
                st.columns(len(result["image_paths"])),
                result["image_paths"],
            ):
                col.image(img_path)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "images": result["image_paths"],
        }
    )