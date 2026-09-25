"""
rag_engine.py

Generalized multimodal RAG engine. Works with ANY uploaded PDF (reports,
tenders, manuals, specs, contracts...) - nothing here is hardcoded to a
single document or domain.

Pipeline:
    PDF -> extract text / tables / images per page
        -> images are summarized by a vision model (charts, diagrams, photos)
        -> everything becomes a LangChain Document with a "modality" tag
        -> embedded and stored in Pinecone, isolated per document via a
           content-hash namespace
        -> at query time: retrieve top-k chunks -> if any are visual,
           answer with the vision model + original images; otherwise
           answer with a plain text LLM chain

Design note: RAGEngine holds only shared, stateless resources (model
clients, embeddings, the Pinecone index). Per-document state (namespace,
retriever) is returned to the caller instead of stored on `self`, so one
engine instance can be safely reused across multiple documents / users in
a web app (see app.py) without their sessions leaking into each other.
"""

from __future__ import annotations

import io
import os
import base64
import hashlib
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


# ---------------------------------------------------------------------------
# Defaults (all overridable via RAGEngine's constructor)
# ---------------------------------------------------------------------------

DEFAULT_TEXT_MODEL = "openai/gpt-oss-20b"
DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_NAME = "multimodal-rag"
DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# Prompts - generalized, no document-specific wording baked in
# ---------------------------------------------------------------------------

VISION_SUMMARY_PROMPT = """
This visual was extracted from page {page_number} of an uploaded document.

Describe the useful information visible in the visual.

If it is a chart or graph:
- mention important values
- mention highest/lowest values
- mention the main trend

If it is a diagram, drawing, or schematic:
- identify important components and any labels or callouts
- explain the flow or relationships between components

If it is a photo or generic figure:
- describe the useful factual information

Keep the description concise and factual. Do not guess at exact numbers you
cannot clearly read.
"""

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant answering questions about a document the user
uploaded.

Use ONLY the retrieved context below.

If the answer is not available in the context, say:
"I could not find that information in the document."

Mention page numbers when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

VISION_ANSWER_PROMPT = """
You are answering a question about a document the user uploaded.

Use ONLY the retrieved context and the attached retrieved visuals.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Answer factually.
- Use the attached visuals when relevant.
- Mention page numbers.
- If the information is missing, say you could not find it.
"""


def make_namespace(file_bytes: bytes) -> str:
    """Stable, content-derived Pinecone namespace.

    Using a hash of the file content (not the filename) means the same
    document uploaded twice always lands in the same namespace (so it can
    be re-used without re-processing), while two different documents never
    collide even if they share a filename.
    """
    return "doc-" + hashlib.sha256(file_bytes).hexdigest()[:16]


def image_to_data_uri(image_path: Path | str) -> str:
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


class RAGEngine:
    """Shared, cacheable core. Create one instance per (api keys, index)
    combination and reuse it for every document and every user session.
    """

    def __init__(
        self,
        groq_api_key: str,
        pinecone_api_key: str,
        index_name: str = DEFAULT_INDEX_NAME,
        text_model: str = DEFAULT_TEXT_MODEL,
        vision_model: str = DEFAULT_VISION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        image_dir: str | Path = "extracted_images",
        top_k: int = DEFAULT_TOP_K,
    ):
        if not groq_api_key or not pinecone_api_key:
            raise ValueError("Both a Groq and a Pinecone API key are required.")

        # ChatGroq reads GROQ_API_KEY from the environment in some versions,
        # so set it explicitly as well as passing it directly.
        os.environ["GROQ_API_KEY"] = groq_api_key

        self.groq_client = Groq(api_key=groq_api_key)
        self.text_llm = ChatGroq(model=text_model, temperature=0, api_key=groq_api_key)
        self.vision_model = vision_model
        self.index_name = index_name
        self.top_k = top_k
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(exist_ok=True, parents=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.embedding_dim = len(self.embeddings.embed_query("dimension check"))

        self.pc = Pinecone(api_key=pinecone_api_key)
        self._ensure_index()

        self.rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.text_rag_chain = self.rag_prompt | self.text_llm | StrOutputParser()

    # -- index lifecycle ----------------------------------------------------

    def _ensure_index(self) -> None:
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(2)
        self.index = self.pc.Index(self.index_name)

    def namespace_exists(self, namespace: str) -> bool:
        """True if this document has already been ingested (any prior
        session), so the app can skip re-processing it."""
        stats = self.index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}) or {}
        return namespace in ns_stats and ns_stats[namespace].get("vector_count", 0) > 0

    # -- ingestion ------------------------------------------------------

    def summarize_visual(self, image_path: Path, page_number: int) -> str:
        image_data = image_to_data_uri(image_path)
        prompt = VISION_SUMMARY_PROMPT.format(page_number=page_number)

        response = self.groq_client.chat.completions.create(
            model=self.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            }],
            temperature=0,
            max_completion_tokens=500,
        )
        return response.choices[0].message.content.strip()

    def extract_multimodal_documents(
        self,
        pdf_path: Path,
        source_name: str,
        namespace: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[Document]:
        doc_image_dir = self.image_dir / namespace
        doc_image_dir.mkdir(exist_ok=True, parents=True)

        pdf = fitz.open(str(pdf_path))
        documents: list[Document] = []
        extracted_xrefs: set[int] = set()

        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_number = page_index + 1

            if progress_callback:
                progress_callback(page_number, len(pdf))

            # 1. TEXT
            text = page.get_text("text").strip()
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"page": page_number, "modality": "text", "source": source_name},
                ))

            # 2. TABLES
            try:
                tables = page.find_tables().tables
                for table_number, table in enumerate(tables, start=1):
                    df = table.to_pandas()
                    if not df.empty:
                        documents.append(Document(
                            page_content=df.to_markdown(index=False),
                            metadata={
                                "page": page_number,
                                "modality": "table",
                                "table_number": table_number,
                                "source": source_name,
                            },
                        ))
            except Exception:
                pass  # a malformed table on one page shouldn't stop ingestion

            # 3. IMAGES / CHARTS / DIAGRAMS
            for image_number, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                if xref in extracted_xrefs:
                    continue
                extracted_xrefs.add(xref)

                image_data = pdf.extract_image(xref)
                image_bytes = image_data["image"]
                image_extension = image_data["ext"]
                image_path = doc_image_dir / f"p{page_number}_{image_number}.{image_extension}"
                image_path.write_bytes(image_bytes)

                try:
                    summary = self.summarize_visual(image_path, page_number)
                    documents.append(Document(
                        page_content=summary,
                        metadata={
                            "page": page_number,
                            "modality": "visual",
                            "image_path": str(image_path),
                            "source": source_name,
                        },
                    ))
                except Exception:
                    pass  # one bad vision call shouldn't stop the whole ingest

        pdf.close()
        return documents

    def get_retriever(self, namespace: str):
        vectorstore = PineconeVectorStore(
            index_name=self.index_name, embedding=self.embeddings, namespace=namespace,
        )
        return vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def ingest(
        self,
        pdf_path: Path | str,
        source_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        force_reprocess: bool = False,
    ) -> tuple[str, Any, dict]:
        """Process (or re-attach to) a document.

        Returns (namespace, retriever, stats). The caller is responsible
        for storing `retriever` (e.g. in Streamlit's session_state) since
        this engine instance is shared and does not hold per-document state.
        """
        pdf_path = Path(pdf_path)
        source_name = source_name or pdf_path.name
        namespace = make_namespace(pdf_path.read_bytes())

        if not force_reprocess and self.namespace_exists(namespace):
            retriever = self.get_retriever(namespace)
            return namespace, retriever, {"skipped": True, "num_chunks": None, "modality_counts": None}

        documents = self.extract_multimodal_documents(pdf_path, source_name, namespace, progress_callback)

        vectorstore = PineconeVectorStore(
            index_name=self.index_name, embedding=self.embeddings, namespace=namespace,
        )
        vectorstore.add_documents(documents)
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        modality_counts = (
            pd.Series([d.metadata["modality"] for d in documents]).value_counts().to_dict()
            if documents else {}
        )
        stats = {"skipped": False, "num_chunks": len(documents), "modality_counts": modality_counts}
        return namespace, retriever, stats

    # -- querying -------------------------------------------------------

    @staticmethod
    def format_context(retrieved_docs) -> str:
        parts = []
        for doc in retrieved_docs:
            page = doc.metadata.get("page")
            modality = doc.metadata.get("modality")
            parts.append(f"[Page {page} | {modality.upper()}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def answer_with_vision(self, question: str, context: str, image_paths: list[str]) -> str:
        image_paths = image_paths[:3]  # cap attached images per call

        content = [{"type": "text", "text": VISION_ANSWER_PROMPT.format(context=context, question=question)}]
        for image_path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}})

        response = self.groq_client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_completion_tokens=900,
        )
        return response.choices[0].message.content.strip()

    def ask(self, question: str, retriever) -> dict:
        """Stateless query - takes the retriever explicitly rather than
        reading it off `self`, so concurrent users never share state."""
        retrieved_docs = retriever.invoke(question)
        context = self.format_context(retrieved_docs)

        image_paths = [
            doc.metadata["image_path"]
            for doc in retrieved_docs
            if doc.metadata.get("modality") == "visual"
            and Path(doc.metadata.get("image_path", "")).exists()
        ]

        if image_paths:
            answer = self.answer_with_vision(question, context, image_paths)
        else:
            answer = self.text_rag_chain.invoke({"context": context, "question": question})

        sources = [
            {"page": d.metadata.get("page"), "modality": d.metadata.get("modality")}
            for d in retrieved_docs
        ]
        return {"answer": answer, "sources": sources, "image_paths": image_paths}

    # -- cleanup ----------------------------------------------------------

    def delete_namespace(self, namespace: str) -> None:
        """Remove a document's vectors from Pinecone and its extracted
        images from disk. Call this when a session ends, if you don't want
        to keep paying to store it."""
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except Exception:
            pass
        shutil.rmtree(self.image_dir / namespace, ignore_errors=True)
