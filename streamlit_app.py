"""Streamlit UI for Agentic RAG System - Simplified Version"""

import sys
import time
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor, safe_upload_filename
from src.graph_builder.graph_builder import GraphBuilder
from src.vectorstore.vectorstore import VectorStore


def _urls_for_ingestion() -> list[str]:
    """
    If data/urls.txt exists: use its non-empty lines (may be [] for local-only).
    If the file is missing: use Config.DEFAULT_URLS (same as CLI).
    """
    urls_file = Config.DATA_DIR / "urls.txt"
    if urls_file.is_file():
        return [
            ln.strip()
            for ln in urls_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    return list(Config.DEFAULT_URLS)


# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered",
)

# Simple CSS
st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "ingestion_version" not in st.session_state:
        st.session_state.ingestion_version = 0


@st.cache_resource
def initialize_rag(ingestion_version: int):
    """Initialize the RAG system. Cache key bumps when uploads change the corpus."""
    llm = Config.get_llm()
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    vector_store = VectorStore()

    urls = _urls_for_ingestion()
    documents = doc_processor.process_sources(urls, Config.DATA_DIR)
    if not documents:
        raise ValueError(
            "No documents loaded. Add URLs (or data/urls.txt), PDF/TXT/MD files under "
            f"{Config.DATA_DIR}, or upload files in the sidebar."
        )
    vector_store.create_vectorstore(documents)

    graph_builder = GraphBuilder(
        retriever=vector_store.get_retriever(),
        llm=llm,
    )
    graph_builder.build()

    return graph_builder, len(documents)


def main():
    init_session_state()

    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")

    with st.sidebar:
        st.markdown("### 📁 Documents")
        uploaded = st.file_uploader(
            "Upload PDF, TXT, or Markdown",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help=f"Files are saved under {Config.UPLOADS_DIR} and included on the next index build.",
        )
        if st.button("Save uploads & rebuild index", type="primary"):
            if not uploaded:
                st.warning("Choose one or more files first.")
            else:
                Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
                Config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                for f in uploaded:
                    dest = Config.UPLOADS_DIR / safe_upload_filename(f.name)
                    dest.write_bytes(f.getvalue())
                st.session_state.ingestion_version += 1
                st.session_state.initialized = False
                st.session_state.rag_system = None
                st.session_state.history = []
                st.success(f"Saved {len(uploaded)} file(s). Rebuilding index…")
                st.rerun()

    if not st.session_state.initialized:
        with st.spinner("Loading system…"):
            try:
                rag_system, num_chunks = initialize_rag(st.session_state.ingestion_version)
            except Exception as e:
                st.error("Failed to initialize the RAG pipeline.")
                st.exception(e)
            else:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?",
        )
        submit = st.form_submit_button("🔍 Search")

    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching…"):
                start_time = time.time()
                result = st.session_state.rag_system.run(question)
                elapsed_time = time.time() - start_time

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "time": elapsed_time,
                    }
                )

                st.markdown("### 💡 Answer")
                st.success(result["answer"])

                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result["retrieved_docs"], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True,
                        )

                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")


if __name__ == "__main__":
    main()
