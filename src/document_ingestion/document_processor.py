"""Document processing module for loading and splitting documents"""

import re
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import (
    PyPDFLoader,
    PyPDFDirectoryLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Filenames treated as URL lists, not document bodies
_SKIP_TEXT_NAMES = frozenset({"urls.txt", "url.txt"})


def safe_upload_filename(name: str) -> str:
    """Sanitize an uploaded file name for storage on disk."""
    base = Path(name).name
    base = re.sub(r"[^a-zA-Z0-9._\- ]", "_", base).strip()
    return (base or "upload")[:200]


class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_local_files(self, data_dir: Path) -> List[Document]:
        """
        Load PDF, TXT, and Markdown files under data_dir (recursive).
        Skips urls.txt / url.txt so URL list files are not ingested as documents.
        """
        docs: List[Document] = []
        if not data_dir.is_dir():
            return docs

        for pdf_path in sorted(data_dir.rglob("*.pdf")):
            docs.extend(PyPDFLoader(str(pdf_path)).load())

        for pattern in ("*.txt", "*.md"):
            for path in sorted(data_dir.rglob(pattern)):
                if path.name.lower() in _SKIP_TEXT_NAMES:
                    continue
                docs.extend(TextLoader(str(path), encoding="utf-8").load())

        return docs

    def load_documents(self, urls: List[str], data_dir: Path) -> List[Document]:
        """Load remote URLs plus all supported files under data_dir."""
        docs: List[Document] = []
        for url in urls:
            if url.startswith("http://") or url.startswith("https://"):
                docs.extend(self.load_from_url(url))
        docs.extend(self.load_local_files(data_dir))
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)

    def process_sources(self, urls: List[str], data_dir: Path) -> List[Document]:
        """Load from URLs and data folder, then split into chunks."""
        docs = self.load_documents(urls, data_dir)
        return self.split_documents(docs)

    def process_urls(self, urls: List[str], data_dir: Path | None = None) -> List[Document]:
        """Backward-compatible alias: pass a data_dir or it defaults to ./data."""
        base = data_dir if data_dir is not None else Path("data")
        return self.process_sources(urls, base)
