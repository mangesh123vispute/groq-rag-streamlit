"""Configuration module for Agentic RAG system"""

import os
from pathlib import Path

# Anaconda on Windows often sets SSL_CERT_FILE to a path that does not exist.
# httpx/then Groq and Hugging Face downloads fail with [Errno 2] No such file or directory.
_cert = os.environ.get("SSL_CERT_FILE")
if _cert and not os.path.isfile(_cert):
    os.environ.pop("SSL_CERT_FILE", None)

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # Project paths (config.py lives in src/config/)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data"
    UPLOADS_DIR = DATA_DIR / "uploads"

    # Groq free tier: create a key at https://console.groq.com/keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Model id — see https://console.groq.com/docs/models
    LLM_MODEL = "llama-3.3-70b-versatile"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the Groq chat model."""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to .env (free at https://console.groq.com/)."
            )
        return ChatGroq(
            model=cls.LLM_MODEL,
            api_key=cls.GROQ_API_KEY,
            temperature=0.2,
        )
