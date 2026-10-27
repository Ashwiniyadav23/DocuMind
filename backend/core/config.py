from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]


class Settings:
    """Application settings loaded from environment variables."""

    app_name: str
    groq_api_key: str | None
    chat_model: str
    embedding_model: str
    chroma_path: Path
    max_file_size: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    cors_origins: list[str]
    redis_url: str | None
    cache_ttl: int
    request_timeout: float
    input_cost_per_million: float
    output_cost_per_million: float
    ai_provider: str
    ollama_base_url: str
    ollama_model: str

    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "FastAPI RAG Chatbot")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.chat_model = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.chroma_path = Path(
            os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))
        )
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))
        self.top_k = int(os.getenv("TOP_K", 4))
        self.redis_url = os.getenv("REDIS_URL")
        self.cache_ttl = int(os.getenv("CACHE_TTL", 300))
        self.request_timeout = float(os.getenv("REQUEST_TIMEOUT", 30))
        self.input_cost_per_million = float(os.getenv("INPUT_COST_PER_MILLION", 0))
        self.output_cost_per_million = float(os.getenv("OUTPUT_COST_PER_MILLION", 0))
        self.ai_provider = os.getenv("AI_PROVIDER", "groq").strip().lower()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        configured_origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        self.cors_origins = [
            origin.strip()
            for origin in configured_origins.split(",")
            if origin.strip()
        ]

    def validate(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.max_file_size <= 0 or self.top_k <= 0 or self.cache_ttl <= 0:
            raise ValueError("MAX_FILE_SIZE, TOP_K, and CACHE_TTL must be positive.")
        if self.ai_provider not in {"groq", "ollama"}:
            raise ValueError("AI_PROVIDER must be either 'groq' or 'ollama'.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    from fastembed import TextEmbedding

    return TextEmbedding(
        model_name=get_settings().embedding_model,
        threads=1,
        lazy_load=True,
    )
