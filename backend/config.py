"""Backward-compatible configuration exports."""

from .core.config import get_embedding_model, get_settings

_settings = get_settings()
CHAT_MODEL = _settings.chat_model
EMBEDDING_MODEL = _settings.embedding_model
MAX_FILE_SIZE = _settings.max_file_size
CHUNK_SIZE = _settings.chunk_size
CHUNK_OVERLAP = _settings.chunk_overlap
TOP_K = _settings.top_k

__all__ = [
    "CHAT_MODEL",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "EMBEDDING_MODEL",
    "MAX_FILE_SIZE",
    "TOP_K",
    "get_embedding_model",
    "get_settings",
]
