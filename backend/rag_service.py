"""Backward-compatible facade for the document repository."""

from __future__ import annotations

from .repositories.document_repository import DocumentRepository
from .routes.dependencies import get_document_repository
from .core.config import get_settings


def store_document_chunks(filename: str, chunks: list[str]) -> int:
    return get_document_repository().store_chunks(filename, chunks)


def get_total_chunks() -> int:
    return get_document_repository().total_chunks()


def get_document_summary() -> list[dict]:
    return get_document_repository().document_summary()


def retrieve_relevant_chunks(question: str, top_k: int | None = None) -> list[dict]:
    return get_document_repository().retrieve(
        question,
        top_k if top_k is not None else get_settings().top_k,
    )


__all__ = [
    "DocumentRepository",
    "get_document_summary",
    "get_total_chunks",
    "retrieve_relevant_chunks",
    "store_document_chunks",
]
