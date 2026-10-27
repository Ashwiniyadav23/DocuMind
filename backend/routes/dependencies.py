from functools import lru_cache

from ..repositories.document_repository import DocumentRepository
from ..services.chat_service import ChatService
from ..services.document_service import DocumentService


@lru_cache(maxsize=1)
def get_document_repository() -> DocumentRepository:
    return DocumentRepository()


@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    return DocumentService(get_document_repository())


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService(get_document_repository())
