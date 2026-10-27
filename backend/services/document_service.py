from __future__ import annotations

import os
from asyncio import to_thread

from fastapi import UploadFile

from ..core.config import get_settings
from ..core.exceptions import DocumentProcessingError, EmptyDocumentError
from ..repositories.document_repository import DocumentRepository
from ..utils.document_utils import extract_text, split_into_chunks
from . import cache_service


class DocumentService:
    def __init__(self, repository: DocumentRepository) -> None:
        self.repository = repository

    async def upload(self, file: UploadFile) -> dict[str, int | str]:
        filename = os.path.basename(file.filename or "")
        if not filename:
            raise DocumentProcessingError("Please provide a filename.")
        if not filename.lower().endswith((".txt", ".pdf")):
            raise DocumentProcessingError("Upload a .txt or .pdf file.")

        file_bytes = await file.read()
        settings = get_settings()
        if not file_bytes:
            raise DocumentProcessingError("The uploaded file is empty.")
        if len(file_bytes) > settings.max_file_size:
            raise DocumentProcessingError("File size must not exceed 10 MB.")

        text = await to_thread(extract_text, filename, file_bytes)
        if not text.strip():
            raise EmptyDocumentError(
                "No readable text was found. A scanned PDF may require OCR."
            )
        chunks = await to_thread(split_into_chunks, text)
        if not chunks:
            raise EmptyDocumentError("The document could not be split into chunks.")

        chunks_created = await to_thread(
            self.repository.store_chunks, filename, chunks
        )
        await to_thread(cache_service.invalidate_answers)
        total_chunks = await to_thread(self.repository.total_chunks)
        return {
            "message": "Document processed successfully.",
            "filename": filename,
            "chunks_created": chunks_created,
            "total_chunks_in_database": total_chunks,
        }

    async def list_documents(self) -> list[dict[str, int | str]]:
        return await to_thread(self.repository.document_summary)
