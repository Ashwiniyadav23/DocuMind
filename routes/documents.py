
import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import MAX_FILE_SIZE
from document_utils import extract_text, split_into_chunks
from rag_service import (
    store_document_chunks,
    get_document_summary,
    get_total_chunks,
)

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "")

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Please provide a filename."
        )

    if not filename.lower().endswith((".txt", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Upload a .txt or .pdf file."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 10 MB."
        )

    text = extract_text(filename, file_bytes)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found. "
                "A scanned PDF may require OCR."
            )
        )

    chunks = split_into_chunks(text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="The document could not be split into chunks."
        )

    chunks_created = store_document_chunks(filename, chunks)

    return {
        "message": "Document processed successfully.",
        "filename": filename,
        "chunks_created": chunks_created,
        "total_chunks_in_database": get_total_chunks()
    }


@router.get("/documents")
def list_documents():
    return {"documents": get_document_summary()}
