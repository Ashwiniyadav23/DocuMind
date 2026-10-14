
import os
from io import BytesIO

from fastapi import HTTPException
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_text(filename: str, file_bytes: bytes) -> str:
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".txt":
        return file_bytes.decode(
            "utf-8-sig",
            errors="replace"
        )

    if extension == ".pdf":
        pdf = PdfReader(BytesIO(file_bytes))

        if pdf.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Password-protected PDFs are not supported."
            )

        pages = []

        for page in pdf.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    raise HTTPException(
        status_code=400,
        detail="Only TXT and PDF files are supported."
    )


def split_into_chunks(text: str) -> list[str]:
    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks
