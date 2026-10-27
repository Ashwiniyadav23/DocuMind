from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from ..core.exceptions import DocumentProcessingError, UnsupportedFileTypeError
from ..core.config import get_settings


def extract_text(filename: str, file_bytes: bytes) -> str:
    extension = Path(filename).suffix.lower()

    if extension == ".txt":
        return file_bytes.decode("utf-8-sig", errors="replace")

    if extension == ".pdf":
        pdf = PdfReader(BytesIO(file_bytes))
        if pdf.is_encrypted:
            raise DocumentProcessingError(
                "Password-protected PDFs are not supported."
            )
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

    raise UnsupportedFileTypeError("Only TXT and PDF files are supported.")


def split_into_chunks(text: str) -> list[str]:
    settings = get_settings()
    normalized_text = " ".join(text.split())
    chunks: list[str] = []
    start = 0

    while start < len(normalized_text):
        end = start + settings.chunk_size
        chunk = normalized_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized_text):
            break
        start = end - settings.chunk_overlap

    return chunks
