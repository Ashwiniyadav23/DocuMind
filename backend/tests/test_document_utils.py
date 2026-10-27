import pytest

from backend.core.config import get_settings
from backend.core.exceptions import UnsupportedFileTypeError
from backend.utils.document_utils import extract_text, split_into_chunks


def test_extract_text_decodes_utf8_text() -> None:
    assert extract_text("notes.txt", b"\xef\xbb\xbfHello") == "Hello"


def test_extract_text_rejects_unsupported_extensions() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("notes.docx", b"content")


def test_split_into_chunks_honors_configured_overlap() -> None:
    settings = get_settings()
    text = "a" * (settings.chunk_size + 10)

    chunks = split_into_chunks(text)

    assert len(chunks) == 2
    assert chunks[0] == "a" * settings.chunk_size
    assert chunks[1] == "a" * (settings.chunk_overlap + 10)
