class DocumentProcessingError(Exception):
    """Raised when an uploaded document cannot be processed."""

    status_code = 400


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised for file types outside the supported TXT/PDF formats."""


class EmptyDocumentError(DocumentProcessingError):
    """Raised when a document contains no readable text."""


class NoRelevantChunksError(DocumentProcessingError):
    """Raised when retrieval cannot find any chunks to answer from."""

    status_code = 404


class AIServiceError(Exception):
    """Raised when the configured AI provider cannot answer a request."""

    status_code = 502
