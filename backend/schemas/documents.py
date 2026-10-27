from pydantic import BaseModel


class DocumentSummary(BaseModel):
    filename: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    total_chunks_in_database: int
