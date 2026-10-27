from fastapi import APIRouter, Depends, File, UploadFile

from ..routes.dependencies import get_document_service
from ..schemas.documents import (
    DocumentListResponse,
    UploadResponse,
)
from ..services.document_service import DocumentService

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> dict:
    return await service.upload(file)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> dict:
    return {"documents": await service.list_documents()}
