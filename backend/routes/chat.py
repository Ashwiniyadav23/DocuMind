from fastapi import APIRouter, Depends

from ..routes.dependencies import get_chat_service
from ..schemas.chat import ChatRequest
from ..services.chat_service import ChatService

router = APIRouter()


@router.post("/chat")
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    return await service.answer(request.question)
