from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.middleware import RequestContextMiddleware
from .routes import chat, documents

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def home() -> dict[str, str]:
    return {"message": "RAG Chatbot API is running!"}


app.include_router(documents.router)
app.include_router(chat.router)
