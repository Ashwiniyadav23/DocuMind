
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import documents, chat

app = FastAPI(title="FastAPI RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "RAG Chatbot API is running!"
    }


app.include_router(documents.router)
app.include_router(chat.router)
