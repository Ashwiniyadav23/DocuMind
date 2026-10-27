
import os
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq
from fastembed import TextEmbedding

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to your .env file."
    )

client = Groq(api_key=api_key)

CHAT_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Load the ONNX embedding model only when an embedding is first needed."""
    return TextEmbedding(
        model_name=EMBEDDING_MODEL,
        threads=1,
        lazy_load=True,
    )

MAX_FILE_SIZE = 10 * 1024 * 1024
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
