# DocuMind backend

## Choose an AI provider

The RAG pipeline (PDF/TXT extraction, chunking, embeddings, Chroma retrieval, and the grounded answer prompt) is shared. `AI_PROVIDER` selects the text generation service. The default is `groq`, preserving the existing behavior. `AI_PROVIDER=ollama` uses the local Ollama HTTP API only; it does not fall back to Groq.

Copy `.env.example` to `.env` and set one provider:

```dotenv
# Existing hosted provider
AI_PROVIDER=groq
GROQ_API_KEY=your_key_here
CHAT_MODEL=openai/gpt-oss-20b

# Or local Ollama (no Groq key is needed)
# AI_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=qwen2.5:3b
REQUEST_TIMEOUT=120
```

Keep `.env` private and never commit API keys. `REQUEST_TIMEOUT` applies to either provider. Ollama token counts are returned when available; the cost estimate remains zero unless local token pricing is explicitly configured.

## Run with Ollama

1. Install Ollama for your operating system from [ollama.com/download](https://ollama.com/download).
2. Start the Ollama application/service. On systems where it is not started automatically, run `ollama serve` in a terminal.
3. Download and run a modest local model (3B is a practical starting point):

   ```sh
   ollama pull qwen2.5:3b
   ollama run qwen2.5:3b
   ```

   Exit the interactive prompt after confirming the model starts. Keep Ollama running; its local API listens at `http://localhost:11434` by default.
4. Set `AI_PROVIDER=ollama`, `OLLAMA_MODEL=qwen2.5:3b`, and `REQUEST_TIMEOUT=120` in `backend/.env`. Increase the timeout if your hardware needs longer for generation.
5. Start FastAPI from the repository root with `uvicorn backend.main:app --reload` and start the frontend as usual. Upload a document and ask a question. `/chat` returns the same answer, source, and retrieved chunk fields; it also reports the selected provider and usage.

The model download is a one-time network operation. Inference requests then go from FastAPI to Ollama on the same machine. With `AI_PROVIDER=ollama`, the backend does not construct a Groq client or send prompts to Groq.

## Troubleshooting

- Connection refused: start Ollama and confirm `OLLAMA_BASE_URL` is reachable from the FastAPI process.
- Model unavailable (HTTP 404): run `ollama pull <OLLAMA_MODEL>` and make sure the configured model name matches.
- Timeout: raise `REQUEST_TIMEOUT`, use a smaller model, or reduce retrieved context size.
- Invalid response: the API reports a service error instead of returning an unvalidated answer. Both providers must return JSON with `answer` and `citations`; Pydantic validates the result before it reaches the frontend.
