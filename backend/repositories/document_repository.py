from typing import Any

import chromadb

from ..core.config import get_embedding_model, get_settings


class DocumentRepository:
    def __init__(self) -> None:
        settings = get_settings()
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"},
        )

    def store_chunks(self, filename: str, chunks: list[str]) -> int:
        if not chunks:
            return 0
        embeddings = list(get_embedding_model().embed(chunks, batch_size=8))
        self.collection.delete(where={"source": filename})
        self.collection.add(
            ids=[f"{filename}__chunk_{index}" for index in range(len(chunks))],
            documents=chunks,
            embeddings=[embedding.tolist() for embedding in embeddings],
            metadatas=[
                {"source": filename, "chunk_index": index}
                for index in range(len(chunks))
            ],
        )
        return len(chunks)

    def total_chunks(self) -> int:
        return self.collection.count()

    def document_summary(self) -> list[dict[str, Any]]:
        if self.total_chunks() == 0:
            return []
        result = self.collection.get(include=["metadatas"])
        summary: dict[str, int] = {}
        for metadata in result["metadatas"]:
            filename = str(metadata["source"])
            summary[filename] = summary.get(filename, 0) + 1
        return [
            {"filename": filename, "chunks": count}
            for filename, count in sorted(summary.items())
        ]

    def retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        total_chunks = self.total_chunks()
        if total_chunks == 0:
            return []
        question_embedding = next(get_embedding_model().query_embed(question))
        results = self.collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=min(top_k, total_chunks),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "source": results["metadatas"][0][index]["source"],
                "text": text,
                "score": 1.0 - float(results["distances"][0][index]),
            }
            for index, text in enumerate(results["documents"][0])
        ]
