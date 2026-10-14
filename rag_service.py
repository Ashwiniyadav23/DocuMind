
from pathlib import Path

import chromadb

from config import embedding_model, TOP_K

# Store the database inside the backend folder.
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_or_create_collection(
    name="rag_documents",
    metadata={"hnsw:space": "cosine"}
)


def store_document_chunks(
    filename: str,
    chunks: list[str]
) -> int:
    if not chunks:
        return 0

    # Generate embeddings for all chunks.
    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )

    # Replace chunks from an earlier upload of this filename.
    collection.delete(where={"source": filename})

    ids = [
        f"{filename}__chunk_{index}"
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": filename,
                "chunk_index": index
            }
            for index in range(len(chunks))
        ]
    )

    return len(chunks)


def get_total_chunks() -> int:
    return collection.count()


def get_document_summary() -> list[dict]:
    if collection.count() == 0:
        return []

    result = collection.get(include=["metadatas"])
    summary = {}

    for metadata in result["metadatas"]:
        filename = metadata["source"]
        summary[filename] = summary.get(filename, 0) + 1

    return [
        {"filename": filename, "chunks": count}
        for filename, count in sorted(summary.items())
    ]


def retrieve_relevant_chunks(
    question: str,
    top_k: int = TOP_K
) -> list[dict]:
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    # Embed the user's question.
    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    # ChromaDB performs the vector similarity search.
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(top_k, total_chunks),
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []

    for index, text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][index]
        distance = results["distances"][0][index]

        retrieved.append({
            "source": metadata["source"],
            "text": text,
            # With cosine distance, similarity = 1 - distance.
            "score": 1.0 - float(distance)
        })

    return retrieved
