import faiss
import numpy as np

from src.embeddings import generate_embeddings


def build_faiss_index(
    chunks: list[dict],
    model
):
    """Build a FAISS index from document chunks."""

    if not chunks:
        raise ValueError("Cannot build an index from empty chunks.")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts, model)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Invalid embedding dimensions.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve_chunks(
    question: str,
    index,
    chunks: list[dict],
    model,
    top_k: int = 4
) -> list[dict]:
    """Retrieve the most relevant chunks using cosine similarity."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not chunks:
        return []

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and chunks are inconsistent.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_embedding = generate_embeddings([question], model)

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, chunk_id in zip(scores[0], indices[0]):

        if chunk_id < 0:
            continue

        result = chunks[int(chunk_id)].copy()
        result["score"] = float(score)

        results.append(result)

    return results
