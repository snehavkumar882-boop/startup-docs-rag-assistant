import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence-transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer
) -> np.ndarray:
    """Generate normalized float32 embeddings for input texts."""

    if not texts:
        raise ValueError("Cannot generate embeddings for empty text input.")

    if any(not text.strip() for text in texts):
        raise ValueError("Input texts cannot contain empty strings.")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(embeddings, dtype=np.float32)
