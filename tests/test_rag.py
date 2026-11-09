
import io

import faiss
import numpy as np
import pytest

from pypdf import PdfWriter

from src.pdf_processor import extract_pdf_pages
from src.chunking import chunk_pages
from src.embeddings import generate_embeddings
from src.retriever import build_faiss_index, retrieve_chunks
from src.generator import (
    generate_answer,
    extract_field_definitions,
    find_definition_answer,
    is_schema_only_document,
)


# ============================================================
# TEST FIXTURES
# ============================================================

class FakeEmbeddingModel:
    """Mock embedding model for unit tests."""

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    ):
        vectors = []

        for text in texts:
            text = text.lower()

            if "apple" in text:
                vector = [1.0, 0.0, 0.0]
            elif "banana" in text:
                vector = [0.0, 1.0, 0.0]
            else:
                vector = [0.0, 0.0, 1.0]

            vectors.append(vector)

        vectors = np.asarray(vectors, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(
                vectors,
                axis=1,
                keepdims=True
            )
            vectors = vectors / norms

        return vectors


@pytest.fixture
def fake_model():
    return FakeEmbeddingModel()


@pytest.fixture
def sample_chunks():
    return [
        {"page": 1, "text": "apple fruit"},
        {"page": 2, "text": "banana fruit"},
        {"page": 3, "text": "orange fruit"},
    ]


@pytest.fixture
def schema_chunks():
    return [
        {
            "page": 1,
            "text": (
                "Column Descriptions: "
                "1. id (Unique id for each patient) "
                "2. age (Age of the patient in years) "
                "3. chol (serum cholesterol in mg/dl) "
                "4. thalach: maximum heart rate achieved "
                "5. thal: [normal; fixed defect; reversible defect] "
                "6. num: the predicted attribute"
            ),
        }
    ]


# ============================================================
# 1. PDF EXTRACTION TESTS
# ============================================================

def test_empty_pdf():
    with pytest.raises(ValueError):
        extract_pdf_pages(b"")


def test_blank_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)

    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(ValueError):
        extract_pdf_pages(buffer.getvalue())


# ============================================================
# 2. CHUNKING TESTS
# ============================================================

def test_chunking_preserves_page_number():
    pages = [
        {"page": 5, "text": "one two three four five six"}
    ]

    chunks = chunk_pages(
        pages,
        chunk_words=3,
        overlap=1
    )

    assert len(chunks) == 3
    assert all(chunk["page"] == 5 for chunk in chunks)


def test_chunking_overlap():
    pages = [
        {"page": 1, "text": "one two three four five six"}
    ]

    chunks = chunk_pages(
        pages,
        chunk_words=4,
        overlap=2
    )

    assert chunks[0]["text"] == "one two three four"
    assert chunks[1]["text"] == "three four five six"


def test_empty_pages():
    pages = [
        {"page": 1, "text": ""},
        {"page": 2, "text": "   "}
    ]

    chunks = chunk_pages(pages)

    assert chunks == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_pages(
            [{"page": 1, "text": "some text"}],
            chunk_words=0
        )


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages(
            [{"page": 1, "text": "some text"}],
            chunk_words=5,
            overlap=5
        )


# ============================================================
# 3. EMBEDDING TESTS
# ============================================================

def test_embedding_output_shape(fake_model):
    embeddings = generate_embeddings(
        ["apple", "banana"],
        fake_model
    )

    assert embeddings.shape == (2, 3)


def test_embedding_dtype(fake_model):
    embeddings = generate_embeddings(
        ["apple", "banana"],
        fake_model
    )

    assert embeddings.dtype == np.float32


def test_embedding_normalization(fake_model):
    embeddings = generate_embeddings(
        ["apple", "banana"],
        fake_model
    )

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0)


def test_empty_embedding_input(fake_model):
    with pytest.raises(ValueError):
        generate_embeddings([], fake_model)


def test_blank_embedding_input(fake_model):
    with pytest.raises(ValueError):
        generate_embeddings(["apple", "   "], fake_model)


# ============================================================
# 4. FAISS RETRIEVAL TESTS
# ============================================================

def test_faiss_index_creation(
    sample_chunks,
    fake_model
):
    index = build_faiss_index(
        sample_chunks,
        fake_model
    )

    assert isinstance(index, faiss.IndexFlatIP)
    assert index.ntotal == 3


def test_retrieval_ranking(
    sample_chunks,
    fake_model
):
    index = build_faiss_index(
        sample_chunks,
        fake_model
    )

    results = retrieve_chunks(
        question="apple",
        index=index,
        chunks=sample_chunks,
        model=fake_model,
        top_k=3
    )

    assert results[0]["text"] == "apple fruit"


def test_retrieval_top_k(
    sample_chunks,
    fake_model
):
    index = build_faiss_index(
        sample_chunks,
        fake_model
    )

    results = retrieve_chunks(
        question="apple",
        index=index,
        chunks=sample_chunks,
        model=fake_model,
        top_k=2
    )

    assert len(results) == 2


def test_empty_index_input(fake_model):
    with pytest.raises(ValueError):
        build_faiss_index([], fake_model)


def test_empty_question(
    sample_chunks,
    fake_model
):
    index = build_faiss_index(
        sample_chunks,
        fake_model
    )

    with pytest.raises(ValueError):
        retrieve_chunks(
            question=" ",
            index=index,
            chunks=sample_chunks,
            model=fake_model
        )


def test_invalid_top_k(
    sample_chunks,
    fake_model
):
    index = build_faiss_index(
        sample_chunks,
        fake_model
    )

    with pytest.raises(ValueError):
        retrieve_chunks(
            question="apple",
            index=index,
            chunks=sample_chunks,
            model=fake_model,
            top_k=0
        )


# ============================================================
# 5. GENERATOR TESTS
# ============================================================

def test_generate_answer_success(monkeypatch):
    def mock_chat(**kwargs):
        return {
            "message": {
                "content": "The answer is supported. [Page 1]"
            }
        }

    monkeypatch.setattr(
        "src.generator.ollama.chat",
        mock_chat
    )

    chunks = [
        {"page": 1, "text": "The document contains a description."}
    ]

    answer = generate_answer(
        "Summarize the document",
        chunks
    )

    assert "supported" in answer


def test_generate_answer_without_context():
    answer = generate_answer(
        "What is the answer?",
        []
    )

    assert answer == "I could not find this information in the document."


def test_generate_answer_empty_question():
    with pytest.raises(ValueError):
        generate_answer("", [])


def test_generate_answer_uses_supplied_model(monkeypatch):
    captured = {}

    def mock_chat(**kwargs):
        captured.update(kwargs)

        return {
            "message": {
                "content": "Test answer"
            }
        }

    monkeypatch.setattr(
        "src.generator.ollama.chat",
        mock_chat
    )

    chunks = [
        {"page": 1, "text": "A general document description."}
    ]

    generate_answer(
        "Summarize this document",
        chunks,
        model_name="test-model"
    )

    assert captured["model"] == "test-model"


# ============================================================
# 6. GENERATOR SAFEGUARD TESTS
# ============================================================

def test_extract_field_definitions(schema_chunks):
    definitions = extract_field_definitions(schema_chunks)

    assert "thalach" in definitions
    assert "thal" in definitions

    assert (
        "maximum heart rate achieved"
        in definitions["thalach"]["definition"]
    )


def test_exact_field_matching_thalach(schema_chunks):
    answer = find_definition_answer(
        "What does thalach represent?",
        schema_chunks
    )

    assert answer is not None
    assert "maximum heart rate achieved" in answer
    assert "fixed defect" not in answer
    assert "[Page 1]" in answer


def test_exact_field_matching_thal(schema_chunks):
    answer = find_definition_answer(
        "What does thal represent?",
        schema_chunks
    )

    assert answer is not None
    assert "normal; fixed defect; reversible defect" in answer
    assert "maximum heart rate achieved" not in answer


def test_schema_only_document_detection(schema_chunks):
    assert is_schema_only_document(schema_chunks) is True


def test_patient_question_abstains_without_llm(
    schema_chunks,
    monkeypatch
):
    def unexpected_llm_call(*args, **kwargs):
        raise AssertionError(
            "LLM should not be called for protected patient-data questions."
        )

    monkeypatch.setattr(
        "src.generator.ollama.chat",
        unexpected_llm_call
    )

    answer = generate_answer(
        "What was the patient's blood test result?",
        schema_chunks
    )

    assert answer == "I could not find this information in the document."


# ============================================================
# END OF TEST SUITE
# ============================================================

