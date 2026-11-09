import hashlib

import streamlit as st

from src.pdf_processor import extract_pdf_pages as read_pdf
from src.chunking import chunk_pages
from src.embeddings import load_embedding_model as load_embedder
from src.retriever import build_faiss_index as build_index
from src.retriever import retrieve_chunks
from src.generator import generate_answer as generate_answer_core


# ==========================================
# CONFIGURATION
# ==========================================

APP_TITLE = "Startup Docs Q&A Assistant"

LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_WORDS = 200
OVERLAP = 40
TOP_K = 4

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide"
)


# ==========================================
# PIPELINE COMPATIBILITY FUNCTIONS
# ==========================================

def retrieve(question, index, chunks, embedder):
    return retrieve_chunks(
        question=question,
        index=index,
        chunks=chunks,
        model=embedder,
        top_k=TOP_K
    )


def generate_answer(question, retrieved_chunks):
    return generate_answer_core(
        question=question,
        retrieved_chunks=retrieved_chunks,
        model_name=LLM_MODEL
    )


def process_document(pdf_bytes, embedder):
    pages = read_pdf(pdf_bytes)

    chunks = chunk_pages(
        pages,
        chunk_words=CHUNK_WORDS,
        overlap=OVERLAP
    )

    if not chunks:
        raise ValueError("No text chunks could be generated.")

    index = build_index(chunks, embedder)

    return pages, chunks, index


# ==========================================
# STREAMLIT INTERFACE
# ==========================================

st.title("📚 Startup Docs Q&A Assistant")

st.caption(
    "Ask questions about your PDF documents using "
    "Retrieval-Augmented Generation (RAG). "
    "Runs locally using FAISS and Ollama."
)

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write(f"**LLM:** {LLM_MODEL}")

    st.write(f"**Embedding:** {EMBEDDING_MODEL}")

    st.write(f"**Chunk size:** {CHUNK_WORDS} words")

    st.write(f"**Overlap:** {OVERLAP} words")

    st.write(f"**Top K:** {TOP_K}")

    st.divider()

    st.caption("Powered by Python, FAISS and Ollama.")


uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    document_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    if st.session_state.get("document_hash") != document_hash:

        st.session_state.pop("document_hash", None)
        st.session_state.pop("chunks", None)
        st.session_state.pop("index", None)
        st.session_state.pop("pages", None)
        st.session_state.pop("chat_history", None)

        with st.spinner("Reading and indexing your document..."):

            try:

                embedder = load_embedder()

                pages, chunks, index = process_document(
                    pdf_bytes,
                    embedder
                )

                st.session_state.document_hash = document_hash
                st.session_state.doc_name = uploaded_file.name
                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.chat_history = []

                st.success("Document indexed successfully!")

            except Exception as error:

                st.error(f"Document processing failed: {error}")
                st.stop()

    if "index" in st.session_state:

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Pages processed",
            len(st.session_state.pages)
        )

        col2.metric(
            "Text chunks",
            len(st.session_state.chunks)
        )

        col3.metric(
            "Retrieved chunks",
            TOP_K
        )

        st.subheader("💬 Ask your document")

        question = st.chat_input(
            "Ask anything about your PDF..."
        )

        if question:

            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            try:

                with st.spinner("Searching document and generating answer..."):

                    embedder = load_embedder()

                    hits = retrieve(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        embedder
                    )

                    reply = generate_answer(
                        question,
                        hits
                    )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": reply,
                        "sources": hits
                    })

            except Exception as error:

                st.error(
                    f"Unable to generate answer: {error}"
                )

        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                if message["role"] == "assistant" and message.get("sources"):

                    with st.expander("📖 View source passages"):

                        for source in message["sources"]:

                            st.markdown(
                                f"**Page {source['page']} "
                                f"| Similarity: {source['score']:.3f}**"
                            )

                            st.write(source["text"])

                            st.divider()

else:

    st.info(
        "Upload a PDF to begin asking questions."
    )

    st.markdown(
        """
        **Features**

        - PDF text extraction
        - Overlapping text chunking
        - Semantic search using FAISS
        - Local LLM-powered answers
        - Page-level source references
        - No external LLM API required
        """
    )
