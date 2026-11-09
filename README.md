# Startup Docs Q&A Assistant

A Retrieval-Augmented Generation (RAG) application for asking questions about uploaded PDF documents.

## Features
- PDF text extraction using PyPDF
- Text chunking with overlap and page metadata
- Semantic embeddings using Sentence Transformers
- FAISS-based similarity search
- Local answer generation using Ollama and Llama 3.2
- Source-page references
- Abstention when information is unavailable
- Interactive Streamlit interface

## Tech Stack
Python, Streamlit, PyPDF, Sentence Transformers, FAISS, Ollama, Llama 3.2, NumPy, Pytest.

## Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Start Ollama: `ollama serve`
3. Pull the model: `ollama pull llama3.2`
4. Run the app: `streamlit run app.py`

## Testing
27 automated tests passed.

## Evaluation
On the current single-PDF evaluation:
- Evidence retrieval: 10/10
- Source-page accuracy: 10/10
- Negative-question abstention: 3/3

This is a limited evaluation using one PDF and one chunk, not a multi-document benchmark.

## Limitations
The application has not yet been validated across a diverse collection of PDFs. Human review is recommended.
