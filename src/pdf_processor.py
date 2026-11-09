
import io
from pypdf import PdfReader


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.
    """

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    reader = PdfReader(io.BytesIO(pdf_bytes))

    if reader.is_encrypted:
        raise ValueError(
            "Password-protected PDFs are not supported."
        )

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    if not pages:
        raise ValueError(
            "No readable text found. The PDF may require OCR."
        )

    return pages
