def chunk_pages(
    pages: list[dict],
    chunk_words: int = 200,
    overlap: int = 40
) -> list[dict]:
    """
    Split extracted PDF pages into overlapping word-based chunks.

    Each chunk retains its original page number.
    """

    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero.")

    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must be >= 0 and smaller than chunk_words.")

    chunks = []
    step = chunk_words - overlap

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"].strip()

        if not text:
            continue

        words = text.split()

        for start in range(0, len(words), step):
            chunk_text = " ".join(words[start:start + chunk_words])

            if chunk_text:
                chunks.append({
                    "page": page_number,
                    "text": chunk_text
                })

            if start + chunk_words >= len(words):
                break

    return chunks
