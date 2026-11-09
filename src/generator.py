

import re
import ollama

LLM_MODEL = "llama3.2"

ABSTENTION = "I could not find this information in the document."

SYSTEM_PROMPT = """
You are a document question-answering assistant.

RULES:
1. Answer using only the supplied document context.
2. Never invent facts, measurements, names, or citations.
3. Distinguish field definitions from actual observations.
4. A column description is not a patient record.
5. Cite factual claims using [Page X].
6. If information is absent, clearly abstain.
7. Keep answers concise and evidence-based.
"""


def extract_field_definitions(chunks):
    """Extract numbered field definitions using exact field names."""
    definitions = {}

    for chunk in chunks:
        text = chunk["text"]

        matches = list(re.finditer(
            r"(?:^|\s)(\d+)\.\s*([A-Za-z_]\w*)\s*",
            text
        ))

        for i, match in enumerate(matches):
            field = match.group(2).lower()
            start = match.end()
            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )

            definition = text[start:end].strip()
            definition = re.sub(r"^[\s:–-]+", "", definition)
            definition = definition.strip(" ;.")

            if definition:
                definitions[field] = {
                    "definition": definition,
                    "page": chunk["page"]
                }

    return definitions


def is_schema_only_document(chunks):
    """Identify documents that describe columns rather than records."""
    text = " ".join(chunk["text"] for chunk in chunks).lower()

    return (
        "column descriptions" in text
        and bool(re.search(r"\b\d+\.\s*[a-z_]\w*", text))
    )


def asks_for_actual_patient_data(question):
    """Detect requests for individual-level information."""
    q = question.lower()

    patterns = [
        r"\bpatient'?s?\s+name\b",
        r"\bmedication\b",
        r"\bmedicine\b",
        r"\bprescribed\b",
        r"\bblood test result\b",
        r"\btest result\b",
        r"\bactual patient\b",
        r"\bpatient'?s?\s+(?:value|measurement|diagnosis|result)\b",
        r"\bactual value\b",
        r"\bindividual\s+(?:record|measurement|result)\b",
    ]

    return any(re.search(pattern, q) for pattern in patterns)


def find_definition_answer(question, chunks):
    """Answer direct field-definition questions deterministically."""
    q = question.lower().strip()

    intents = (
        "what does",
        "what is",
        "what are",
        "define",
        "meaning of",
        "represent",
        "describe",
    )

    if not any(intent in q for intent in intents):
        return None

    definitions = extract_field_definitions(chunks)

    tokens = re.findall(r"[a-z_][a-z0-9_]*", q)

    matched = [
        field for field in definitions
        if field in tokens
    ]

    if not matched:
        return None

    field = max(matched, key=len)
    item = definitions[field]

    return (
        f"{field} represents {item['definition']}. "
        f"[Page {item['page']}]"
    )


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    model_name: str = LLM_MODEL
) -> str:

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return ABSTENTION

    # 1. Direct, exact field-definition answer.
    direct_answer = find_definition_answer(
        question, retrieved_chunks
    )

    if direct_answer is not None:
        return direct_answer

    # 2. Do not invent patient records from schema-only documents.
    if (
        is_schema_only_document(retrieved_chunks)
        and asks_for_actual_patient_data(question)
    ):
        return ABSTENTION

    # 3. Use the LLM for other questions.
    context = "\n\n".join(
        f"[Source {i} | Page {chunk['page']}]\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks, start=1)
    )

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"DOCUMENT CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "Answer only from the document. "
                    "Do not confuse definitions with observations. "
                    "Cite supported claims with page numbers."
                )
            }
        ],
        options={"temperature": 0.1}
    )

    return response["message"]["content"].strip()
