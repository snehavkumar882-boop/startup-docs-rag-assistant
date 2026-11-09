
from pathlib import Path

from src.pdf_processor import extract_pdf_pages
from src.chunking import chunk_pages
from src.embeddings import load_embedding_model
from src.retriever import build_faiss_index, retrieve_chunks
from src.generator import generate_answer


PDF_PATH = Path("data/heart_datasetDescription.pdf")
TOP_K = 4

# Questions with answers supported by the PDF.
POSITIVE_TEST_CASES = [
    {
        "question": "What does thalach represent?",
        "expected": "maximum heart rate achieved",
        "page": 1
    },
    {
        "question": "What is the predicted attribute?",
        "expected": "num: the predicted attribute",
        "page": 1
    },
    {
        "question": "What does exang represent?",
        "expected": "exercise-induced angina",
        "page": 1
    },
    {
        "question": "What does oldpeak measure?",
        "expected": "ST depression induced by exercise relative to rest",
        "page": 1
    },
    {
        "question": "What does chol represent?",
        "expected": "serum cholesterol",
        "page": 1
    },
    {
        "question": "What does trestbps represent?",
        "expected": "resting blood pressure",
        "page": 1
    },
    {
        "question": "What does restecg represent?",
        "expected": "resting electrocardiographic results",
        "page": 1
    },
    {
        "question": "What does ca represent?",
        "expected": "number of major vessels",
        "page": 1
    },
    {
        "question": "What values are described for thal?",
        "expected": "normal; fixed defect; reversible defect",
        "page": 1
    },
    {
        "question": "What does sex represent?",
        "expected": "Male/Female",
        "page": 1
    }
]

# Questions whose answers are not provided by this PDF.
NEGATIVE_TEST_CASES = [
    {
        "question": "What is the patient's name?"
    },
    {
        "question": "What medication was prescribed?"
    },
    {
        "question": "What was the patient's blood test result?"
    }
]

ABSTENTION_PHRASES = [
    "could not find",
    "not specified",
    "not mentioned",
    "not provided",
    "does not contain",
    "cannot determine",
    "unable to determine",
    "not available in the document"
]


def load_evaluation_document():
    """Extract and chunk the evaluation PDF."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation PDF not found: {PDF_PATH}"
        )

    pdf_bytes = PDF_PATH.read_bytes()

    pages = extract_pdf_pages(pdf_bytes)
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No chunks generated from evaluation PDF.")

    return pages, chunks


def evaluate_retrieval(chunks, index, model):
    """Evaluate whether retrieved passages contain expected evidence."""

    passed = 0
    reciprocal_ranks = []

    print("\nPOSITIVE RETRIEVAL EVALUATION")
    print("=" * 65)

    for number, test in enumerate(POSITIVE_TEST_CASES, start=1):

        results = retrieve_chunks(
            question=test["question"],
            index=index,
            chunks=chunks,
            model=model,
            top_k=TOP_K
        )

        relevant_ranks = []

        for rank, result in enumerate(results, start=1):
            evidence = test["expected"].lower()

            if evidence in result["text"].lower():
                relevant_ranks.append(rank)

        evidence_found = bool(relevant_ranks)

        if evidence_found:
            passed += 1
            reciprocal_ranks.append(1 / relevant_ranks[0])
        else:
            reciprocal_ranks.append(0)

        print(f"\n{number}. {test['question']}")
        print(f"Expected evidence: {test['expected']}")
        print(f"Result: {'PASS' if evidence_found else 'FAIL'}")

        if results:
            print(f"Top retrieved page: {results[0]['page']}")
            print(f"Top similarity: {results[0]['score']:.4f}")

        if relevant_ranks:
            print(f"First relevant rank: {relevant_ranks[0]}")

    total = len(POSITIVE_TEST_CASES)

    hit_rate = passed / total
    mrr = sum(reciprocal_ranks) / total

    print("\nRETRIEVAL SUMMARY")
    print("-" * 65)
    print(f"Evidence hits: {passed}/{total}")
    print(f"Evidence hit rate: {hit_rate:.2%}")
    print(f"MRR: {mrr:.4f}")

    print(
        "\nNote: These retrieval metrics are not meaningful "
        "benchmarks when the document contains only one chunk."
    )

    return passed, total


def evaluate_citations(chunks, index, model):
    """Check whether retrieved evidence has the expected page number."""

    passed = 0

    print("\nCITATION SOURCE EVALUATION")
    print("=" * 65)

    for test in POSITIVE_TEST_CASES:

        results = retrieve_chunks(
            question=test["question"],
            index=index,
            chunks=chunks,
            model=model,
            top_k=TOP_K
        )

        correct_source = any(
            test["expected"].lower() in result["text"].lower()
            and result["page"] == test["page"]
            for result in results
        )

        if correct_source:
            passed += 1

        print(
            f"{test['question']} -> "
            f"{'PASS' if correct_source else 'FAIL'}"
        )

    total = len(POSITIVE_TEST_CASES)

    print(f"\nCorrect source pages: {passed}/{total}")
    print(f"Source page accuracy: {passed / total:.2%}")


def evaluate_abstention(chunks, index, model):
    """Check whether the LLM abstains on unsupported questions."""

    passed = 0

    print("\nNEGATIVE QUESTION / ABSTENTION EVALUATION")
    print("=" * 65)

    for number, test in enumerate(NEGATIVE_TEST_CASES, start=1):

        results = retrieve_chunks(
            question=test["question"],
            index=index,
            chunks=chunks,
            model=model,
            top_k=TOP_K
        )

        answer = generate_answer(
            question=test["question"],
            retrieved_chunks=results
        )

        abstained = any(
            phrase in answer.lower()
            for phrase in ABSTENTION_PHRASES
        )

        if abstained:
            passed += 1

        print(f"\n{number}. {test['question']}")
        print(f"Model answer: {answer}")
        print(f"Abstention: {'PASS' if abstained else 'FAIL'}")

    total = len(NEGATIVE_TEST_CASES)

    print("\nABSTENTION SUMMARY")
    print("-" * 65)
    print(f"Correct abstentions: {passed}/{total}")
    print(f"Abstention rate: {passed / total:.2%}")

    return passed, total


def main():
    print("\nSTARTUP DOCS Q&A ASSISTANT - RAG EVALUATION")
    print("=" * 65)

    pages, chunks = load_evaluation_document()

    print(f"Document: {PDF_PATH.name}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks generated: {len(chunks)}")

    print("\nLoading embedding model...")
    model = load_embedding_model()

    print("Building FAISS index...")
    index = build_faiss_index(chunks, model)

    retrieval_passed, retrieval_total = evaluate_retrieval(
        chunks, index, model
    )

    evaluate_citations(chunks, index, model)

    abstention_passed, abstention_total = evaluate_abstention(
        chunks, index, model
    )

    print("\nFINAL EVALUATION REPORT")
    print("=" * 65)
    print(
        f"Positive evidence retrieval: "
        f"{retrieval_passed}/{retrieval_total}"
    )
    print(
        f"Negative question abstention: "
        f"{abstention_passed}/{abstention_total}"
    )

    print("\nEVALUATION LIMITATIONS")
    print("- Only one PDF was evaluated.")
    print("- The PDF generates only one chunk.")
    print("- MRR is not a meaningful ranking benchmark here.")
    print("- Abstention checks use response wording heuristics.")
    print("- Human review is still required to verify answer faithfulness.")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()

