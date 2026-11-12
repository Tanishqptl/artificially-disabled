
import re
import time
import pandas as pd

from src.pdf_processor import process_all_pdfs
from src.chunking import create_chunks
from src.retrieval import (
    build_index,
    dense_search,
    hybrid_search,
    rerank_results
)
from src.answer_generator import generate_answer


QUESTION_FILE = "evaluation/evaluation_questions.csv"
RESULT_FILE = "evaluation/evaluation_results.csv"
COMPARISON_FILE = "evaluation/retrieval_comparison.csv"


def extract_articles(results):
    articles = []

    for item in results:
        section = str(item.get("section", ""))

        match = re.search(
            r"article\s+(\d{1,3})",
            section,
            re.IGNORECASE
        )

        if match:
            articles.append(match.group(1))

    return sorted(set(articles), key=int)


def extract_expected_articles(value):
    return sorted(
        set(re.findall(r"article\s+(\d{1,3})", str(value), re.I)),
        key=int
    )


def main():

    questions = pd.read_csv(QUESTION_FILE)

    print("Total evaluation questions:", len(questions))

    # 1. Extract PDFs
    print("\nExtracting legal PDFs...")

    pages = process_all_pdfs()

    if not pages:
        raise ValueError("No PDF documents found.")

    # 2. Create chunks
    print("\nCreating chunks...")

    chunks = create_chunks(pages)

    print("Total pages:", len(pages))
    print("Total chunks:", len(chunks))

    # 3. Build retrieval index once
    print("\nBuilding retrieval index...")

    bm25, embeddings, chunks = build_index(chunks)

    results_rows = []
    comparison_rows = []

    # 4. Evaluate every question
    for number, row in questions.iterrows():

        question_id = row["question_id"]
        question = str(row["question"])

        print("\n" + "=" * 60)
        print(f"Question {number + 1}/{len(questions)}")
        print(question)

        # --------------------------------
        # A. Dense-only retrieval
        # --------------------------------

        start = time.perf_counter()

        dense_results = dense_search(
            question,
            embeddings,
            top_k=10
        )

        dense_time = time.perf_counter() - start

        dense_evidence = []

        for idx, score in dense_results:

            chunk = chunks[idx]

            dense_evidence.append({
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "section": chunk["section"],
                "text": chunk["text"],
                "dense_score": float(score)
            })

        # --------------------------------
        # B. Hybrid retrieval
        # --------------------------------

        start = time.perf_counter()

        hybrid_results = hybrid_search(
            query=question,
            bm25=bm25,
            embeddings=embeddings,
            chunks=chunks,
            top_k=5,
            candidate_k=10
        )

        hybrid_time = time.perf_counter() - start

        # --------------------------------
        # C. Cross-encoder reranking
        # --------------------------------

        start = time.perf_counter()

        final_results = rerank_results(
            question,
            hybrid_results,
            chunks
        )

        rerank_time = time.perf_counter() - start

        # --------------------------------
        # D. Generate legal answer
        # --------------------------------

        start = time.perf_counter()

        answer = generate_answer(
            question,
            final_results
        )

        answer_time = time.perf_counter() - start

        print("\nMODEL ANSWER:")
        print(answer)

        # --------------------------------
        # E. Collect citation information
        # --------------------------------

        citations = sorted(set(
            f"{item['document_name']} - "
            f"Page {item['page_number']} - "
            f"{item['section']}"
            for item in final_results
        ))

        dense_articles = extract_articles(dense_evidence)
        hybrid_articles = extract_articles(final_results)

        expected_articles = extract_expected_articles(
            row["expected_article_or_section"]
        )

        dense_hits = sum(
            article in dense_articles
            for article in expected_articles
        )

        hybrid_hits = sum(
            article in hybrid_articles
            for article in expected_articles
        )

        # --------------------------------
        # F. Save answer results
        # --------------------------------

        results_rows.append({
            "question_id": question_id,
            "question_type": row["question_type"],
            "question": question,
            "model_answer": answer,
            "retrieved_citations": "; ".join(citations),
            "correctness_score": "",
            "citation_precision": "",
            "abstained_correctly": "",
            "latency_seconds": round(
                dense_time + hybrid_time +
                rerank_time + answer_time, 3
            ),
            "notes": "Manual evaluation required"
        })

        # --------------------------------
        # G. Save retrieval comparison
        # --------------------------------

        comparison_rows.append({
            "question_id": question_id,
            "question_type": row["question_type"],
            "expected_articles": "; ".join(expected_articles),
            "dense_top10_articles": "; ".join(dense_articles),
            "hybrid_top10_articles": "; ".join(hybrid_articles),
            "dense_expected_article_hits": dense_hits,
            "hybrid_expected_article_hits": hybrid_hits,
            "dense_latency_seconds": round(dense_time, 3),
            "hybrid_latency_seconds": round(hybrid_time, 3),
            "reranking_latency_seconds": round(rerank_time, 3),
            "answer_latency_seconds": round(answer_time, 3)
        })

        # Save after every question
        pd.DataFrame(results_rows).to_csv(
            RESULT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        pd.DataFrame(comparison_rows).to_csv(
            COMPARISON_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nQuestion completed:", question_id)
        print("Dense latency:", round(dense_time, 3), "seconds")
        print("Hybrid latency:", round(hybrid_time, 3), "seconds")
        print("Answer latency:", round(answer_time, 3), "seconds")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")

    print("Results saved:", RESULT_FILE)
    print("Comparison saved:", COMPARISON_FILE)

    print("Questions evaluated:", len(results_rows))

    print(
        "\nRemember: manually verify correctness, "
        "citation precision and abstention."
    )


if __name__ == "__main__":
    main()