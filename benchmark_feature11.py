
import csv
import time
from pathlib import Path

from src.pdf_processor import process_all_pdfs
from src.chunking import create_chunks
from src.retrieval import (
    build_index,
    hybrid_search,
    rerank_results,
)
from src.answer_generator import generate_answer


# ==================================================
# FEATURE 11: LATENCY AND COST BENCHMARK
# ==================================================

QUESTION = (
    "Compare GDPR Article 22 on automated individual decision-making "
    "with EU AI Act Article 14 on human oversight of high-risk AI systems. "
    "What safeguards protect individuals?"
)

OUTPUT_DIR = Path("evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "feature11_results.csv"

# Gemini 3.8 Flash Standard paid-tier pricing
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 3.75


# ==================================================
# 1. HELPER FUNCTIONS
# ==================================================

def measure_time(function, *args, **kwargs):

    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start

    return result, round(elapsed, 4)


def check_target_evidence(results):

    article_22_found = any(
        "Article 22" in str(item.get("section", ""))
        for item in results
    )

    article_14_found = any(
        "Article 14" in str(item.get("section", ""))
        for item in results
    )

    return article_22_found, article_14_found


def calculate_cost(prompt_tokens, answer_tokens):

    input_cost = (
        prompt_tokens * INPUT_PRICE_PER_MILLION / 1_000_000
    )

    output_cost = (
        answer_tokens * OUTPUT_PRICE_PER_MILLION / 1_000_000
    )

    return round(input_cost + output_cost, 10)


def save_results(rows):

    columns = [
        "experiment",
        "candidate_k",
        "top_k",
        "pdf_extraction_seconds",
        "chunking_seconds",
        "index_build_seconds",
        "retrieval_seconds",
        "reranking_seconds",
        "retrieval_and_reranking_seconds",
        "results_count",
        "documents_retrieved",
        "article_22_found",
        "article_14_found",
        "gemini_seconds",
        "model",
        "prompt_tokens",
        "answer_tokens",
        "total_tokens",
        "estimated_paid_cost_usd",
        "notes",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\nResults saved to:", OUTPUT_FILE)


# ==================================================
# 2. EXTRACT DOCUMENTS ONCE
# ==================================================

print("\n========== ARTIFICIALLY DISABLED: FEATURE 11 ==========")

print("\n[1/6] Extracting PDFs...", flush=True)

pages, extraction_time = measure_time(process_all_pdfs)

if not pages:
    raise SystemExit("No PDF pages were extracted.")

print("Pages:", len(pages))
print("Extraction time:", extraction_time, "seconds")


# ==================================================
# 3. CREATE CHUNKS ONCE
# ==================================================

print("\n[2/6] Creating chunks...", flush=True)

chunks, chunking_time = measure_time(create_chunks, pages)

if not chunks:
    raise SystemExit("No chunks were created.")

print("Total chunks:", len(chunks))
print("Chunking time:", chunking_time, "seconds")


# ==================================================
# 4. BUILD INDEX ONCE
# ==================================================

print("\n[3/6] Building retrieval index...", flush=True)

(bm25, embeddings, chunks), index_time = measure_time(
    build_index,
    chunks
)

print("Index build time:", index_time, "seconds")


# ==================================================
# 5. WARM UP CROSS-ENCODER
# ==================================================

print("\nWarming up cross-encoder...", flush=True)

# This warm-up is excluded from benchmark timings.
if chunks:

    from src.retrieval import reranker_model

    reranker_model.predict([
        (QUESTION, chunks[0]["text"])
    ])

print("Warm-up complete.")


# ==================================================
# 6. BASELINE VS OPTIMISED RETRIEVAL
# ==================================================

experiments = [
    {
        "name": "baseline",
        "candidate_k": 20,
        "top_k": 10,
    },
    {
        "name": "optimised",
        "candidate_k": 10,
        "top_k": 5,
    },
]

rows = []
experiment_results = {}

for experiment in experiments:

    name = experiment["name"]
    candidate_k = experiment["candidate_k"]
    top_k = experiment["top_k"]

    print("\n========================================")
    print("EXPERIMENT:", name.upper())
    print("Candidate K:", candidate_k)
    print("Top K:", top_k)
    print("========================================")

    # Hybrid retrieval timing
    print("\nRunning hybrid retrieval...", flush=True)

    results, retrieval_time = measure_time(
        hybrid_search,
        query=QUESTION,
        bm25=bm25,
        embeddings=embeddings,
        chunks=chunks,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    # Cross-encoder reranking timing
    print("\nRunning cross-encoder reranking...", flush=True)

    final_results, reranking_time = measure_time(
        rerank_results,
        QUESTION,
        results,
        chunks,
    )

    total_retrieval_time = round(
        retrieval_time + reranking_time,
        4
    )

    article_22_found, article_14_found = (
        check_target_evidence(final_results)
    )

    documents = sorted({
        item["document_name"]
        for item in final_results
    })

    experiment_results[name] = final_results

    print("\nResults:", len(final_results))
    print("Documents:", documents)
    print("Article 22 found:", article_22_found)
    print("Article 14 found:", article_14_found)
    print("Retrieval time:", retrieval_time, "seconds")
    print("Reranking time:", reranking_time, "seconds")
    print("Combined time:", total_retrieval_time, "seconds")

    rows.append({
        "experiment": name,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "pdf_extraction_seconds": extraction_time,
        "chunking_seconds": chunking_time,
        "index_build_seconds": index_time,
        "retrieval_seconds": retrieval_time,
        "reranking_seconds": reranking_time,
        "retrieval_and_reranking_seconds": total_retrieval_time,
        "results_count": len(final_results),
        "documents_retrieved": "; ".join(documents),
        "article_22_found": article_22_found,
        "article_14_found": article_14_found,
        "gemini_seconds": "",
        "model": "",
        "prompt_tokens": "",
        "answer_tokens": "",
        "total_tokens": "",
        "estimated_paid_cost_usd": "",
        "notes": "Retrieval benchmark; Gemini not called.",
    })


# ==================================================
# 7. GENERATE ONE ANSWER USING OPTIMISED EVIDENCE
# ==================================================

print("\n[6/6] Generating Gemini answer...", flush=True)

optimized_results = experiment_results["optimised"]

if not optimized_results:
    raise SystemExit("No optimised evidence available.")

# Use up to five reranked chunks.
answer, usage = generate_answer(
    QUESTION,
    optimized_results[:5],
    return_usage=True,
)

print("\n========== GEMINI ANSWER ==========")
print(answer)

gemini_time = usage.get("answer_generation_seconds", "")

prompt_tokens = usage.get("prompt_tokens", 0)
answer_tokens = usage.get("answer_tokens", 0)
total_tokens = usage.get("total_tokens", 0)

estimated_cost = calculate_cost(
    prompt_tokens,
    answer_tokens
)

print("\n========== GEMINI USAGE ==========")
print("Model:", usage.get("model", "Not available"))
print("Prompt tokens:", prompt_tokens)
print("Answer tokens:", answer_tokens)
print("Total tokens:", total_tokens)
print("Generation time:", gemini_time, "seconds")
print("Estimated paid-tier cost: $", estimated_cost)


# ==================================================
# 8. SAVE GEMINI MEASUREMENTS
# ==================================================

rows.append({
    "experiment": "gemini_optimised_answer",
    "candidate_k": 10,
    "top_k": 5,
    "pdf_extraction_seconds": "",
    "chunking_seconds": "",
    "index_build_seconds": "",
    "retrieval_seconds": "",
    "reranking_seconds": "",
    "retrieval_and_reranking_seconds": "",
    "results_count": len(optimized_results),
    "documents_retrieved": "; ".join(sorted({
        item["document_name"]
        for item in optimized_results
    })),
    "article_22_found": check_target_evidence(
        optimized_results
    )[0],
    "article_14_found": check_target_evidence(
        optimized_results
    )[1],
    "gemini_seconds": gemini_time,
    "model": usage.get("model", ""),
    "prompt_tokens": prompt_tokens,
    "answer_tokens": answer_tokens,
    "total_tokens": total_tokens,
    "estimated_paid_cost_usd": estimated_cost,
    "notes": (
        "One Gemini API call using optimised retrieval evidence. "
        "Cost is a paid-tier list-price estimate."
    ),
})


# ==================================================
# 9. COMPARE PERFORMANCE
# ==================================================

baseline = next(
    row for row in rows
    if row["experiment"] == "baseline"
)

optimized = next(
    row for row in rows
    if row["experiment"] == "optimised"
)

baseline_time = baseline["retrieval_and_reranking_seconds"]
optimized_time = optimized["retrieval_and_reranking_seconds"]

if baseline_time > 0:

    improvement = (
        (baseline_time - optimized_time)
        / baseline_time
    ) * 100

else:
    improvement = 0

print("\n========== PERFORMANCE COMPARISON ==========")

print("Baseline time:", baseline_time, "seconds")
print("Optimised time:", optimized_time, "seconds")
print("Time improvement:", round(improvement, 2), "%")

print(
    "Baseline Article 22 coverage:",
    baseline["article_22_found"]
)

print(
    "Optimised Article 22 coverage:",
    optimized["article_22_found"]
)

print(
    "Baseline Article 14 coverage:",
    baseline["article_14_found"]
)

print(
    "Optimised Article 14 coverage:",
    optimized["article_14_found"]
)


# ==================================================
# 10. SAVE CSV REPORT
# ==================================================

save_results(rows)

print("\n========== FEATURE 11 COMPLETED ==========")