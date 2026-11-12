
import time

from src.pdf_processor import process_all_pdfs
from src.chunking import create_chunks
from src.retrieval import (
    build_index,
    hybrid_search,
    rerank_results
)
from src.answer_generator import generate_answer


# --------------------------------------------------
# 1. Question
# --------------------------------------------------

question = (
    "Compare GDPR Article 22 on automated individual decision-making "
    "with EU AI Act Article 14 on human oversight of high-risk AI systems. "
    "What safeguards protect individuals?"
)


# --------------------------------------------------
# 2. Extract and process PDFs
# --------------------------------------------------

print("\n[1/5] Extracting PDF documents...", flush=True)

start_time = time.time()

pages = process_all_pdfs()

print("Pages extracted:", len(pages), flush=True)


# --------------------------------------------------
# 3. Create chunks
# --------------------------------------------------

print("\n[2/5] Creating document chunks...", flush=True)

chunks = create_chunks(pages)

print("Total chunks:", len(chunks), flush=True)


# --------------------------------------------------
# 4. Build retrieval index
# --------------------------------------------------

print("\n[3/5] Building retrieval index...", flush=True)

bm25, embeddings, chunks = build_index(chunks)

print("Index built successfully.", flush=True)


# --------------------------------------------------
# 5. Hybrid retrieval
# --------------------------------------------------

print("\n[4/5] Searching both legal documents...", flush=True)

results = hybrid_search(
    query=question,
    bm25=bm25,
    embeddings=embeddings,
    chunks=chunks,
    top_k=5,
    candidate_k=10
)

print("Hybrid retrieval completed.", flush=True)


# --------------------------------------------------
# 6. Cross-encoder reranking
# --------------------------------------------------

print("Reranking top candidates...", flush=True)

final_results = rerank_results(
    question,
    results[:10],
    chunks
)

print("\n========== CROSS-DOCUMENT RESULTS ==========")

for result in final_results[:5]:
    print("\nDocument:", result["document_name"])
    print("Page:", result["page_number"])
    print("Section:", result["section"])
    print("Score:", result["reranker_score"])
    print("Text:", result["text"][:500])


# --------------------------------------------------
# 7. Confirm both documents were retrieved
# --------------------------------------------------

documents = set(
    result["document_name"]
    for result in final_results
)

print("\nDocuments retrieved:")
print(documents)

if len(documents) < 2:
    print("\nWARNING: Both legal documents were not retrieved.")
    print("Please check the retrieval results.")
    raise SystemExit


# --------------------------------------------------
# 8. Generate comparison answer
# --------------------------------------------------

print("\n[5/5] Preparing evidence for Gemini...", flush=True)

# Send only the top 2 reranked results
evidence = final_results[:2]

print("Sending question to Gemini...", flush=True)

answer_start = time.time()

answer = generate_answer(
    question,
    evidence
)

print("\n========== AI COMPARISON ANSWER ==========")
print(answer)

print(
    "\nAnswer generation time:",
    round(time.time() - answer_start, 2),
    "seconds"
)

print("\n========== FEATURE 9 COMPLETED ==========")
