
import re
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


# Support both:
# python src/retrieval.py
# from src.retrieval import ...
try:
    from .pdf_processor import process_all_pdfs
    from .chunking import create_chunks
    from .answer_generator import generate_answer
except ImportError:
    from pdf_processor import process_all_pdfs
    from chunking import create_chunks
    from answer_generator import generate_answer


# --------------------------------------------------
# 1. Load pretrained retrieval models
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading cross-encoder model...")

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------------------------
# 2. Text preprocessing
# --------------------------------------------------

def tokenize(text):

    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# 3. Build retrieval index
# --------------------------------------------------

def build_index(chunks):

    if not chunks:
        raise ValueError("No chunks available for indexing.")

    texts = [chunk["text"] for chunk in chunks]

    print("Building BM25 keyword index...")

    tokenized_texts = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    print("Generating transformer embeddings...")

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Retrieval index created successfully.")

    return bm25, embeddings, chunks


# --------------------------------------------------
# 4. BM25 keyword retrieval
# --------------------------------------------------

def bm25_search(query, bm25, chunks, top_k=20):

    scores = bm25.get_scores(tokenize(query))

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (int(i), float(scores[i]))
        for i in ranked_indices
        if scores[i] > 0
    ]


# --------------------------------------------------
# 5. Dense semantic retrieval
# --------------------------------------------------

def dense_search(query, embeddings, top_k=20):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (int(i), float(scores[i]))
        for i in ranked_indices
    ]


# --------------------------------------------------
# 6. Hybrid retrieval using Reciprocal Rank Fusion
# --------------------------------------------------

def hybrid_search(
    query,
    bm25,
    embeddings,
    chunks,
    top_k=10,
    candidate_k=20
):

    print("\nRunning BM25 retrieval...")

    bm25_results = bm25_search(
        query,
        bm25,
        chunks,
        top_k=candidate_k
    )

    print("BM25 candidates:", len(bm25_results))

    print("\nRunning dense semantic retrieval...")

    dense_results = dense_search(
        query,
        embeddings,
        top_k=candidate_k
    )

    print("Dense candidates:", len(dense_results))

    # Reciprocal Rank Fusion
    rrf_scores = {}
    k = 60

    for rank, (idx, score) in enumerate(bm25_results, 1):

        rrf_scores[idx] = (
            rrf_scores.get(idx, 0) + 1 / (k + rank)
        )

    for rank, (idx, score) in enumerate(dense_results, 1):

        rrf_scores[idx] = (
            rrf_scores.get(idx, 0) + 1 / (k + rank)
        )

    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nHybrid retrieval candidates:", len(ranked))

    return ranked[:top_k]


# --------------------------------------------------
# 7. Cross-encoder reranking
# --------------------------------------------------

def rerank_results(query, results, chunks):

    if not results:
        return []

    print("\nReranking evidence using cross-encoder...")

    pairs = [
        (query, chunks[idx]["text"])
        for idx, score in results
    ]

    scores = reranker_model.predict(pairs)

    reranked = []

    for (idx, old_score), new_score in zip(results, scores):

        chunk = chunks[idx]

        reranked.append({
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "section": chunk["section"],
            "text": chunk["text"],
            "hybrid_score": float(old_score),
            "reranker_score": float(new_score)
        })

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked


# --------------------------------------------------
# 8. Display retrieved evidence
# --------------------------------------------------

def display_evidence(results):

    print("\n========== TOP RETRIEVED EVIDENCE ==========")

    if not results:
        print("No relevant evidence found.")
        return

    for i, result in enumerate(results, 1):

        print(f"\nResult {i}")
        print("Document:", result["document_name"])
        print("Page:", result["page_number"])
        print("Section:", result["section"])
        print("Hybrid score:", round(result["hybrid_score"], 5))
        print("Reranker score:", round(result["reranker_score"], 5))
        print("Evidence:", result["text"][:500])
        print("----------------------------------")


# --------------------------------------------------
# 9. Complete Artificially Disabled pipeline
# --------------------------------------------------

def run_pipeline(query):

    if not query.strip():
        print("Please enter a legal question.")
        return

    print("\n========== ARTIFICIALLY DISABLED PIPELINE ==========")

    print("\nExtracting documents...")

    pages = process_all_pdfs()

    if not pages:
        print("No documents could be extracted.")
        return

    print("\nCreating section-aware chunks...")

    chunks = create_chunks(pages)

    print("Total chunks:", len(chunks))

    if not chunks:
        print("No chunks were created.")
        return

    print("\nBuilding retrieval index...")

    bm25, embeddings, chunks = build_index(chunks)

    print("\nSearching documents...")

    results = hybrid_search(
        query=query,
        bm25=bm25,
        embeddings=embeddings,
        chunks=chunks,
        top_k=10,
        candidate_k=20
    )

    final_results = rerank_results(
        query,
        results,
        chunks
    )

    display_evidence(final_results)

    print("\n========== GENERATING LEGAL ANSWER ==========")

    answer = generate_answer(
        query,
        final_results
    )

    print("\n========== FINAL ARTIFICIALLY DISABLED ANSWER ==========")
    print(answer)

    return {
        "question": query,
        "evidence": final_results,
        "answer": answer
    }


# --------------------------------------------------
# 10. Run from terminal
# --------------------------------------------------

if __name__ == "__main__":

    question = input("\nEnter your legal question: ").strip()

    run_pipeline(question)