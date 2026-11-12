
import os
import tempfile
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Artificially Disabled",
    page_icon="⚖️",
    layout="wide"
)

# White background and dark text
st.markdown(
    """
    <style>
        .stApp {
            background-color: #ffffff;
            color: #1f2937;
        }

        [data-testid="stHeader"] {
            background-color: #ffffff;
        }

        [data-testid="stSidebar"] {
            background-color: #f8fafc;
        }

        h1, h2, h3, h4, h5, h6, p, label,
        .stMarkdown, .stText, .stCaption {
            color: #1f2937;
        }

        [data-testid="stFileUploader"] {
            background-color: #f8fafc;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            padding: 12px;
        }

        [data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }

        .stTextArea textarea {
            background-color: #ffffff;
            color: #1f2937;
            border: 1px solid #d1d5db;
        }

        .stButton button {
            border-radius: 8px;
            font-weight: 600;
        }

        [data-testid="stMetric"] {
            background-color: #f8fafc;
            padding: 12px;
            border-radius: 8px;
        }

        hr {
            border-color: #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 2. Application heading
# --------------------------------------------------

st.title("⚖️ Artificially Disabled")

st.success("Artificially Disabled application loaded successfully!")

st.subheader("Evidence-Grounded Legal Document Analyst")

st.write(
    "Upload legal documents, ask questions, and receive "
    "AI-generated answers supported by document evidence."
)

st.divider()


# --------------------------------------------------
# 3. Upload legal documents
# --------------------------------------------------

st.header("1. Upload Legal Documents")

uploaded_files = st.file_uploader(
    "Choose your legal PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload between 1 and 5 PDF documents."
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("Maximum 5 PDF documents allowed.")
    else:
        st.success(f"{len(uploaded_files)} document(s) uploaded.")

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

st.divider()


# --------------------------------------------------
# 4. Ask a legal question
# --------------------------------------------------

st.header("2. Ask a Legal Question")

question = st.text_area(
    "Enter your question",
    placeholder="Example: What are the administrative fines under GDPR?",
    height=100
)


# --------------------------------------------------
# 5. Analyze documents
# --------------------------------------------------

if st.button("Analyze Documents", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    elif len(uploaded_files) > 5:
        st.error("Please upload no more than 5 PDFs.")

    elif not question.strip():
        st.warning("Please enter a question.")

    elif not os.getenv("GEMINI_API_KEY"):
        st.error(
            "Gemini API key not found. Set GEMINI_API_KEY "
            "in PowerShell and restart Streamlit."
        )

    else:
        try:
            # Import project modules only when analysis starts
            with st.spinner("Loading AI models..."):

                from src.pdf_processor import extract_pdf
                from src.chunking import create_chunks

                from src.retrieval import (
                    build_index,
                    hybrid_search,
                    rerank_results
                )

                from src.answer_generator import generate_answer

            # Extract and chunk PDFs
            all_chunks = []

            with st.spinner("Extracting PDF text..."):

                with tempfile.TemporaryDirectory() as temp_dir:

                    for uploaded_file in uploaded_files:

                        safe_filename = Path(uploaded_file.name).name
                        pdf_path = Path(temp_dir) / safe_filename

                        pdf_path.write_bytes(uploaded_file.getvalue())

                        pages = extract_pdf(str(pdf_path))
                        document_chunks = create_chunks(pages)

                        all_chunks.extend(document_chunks)

            if not all_chunks:
                st.error("No readable text was found in the PDFs.")
                st.stop()

            st.success(
                f"Processed {len(uploaded_files)} document(s)."
            )

            st.info(f"Total text chunks: {len(all_chunks)}")

            # Build retrieval index
            with st.spinner("Building hybrid retrieval index..."):

                bm25, embeddings, chunks = build_index(all_chunks)

            # Hybrid retrieval
            with st.spinner("Searching for relevant evidence..."):

                results = hybrid_search(
                    query=question,
                    bm25=bm25,
                    embeddings=embeddings,
                    chunks=chunks,
                    top_k=10,
                    candidate_k=20
                )

            # Reranking
            with st.spinner("Reranking evidence..."):

                final_results = rerank_results(
                    question,
                    results,
                    chunks
                )

            if not final_results:
                st.warning(
                    "Insufficient evidence in the uploaded documents."
                )
                st.stop()

            # Generate answer
            with st.spinner("Generating legal answer with Gemini..."):

                answer = generate_answer(
                    question,
                    final_results[:8]
                )

            # Display answer
            st.divider()
            st.header("3. Artificially Disabled Answer")
            st.markdown(answer)

            # Display supporting evidence
            st.divider()
            st.header("4. Supporting Evidence")

            st.caption(
                "Evidence retrieved using hybrid search "
                "and reranked using a cross-encoder."
            )

            for i, result in enumerate(final_results[:5], start=1):

                title = (
                    f"Evidence {i} | "
                    f"{result['section']} | "
                    f"Page {result['page_number']}"
                )

                with st.expander(title):

                    st.write("**Document:**", result["document_name"])
                    st.write("**Section:**", result["section"])
                    st.write("**Page:**", result["page_number"])
                    st.write("**Evidence text:**")
                    st.write(result["text"])

                    st.caption(
                        f"Reranker score: "
                        f"{result['reranker_score']:.4f}"
                    )

            st.success("Document analysis completed successfully.")

        except Exception as error:
            st.error("An error occurred during document analysis.")
            st.exception(error)


# --------------------------------------------------
# 6. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Artificially Disabled ! Dumb AI trying it's best to give answers"
)
