
import re


def create_chunks(pages, chunk_size=1200, overlap=200):

    chunks = []
    current_section = "General"

    article_pattern = re.compile(
        r"^\s*Article\s+(\d{1,3})\s*$",
        re.IGNORECASE | re.MULTILINE
    )

    for page in pages:

        text = page["text"]

        if not text.strip():
            continue

        matches = list(article_pattern.finditer(text))
        segments = []

        if not matches:
            segments.append((current_section, text))

        else:

            if matches[0].start() > 0:
                segments.append((
                    current_section,
                    text[:matches[0].start()]
                ))

            for i, match in enumerate(matches):

                current_section = f"Article {match.group(1)}"

                start = match.start()

                end = (
                    matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(text)
                )

                segments.append((
                    current_section,
                    text[start:end]
                ))

        for section, segment_text in segments:

            segment_text = segment_text.strip()

            if not segment_text:
                continue

            start = 0
            step = chunk_size - overlap

            while start < len(segment_text):

                end = start + chunk_size
                chunk_text = segment_text[start:end]

                chunks.append({
                    "document_name": page["document_name"],
                    "page_number": page["page_number"],
                    "section": section,
                    "text": chunk_text
                })

                start += step

    return chunks


def print_chunk_summary(chunks):

    print("\n========== CHUNK SUMMARY ==========")
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:10], start=1):

        print(f"\nChunk {i}")
        print("Document:", chunk["document_name"])
        print("Page:", chunk["page_number"])
        print("Section:", chunk["section"])
        print("Text:", chunk["text"][:200])
        print("---------------------------------")


if __name__ == "__main__":

    from pdf_processor import process_all_pdfs

    pages = process_all_pdfs()
    chunks = create_chunks(pages)

    print_chunk_summary(chunks)