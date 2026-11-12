
import fitz
from pathlib import Path


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# PDF storage directory
PDF_FOLDER = BASE_DIR / "data" / "legal_pdfs"


def extract_pdf(pdf_path):

    document = fitz.open(pdf_path)
    extracted_pages = []

    try:
        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text", sort=True).strip()

            page_data = {
                "document_name": Path(pdf_path).name,
                "page_number": page_number,
                "text": text,
                "is_scanned": len(text) < 40,
                "character_count": len(text)
            }

            extracted_pages.append(page_data)

    finally:
        document.close()

    return extracted_pages


def process_all_pdfs():

    PDF_FOLDER.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in:", PDF_FOLDER)
        return []

    all_pages = []

    print("\n========== ARTIFICIALLY DISABLED PDF PROCESSOR ==========")
    print("PDF files found:", len(pdf_files))

    for pdf_path in pdf_files:

        print("\nProcessing:", pdf_path.name)

        try:
            pages = extract_pdf(pdf_path)
            all_pages.extend(pages)

            print("Pages extracted:", len(pages))

            print(
                "Potentially scanned pages:",
                sum(page["is_scanned"] for page in pages)
            )

            for page in pages:
                print(
                    f"Page {page['page_number']}: "
                    f"{page['character_count']} characters"
                )

        except Exception as error:
            print("Could not process:", pdf_path.name)
            print("Reason:", error)

    print("\n========== PROCESSING COMPLETE ==========")
    print("Total pages extracted:", len(all_pages))

    return all_pages


if __name__ == "__main__":

    all_pages = process_all_pdfs()