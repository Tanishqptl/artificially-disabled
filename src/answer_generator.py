
import os
import time

from google import genai
from google.genai import errors


# --------------------------------------------------
# 1. Connect to Gemini
# --------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.8-flash"


# --------------------------------------------------
# 2. Generate legal answers
# --------------------------------------------------

def generate_answer(question, evidence, return_usage=False):

    # Handle missing evidence
    if not evidence:
        answer = "Insufficient evidence in the uploaded documents."

        if return_usage:
            return answer, {}

        return answer

    # Prepare retrieved evidence
    context = ""

    for i, item in enumerate(evidence, 1):
        context += f"""
Evidence {i}
Document: {item['document_name']}
Page: {item['page_number']}
Section: {item['section']}
Text: {item['text']}
"""

    # Build evidence-grounded legal prompt
    prompt = f"""
You are Artificially Disabled, a legal document analysis assistant.

Answer using ONLY the supplied evidence.

Rules:
1. Do not invent legal information.
2. Cite every important claim.
3. Include document name, page number and section.
4. If evidence is insufficient, say:
   "Insufficient evidence in the uploaded documents."
5. Do not provide unsupported legal conclusions.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Provide a clear, structured answer with citations.
"""

    # --------------------------------------------------
    # 3. Gemini API with retry mechanism
    # --------------------------------------------------

    for attempt in range(3):

        try:
            start_time = time.perf_counter()

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            answer_time = time.perf_counter() - start_time

            answer = response.text or "No answer was generated."

            # --------------------------------------------------
            # 4. Capture token usage for Feature 11
            # --------------------------------------------------

            if return_usage:

                metadata = response.usage_metadata

                usage = {
                    "model": MODEL_NAME,
                    "prompt_tokens": getattr(
                        metadata, "prompt_token_count", 0
                    ) or 0,

                    "answer_tokens": getattr(
                        metadata, "candidates_token_count", 0
                    ) or 0,

                    "total_tokens": getattr(
                        metadata, "total_token_count", 0
                    ) or 0,

                    "answer_generation_seconds": round(
                        answer_time, 4
                    )
                }

                return answer, usage

            return answer

        except errors.ServerError:

            print(
                f"Gemini server error. Attempt {attempt + 1}/3"
            )

            if attempt == 2:

                answer = (
                    "Gemini is temporarily unavailable. "
                    "Please try again later."
                )

                if return_usage:
                    return answer, {}

                return answer

            time.sleep(5 * (attempt + 1))

        except Exception as error:

            answer = f"Gemini error: {error}"

            if return_usage:
                return answer, {}

            return answer


# --------------------------------------------------
# 5. Test answer generator and token tracking
# --------------------------------------------------

if __name__ == "__main__":

    sample_evidence = [{
        "document_name": "GDPR.pdf",
        "page_number": 1,
        "section": "Article 83",
        "text": (
            "Administrative fines shall be imposed "
            "in accordance with Article 83."
        )
    }]

    question = "What does Article 83 discuss?"

    answer, usage = generate_answer(
        question,
        sample_evidence,
        return_usage=True
    )

    print("\n========== ARTIFICIALLY DISABLED ANSWER ==========")
    print(answer)

    print("\n========== API USAGE REPORT ==========")
    print("Model:", usage.get("model"))
    print("Prompt tokens:", usage.get("prompt_tokens"))
    print("Answer tokens:", usage.get("answer_tokens"))
    print("Total tokens:", usage.get("total_tokens"))
    print(
        "Answer generation time:",
        usage.get("answer_generation_seconds"),
        "seconds"
    )