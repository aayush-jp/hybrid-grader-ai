import asyncio
import json
import os
import re

from google import genai
from google.genai import types

from schemas.api_models import LLMScoreResponse

_OCR_PROMPT = (
    "You are an expert OCR system. Extract the handwritten or printed text from this "
    "image exactly as written. Preserve formatting. Do not add conversational filler."
)

_SUBJECTIVE_EVAL_PROMPT_TEMPLATE = """\
You are an expert examiner evaluating a student's answer.

Question:
{question}

Rubric / Expected Concepts:
{rubric_context}

Student Answer:
{student_text}

Evaluate the student's answer on two dimensions:
1. coherence_score  – How logically structured and readable the answer is (0.0 to 1.0).
2. correctness_score – How factually and conceptually accurate the answer is relative to the rubric (0.0 to 1.0).

Respond with STRICTLY valid JSON and nothing else – no markdown, no backticks, no explanation outside the JSON.
The JSON must match this exact schema:
{{
  "coherence_score": <float between 0.0 and 1.0>,
  "correctness_score": <float between 0.0 and 1.0>,
  "justification": "<one or two sentences explaining the scores>"
}}
"""


class GeminiService:
    """Service wrapper around the Google Gemini API for vision and language tasks."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1alpha"),
        )

    async def extract_text_from_image(self, image_bytes: bytes) -> str:
        """Extract text from an image using Gemini Vision (OCR).

        Args:
            image_bytes: Raw bytes of the image file to process.

        Returns:
            The text extracted from the image as a plain string.

        Raises:
            RuntimeError: If the Gemini API call fails or returns an empty response.
        """
        try:
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            )

            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[_OCR_PROMPT, image_part],
                ),
            )

            extracted: str = (response.text or "").strip()
            if not extracted:
                raise RuntimeError("Gemini returned an empty OCR response.")
            return extracted
        except Exception as exc:
            raise RuntimeError(f"Gemini OCR extraction failed: {exc}") from exc

    async def evaluate_subjective_quality(
        self,
        student_text: str,
        question: str,
        rubric_context: str,
    ) -> LLMScoreResponse:
        """Evaluate a student's answer for coherence and correctness using Gemini.

        The model is instructed to return strictly valid JSON matching the
        LLMScoreResponse schema. The overall LLM quality score is the average
        of ``coherence_score`` and ``correctness_score``.

        Args:
            student_text: The student's raw answer to assess.
            question: The original exam question posed to the student.
            rubric_context: A plain-text summary of the expected concepts/rubric.

        Returns:
            A populated LLMScoreResponse with coherence_score, correctness_score,
            and a natural-language justification.

        Raises:
            RuntimeError: If the Gemini API call fails, returns an empty response,
                or the response cannot be parsed as valid JSON.
        """
        prompt = _SUBJECTIVE_EVAL_PROMPT_TEMPLATE.format(
            question=question,
            rubric_context=rubric_context,
            student_text=student_text,
        )

        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt],
                ),
            )

            raw: str = (response.text or "").strip()
            if not raw:
                raise RuntimeError("Gemini returned an empty evaluation response.")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Gemini subjective evaluation failed: {exc}") from exc

        # Strip markdown code fences that Gemini sometimes wraps around JSON
        # e.g. ```json\n{...}\n``` or plain ```{...}```
        cleaned: str = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            data: dict = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini response was not valid JSON after stripping code fences. "
                f"Cleaned response: {cleaned!r}"
            ) from exc

        try:
            return LLMScoreResponse(
                coherence_score=float(data["coherence_score"]),
                correctness_score=float(data["correctness_score"]),
                justification=str(data["justification"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Gemini JSON response is missing required fields: {exc}. Data: {data}"
            ) from exc