import os
import asyncio
from google import genai
from google.genai import types

_OCR_PROMPT = (
    "You are an expert OCR system. Extract the handwritten or printed text from this "
    "image exactly as written. Preserve formatting. Do not add conversational filler."
)

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

            extracted: str = response.text.strip()
            if not extracted:
                raise RuntimeError("Gemini returned an empty OCR response.")
            return extracted
        except Exception as exc:
            raise RuntimeError(f"Gemini OCR extraction failed: {exc}") from exc