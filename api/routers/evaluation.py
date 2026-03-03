from fastapi import APIRouter, HTTPException, UploadFile, status

from schemas.api_models import OCRResponse
from services.gemini_service import GeminiService

router = APIRouter()
_gemini_service = GeminiService()


@router.post(
    "/extract-text",
    response_model=OCRResponse,
    summary="Extract text from an uploaded answer-sheet image",
    status_code=status.HTTP_200_OK,
)
async def extract_text(file: UploadFile) -> OCRResponse:
    """Accept an image upload and return OCR-extracted text via Gemini Vision.

    Args:
        file: The image file uploaded by the client (JPEG, PNG, etc.).

    Returns:
        An OCRResponse containing the extracted text.

    Raises:
        HTTPException 422: If the uploaded file cannot be read.
        HTTPException 500: If the Gemini OCR call fails.
    """
    try:
        image_bytes: bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc

    try:
        extracted_text: str = await _gemini_service.extract_text_from_image(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return OCRResponse(extracted_text=extracted_text)
