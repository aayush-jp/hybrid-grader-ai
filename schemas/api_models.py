from pydantic import BaseModel, Field


class OCRResponse(BaseModel):
    extracted_text: str = Field(
        ...,
        description="The raw text extracted from the uploaded image via OCR.",
    )
