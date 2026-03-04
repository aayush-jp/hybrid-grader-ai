import json

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from schemas.api_models import (
    FinalEvaluationResponse,
    KGScoreResponse,
    LLMScoreResponse,
    OCRResponse,
    RubricGraph,
)
from services.gemini_service import GeminiService
from services.graph_service import GraphService
from services.scoring_service import ScoringService

router = APIRouter()
_gemini_service = GeminiService()
_graph_service = GraphService()
_scoring_service = ScoringService()


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


class _EvaluateGraphRequest(RubricGraph):
    """Request body for /evaluate-graph.

    Extends RubricGraph with the student's answer text so both can be
    delivered in a single JSON payload.
    """

    student_text: str = Body(..., description="The student's raw answer text to evaluate.")


@router.post(
    "/evaluate-graph",
    response_model=KGScoreResponse,
    summary="Evaluate student answer coverage against a rubric knowledge graph",
    status_code=status.HTTP_200_OK,
)
async def evaluate_graph(payload: _EvaluateGraphRequest) -> KGScoreResponse:
    """Score how well a student answer covers a rubric expressed as a knowledge graph.

    Args:
        payload: JSON body containing ``student_text``, ``nodes``, and ``edges``.

    Returns:
        A KGScoreResponse with ``coverage_score``, ``matched_concepts``, and
        ``missing_concepts``.

    Raises:
        HTTPException 500: If the graph evaluation pipeline fails unexpectedly.
    """
    rubric = RubricGraph(nodes=payload.nodes, edges=payload.edges)
    try:
        return _graph_service.evaluate_coverage(
            student_text=payload.student_text,
            rubric=rubric,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph evaluation failed: {exc}",
        ) from exc


@router.post(
    "/evaluate-full",
    response_model=FinalEvaluationResponse,
    summary="Full hybrid evaluation: OCR → KG matching → LLM scoring → final grade",
    status_code=status.HTTP_200_OK,
)
async def evaluate_full(
    file: UploadFile = File(..., description="Answer-sheet image (JPEG, PNG, etc.)."),
    question: str = Form(..., description="The exam question the student is answering."),
    rubric_json: str = Form(..., description="Stringified JSON of a RubricGraph object."),
    alpha: float = Form(0.5, description="KG weight in hybrid score formula (0.0 – 1.0)."),
) -> FinalEvaluationResponse:
    """Run the full hybrid evaluation pipeline on a student answer-sheet image.

    Pipeline steps:
    1. Parse ``rubric_json`` into a RubricGraph.
    2. Extract student text from the uploaded image via Gemini Vision (OCR).
    3. Score KG coverage with GraphService.
    4. Score subjective quality (coherence + correctness) with GeminiService.
    5. Combine scores: Final Score = α × KG Score + (1 − α) × LLM Score.
    6. Return a FinalEvaluationResponse with all intermediate and final results.

    Args:
        file: The uploaded answer-sheet image.
        question: The exam question being assessed.
        rubric_json: A JSON string representing the rubric as a RubricGraph.
        alpha: Blending weight for the KG score (default 0.5).

    Returns:
        FinalEvaluationResponse containing OCR text, KG result, LLM result,
        and the hybrid final score.

    Raises:
        HTTPException 400: If ``rubric_json`` cannot be parsed or is invalid.
        HTTPException 422: If the uploaded file cannot be read.
        HTTPException 500: If any pipeline stage fails.
    """
    # --- Step 1: Parse rubric ---
    try:
        rubric_dict: dict = json.loads(rubric_json)
        rubric = RubricGraph(**rubric_dict)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"rubric_json is not valid JSON: {exc}",
        ) from exc
    except (ValidationError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"rubric_json does not match the RubricGraph schema: {exc}",
        ) from exc

    # --- Step 2: OCR ---
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

    # --- Step 3: KG coverage ---
    try:
        kg_result: KGScoreResponse = _graph_service.evaluate_coverage(
            student_text=extracted_text,
            rubric=rubric,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge-graph evaluation failed: {exc}",
        ) from exc

    # --- Step 4: LLM subjective quality ---
    rubric_context = ", ".join(node.label for node in rubric.nodes)
    try:
        llm_result: LLMScoreResponse = await _gemini_service.evaluate_subjective_quality(
            student_text=extracted_text,
            question=question,
            rubric_context=rubric_context,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # --- Step 5: Hybrid score ---
    llm_score = (llm_result.coherence_score + llm_result.correctness_score) / 2.0
    try:
        final_score: float = _scoring_service.calculate_hybrid_score(
            kg_score=kg_result.coverage_score,
            llm_score=llm_score,
            alpha=alpha,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # --- Step 6: Return composite result ---
    return FinalEvaluationResponse(
        extracted_text=extracted_text,
        kg_result=kg_result,
        llm_result=llm_result,
        final_score=final_score,
    )
