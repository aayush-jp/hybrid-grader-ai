from typing import List

from pydantic import BaseModel, Field


class OCRResponse(BaseModel):
    extracted_text: str = Field(
        ...,
        description="The raw text extracted from the uploaded image via OCR.",
    )


class ConceptNode(BaseModel):
    id: str = Field(..., description="Unique identifier for this concept node.")
    label: str = Field(..., description="Human-readable label for this concept.")
    weight: float = Field(1.0, description="Importance weight of this concept in the rubric.")


class ConceptEdge(BaseModel):
    source: str = Field(..., description="ID of the source concept node.")
    target: str = Field(..., description="ID of the target concept node.")
    relationship: str = Field(..., description="Label describing the relationship between the two nodes.")


class RubricGraph(BaseModel):
    nodes: List[ConceptNode] = Field(..., description="All concept nodes in the rubric knowledge graph.")
    edges: List[ConceptEdge] = Field(..., description="Directed edges encoding relationships between concepts.")


class KGScoreResponse(BaseModel):
    coverage_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of total rubric weight covered by the student's answer (0.0 – 1.0).",
    )
    matched_concepts: List[str] = Field(..., description="Rubric concept node IDs matched in the student text.")
    missing_concepts: List[str] = Field(..., description="Rubric concept node IDs absent from the student text.")


class LLMScoreResponse(BaseModel):
    coherence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How logically structured and readable the answer is (0.0 – 1.0).",
    )
    correctness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How factually and conceptually accurate the answer is (0.0 – 1.0).",
    )
    justification: str = Field(
        ...,
        description="Natural-language explanation of the scores given by the LLM.",
    )


class FinalEvaluationResponse(BaseModel):
    extracted_text: str = Field(..., description="Text extracted from the answer-sheet image via OCR.")
    kg_result: KGScoreResponse = Field(..., description="Knowledge-graph coverage result.")
    llm_result: LLMScoreResponse = Field(..., description="LLM coherence and correctness result.")
    final_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Hybrid final score: α × KG Score + (1 − α) × LLM Score.",
    )
