"""Knowledge Graph Matching Engine (Phase 3).

Builds a directed concept graph from a rubric, extracts concepts from student
text via spaCy, then uses sentence-transformers cosine similarity to measure
how well the student's answer covers the expected knowledge graph.
"""

from typing import List

import networkx as nx
import spacy
from sentence_transformers import SentenceTransformer, util

from schemas.api_models import KGScoreResponse, RubricGraph

_SIMILARITY_THRESHOLD: float = 0.75


class GraphService:
    """Service responsible for knowledge-graph-based coverage evaluation."""

    def __init__(self) -> None:
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._nlp = spacy.load("en_core_web_sm")

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self, rubric: RubricGraph) -> nx.DiGraph:
        """Convert a RubricGraph Pydantic model into a NetworkX directed graph.

        Args:
            rubric: The validated rubric containing concept nodes and edges.

        Returns:
            A directed graph where each node carries its label and weight
            attributes, and each edge carries its relationship label.
        """
        graph = nx.DiGraph()

        for node in rubric.nodes:
            graph.add_node(node.id, label=node.label, weight=node.weight)

        for edge in rubric.edges:
            graph.add_edge(edge.source, edge.target, relationship=edge.relationship)

        return graph

    # ------------------------------------------------------------------
    # Concept extraction
    # ------------------------------------------------------------------

    def extract_student_concepts(self, text: str) -> List[str]:
        """Extract candidate concepts from student text using spaCy.

        Combines noun chunks and named entities to form a de-duplicated list
        of concept strings ready for semantic comparison.

        Args:
            text: Raw student answer text.

        Returns:
            A list of unique concept strings (lower-cased, stripped).
        """
        doc = self._nlp(text)

        concepts: list[str] = []
        seen: set[str] = set()

        for chunk in doc.noun_chunks:
            normalised = chunk.text.strip().lower()
            if normalised and normalised not in seen:
                concepts.append(normalised)
                seen.add(normalised)

        for ent in doc.ents:
            normalised = ent.text.strip().lower()
            if normalised and normalised not in seen:
                concepts.append(normalised)
                seen.add(normalised)

        return concepts

    # ------------------------------------------------------------------
    # Coverage evaluation
    # ------------------------------------------------------------------

    def evaluate_coverage(
        self,
        student_text: str,
        rubric: RubricGraph,
    ) -> KGScoreResponse:
        """Measure how well a student answer covers the rubric knowledge graph.

        Algorithm:
        1. Extract concept strings from the student text via spaCy.
        2. Build the rubric directed graph.
        3. Encode both student concepts and rubric node labels with
           sentence-transformers.
        4. For each rubric node, check whether any student concept embeds
           within *_SIMILARITY_THRESHOLD* cosine similarity.
        5. Accumulate matched weight and compute coverage as
           ``matched_weight / total_weight``.

        Args:
            student_text: The student's raw answer text.
            rubric: The structured rubric graph.

        Returns:
            A KGScoreResponse with coverage_score, matched_concepts, and
            missing_concepts.
        """
        graph = self.build_graph(rubric)
        student_concepts = self.extract_student_concepts(student_text)

        if not rubric.nodes:
            return KGScoreResponse(
                coverage_score=0.0,
                matched_concepts=[],
                missing_concepts=[],
            )

        node_labels: List[str] = [
            graph.nodes[node.id]["label"] for node in rubric.nodes
        ]

        if student_concepts:
            student_embeddings = self._encoder.encode(student_concepts, convert_to_tensor=True)
            node_embeddings = self._encoder.encode(node_labels, convert_to_tensor=True)
            # similarity matrix: shape (num_student_concepts, num_nodes)
            similarity_matrix = util.cos_sim(student_embeddings, node_embeddings)
        else:
            similarity_matrix = None

        matched_concepts: List[str] = []
        missing_concepts: List[str] = []
        matched_weight: float = 0.0
        total_weight: float = sum(node.weight for node in rubric.nodes)

        for idx, node in enumerate(rubric.nodes):
            is_matched = False
            if similarity_matrix is not None:
                # column `idx` holds similarity scores for this rubric node
                max_similarity: float = float(similarity_matrix[:, idx].max().item())
                if max_similarity >= _SIMILARITY_THRESHOLD:
                    is_matched = True

            if is_matched:
                matched_concepts.append(node.id)
                matched_weight += node.weight
            else:
                missing_concepts.append(node.id)

        coverage_score = matched_weight / total_weight if total_weight > 0.0 else 0.0

        return KGScoreResponse(
            coverage_score=round(coverage_score, 4),
            matched_concepts=matched_concepts,
            missing_concepts=missing_concepts,
        )
