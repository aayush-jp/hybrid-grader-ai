"""Hybrid scoring layer combining Knowledge Graph and LLM evaluation signals."""


class ScoringService:
    """Combines KG coverage and LLM quality scores into a single hybrid score.

    Implements the weighted formula from the paper:

        Final Score = α × KG Score + (1 − α) × LLM Score

    where ``alpha`` controls the balance between structural coverage (KG) and
    subjective quality (LLM). A value of ``0.5`` weights both signals equally.
    """

    def calculate_hybrid_score(
        self,
        kg_score: float,
        llm_score: float,
        alpha: float = 0.5,
    ) -> float:
        """Compute the weighted hybrid score from KG and LLM sub-scores.

        Args:
            kg_score: Knowledge-graph coverage score in the range [0.0, 1.0].
            llm_score: LLM-derived quality score in the range [0.0, 1.0],
                typically the average of coherence and correctness.
            alpha: Weighting factor for the KG score (default 0.5).
                Must be in [0.0, 1.0]. The LLM score receives weight (1 − alpha).

        Returns:
            The hybrid final score in [0.0, 1.0], rounded to four decimal places.

        Raises:
            ValueError: If ``alpha`` is outside [0.0, 1.0].
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}.")

        hybrid = alpha * kg_score + (1.0 - alpha) * llm_score
        return round(hybrid, 4)
