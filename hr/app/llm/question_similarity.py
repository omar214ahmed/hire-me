import logging
import math

logger = logging.getLogger("hr_interview.question_similarity")


class QuestionUniquenessGuard:
    """
    Semantic duplicate guard for a single interview session.

    Exact-string / post-hoc dedup only catches questions that are worded
    (almost) identically. An LLM asked "don't repeat yourself" will happily
    reword the same question ("What is a race condition?" vs "Can you
    explain what causes race conditions?") — those are duplicates from the
    candidate's point of view, but a string/set comparison never notices.

    This guard embeds every accepted question and rejects new candidates
    that are too close, by cosine similarity, to anything already asked —
    catching meaning-level repeats regardless of phrasing.
    """

    def __init__(self, embeddings, similarity_threshold: float = 0.86):
        if embeddings is None:
            raise ValueError(
                "QuestionUniquenessGuard requires an embeddings client "
                "(Chains.embeddings was None)."
            )
        self._embeddings = embeddings
        self._threshold = similarity_threshold
        self._seen_vectors: list[list[float]] = []

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def check_and_register(self, question: str) -> tuple[bool, float]:
        """
        Embeds `question` and compares it against every question already
        accepted in this session.

        Returns (True, best_similarity) and registers the question if it's
        sufficiently different from everything seen so far.
        Returns (False, best_similarity) without registering it if it's a
        near-duplicate of an existing question, so the caller can retry
        with a different skill/angle.
        """
        try:
            vector = self._embeddings.embed_query(question)
        except Exception as e:
            # If the embedding backend is unavailable, fail open rather than
            # blocking question generation entirely — better an occasional
            # duplicate than a broken interview.
            logger.error("Embedding call failed, allowing question through: %s", e)
            return True, 0.0

        best_similarity = 0.0
        for existing_vector in self._seen_vectors:
            similarity = self._cosine_similarity(vector, existing_vector)
            best_similarity = max(best_similarity, similarity)
            if similarity >= self._threshold:
                return False, best_similarity

        self._seen_vectors.append(vector)
        return True, best_similarity

    def register(self, question: str) -> None:
        """Force-register a question (used for the last-resort fallback
        candidate) without re-running the rejection check."""
        try:
            vector = self._embeddings.embed_query(question)
            self._seen_vectors.append(vector)
        except Exception as e:
            logger.error("Embedding call failed while force-registering: %s", e)
