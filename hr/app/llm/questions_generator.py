import logging
from .chains import Chains
from .skill_planner import SkillCoveragePlanner
from .question_similarity import QuestionUniquenessGuard

logger = logging.getLogger("hr_interview.questions_generator")

# Only the last few questions are shown back to the LLM as style/phrasing
# context. Real duplicate prevention is enforced structurally (the skill
# planner) and semantically (the uniqueness guard) below — NOT by hoping an
# ever-growing "previous_questions" blob keeps fitting in the model's
# context window as the interview goes on.
_RECENT_HISTORY_WINDOW = 5

_FALLBACK_QUESTION = "Tell me about a challenging technical problem you solved recently and how you approached it."


class QuestionsGenerator:
    """
    Generates interview questions that are unique and diverse *by
    construction*, rather than generating freely and cleaning up
    duplicates afterward.

    Two mechanisms work together, each covering a failure mode the other
    doesn't:

    1. SkillCoveragePlanner — deterministically points each question at a
       different (skill, angle) combination from the job's skill list, so
       consecutive questions are structurally unlikely to overlap. This
       also finally makes {variety_hint} in the prompt meaningful (it used
       to be referenced by the template but never supplied).

    2. QuestionUniquenessGuard — embeds each accepted question and rejects
       new candidates that are semantically too close to one already
       asked, even if worded differently. This catches the case a planner
       alone can't: the LLM rephrasing the same underlying question.

    One QuestionsGenerator instance is created per interview session (see
    routers/sessions.py), so the planner/guard state naturally persists for
    the lifetime of that interview and resets for the next one.
    """

    def __init__(
        self,
        chains: Chains,
        similarity_threshold: float | None = None,
        max_attempts: int = 3,
    ):
        self._chain = chains.question_chain
        self._embeddings = chains.embeddings
        self._max_attempts = max(1, max_attempts)
        self._similarity_threshold = similarity_threshold if similarity_threshold is not None else 0.86

        # Built lazily, once we know the skills for this session. Kept
        # keyed on the skills string so a generator instance would still
        # behave correctly if it were ever reused across two different
        # role/skills combinations.
        self._planner: SkillCoveragePlanner | None = None
        self._guard: QuestionUniquenessGuard | None = None
        self._state_key: str | None = None

    def _ensure_state(self, skills: str) -> None:
        if self._planner is not None and self._state_key == skills:
            return
        self._planner = SkillCoveragePlanner(skills)
        self._guard = QuestionUniquenessGuard(
            self._embeddings, similarity_threshold=self._similarity_threshold
        )
        self._state_key = skills

    def generate(self, role: str, description: str, previous_questions: list[str] | None = None) -> str:
        """
        Generates the next interview question for `role`, focused on
        `description` (the job's skills), guaranteed unique against every
        question already returned by this generator instance.
        """
        self._ensure_state(description)
        previous_questions = previous_questions or []
        recent_history = previous_questions[-_RECENT_HISTORY_WINDOW:]
        recent_history_text = "\n".join(f"- {q}" for q in recent_history) or "None yet."

        best_candidate: str | None = None
        best_candidate_similarity = 1.0

        for attempt in range(1, self._max_attempts + 1):
            skill, angle = self._planner.next()

            try:
                result = self._chain.invoke({
                    "role": role,
                    "description": skill,
                    "variety_hint": angle,
                    "previous_questions": recent_history_text,
                })
            except Exception as e:
                logger.error("question_chain failed on attempt %d/%d: %s", attempt, self._max_attempts, e)
                continue

            # include_raw=True -> result is {"raw": ..., "parsed": ..., "parsing_error": ...}
            parsed = result.get("parsed") if isinstance(result, dict) else result
            raw = result.get("raw") if isinstance(result, dict) else None
            parsing_error = result.get("parsing_error") if isinstance(result, dict) else None

            if parsed is None:
                raw_content = getattr(raw, "content", None)
                logger.warning(
                    "LLM failed to produce a parseable question | attempt=%d | skill=%r | "
                    "parsing_error=%r | raw_content=%r",
                    attempt, skill, parsing_error, raw_content,
                )
                continue

            question = (parsed.question or "").strip()
            if not question:
                logger.warning("LLM returned an empty question on attempt %d | skill=%r", attempt, skill)
                continue

            is_unique, similarity = self._guard.check_and_register(question)
            if is_unique:
                logger.info(
                    "Accepted question | attempt=%d | skill=%r | angle=%r | max_similarity=%.3f",
                    attempt, skill, angle, similarity,
                )
                return question

            logger.info(
                "Rejected near-duplicate question | attempt=%d | skill=%r | similarity=%.3f (>= %.2f threshold)",
                attempt, skill, similarity, self._similarity_threshold,
            )
            if similarity < best_candidate_similarity:
                best_candidate, best_candidate_similarity = question, similarity

        # Every attempt either failed or was rejected as a near-duplicate.
        # Fall back to the least-similar candidate we saw rather than
        # silently returning nothing, and register it so the next call
        # still treats it as asked.
        if best_candidate is not None:
            logger.warning(
                "Exhausted %d attempts without a fully unique question; "
                "accepting least-similar candidate (similarity=%.3f)",
                self._max_attempts, best_candidate_similarity,
            )
            self._guard.register(best_candidate)
            return best_candidate

        logger.error("All %d generation attempts failed outright; returning fallback question", self._max_attempts)
        self._guard.register(_FALLBACK_QUESTION)
        return _FALLBACK_QUESTION
