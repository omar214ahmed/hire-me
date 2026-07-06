import time
import random
import logging
from .chains import Chains

logger = logging.getLogger("hr_interview.question_generator")

# Rotated through on every call so the model can't settle on the same
# "highest probability" question for a given role/skills combo, even
# when previous_questions is empty (first question of a session) or
# temperature is low.
VARIETY_HINTS = [
    "core concepts and definitions",
    "trade-offs between two common approaches",
    "a debugging or troubleshooting scenario",
    "real-world design decision",
    "performance or scalability considerations",
    "common pitfalls or misconceptions",
    "how it compares to an alternative technology",
    "a scenario involving failure or edge cases",
]

class QuestionsGenerator:
    def __init__(self, chains: Chains):
        self.question_generator = chains.question_chain

    def generate(self, role: str, description: str, previous_questions: list[str] | None = None) -> str:
        prev_qs = previous_questions or []
        prev_text = "\n".join(f"- {q}" for q in prev_qs) if prev_qs else "None"
        variety_hint = random.choice(VARIETY_HINTS)

        logger.info(
            "Generating question for role='%s' | description='%s' | previous_questions=%d | variety_hint='%s'",
            role, description, len(prev_qs), variety_hint,
        )
        start = time.perf_counter()

        result = self.question_generator.invoke({
            "role": role,
            "description": description,
            "previous_questions": prev_text,
            "variety_hint": variety_hint,
        })

        elapsed = time.perf_counter() - start
        threshold_warn = 120.0
        if elapsed >= threshold_warn:
            logger.warning(
                "SLOW | question_generator.invoke took %.2f seconds (>= %.0f s) | role='%s'",
                elapsed, threshold_warn, role,
            )
        else:
            logger.info(
                "question_generator.invoke completed in %.2f seconds | role='%s'",
                elapsed, role,
            )

        # With include_raw=True, `result` is now a dict:
        # {"raw": <the raw AIMessage>, "parsed": <QuestionSchema | None>, "parsing_error": <Exception | None>}
        parsed = result.get("parsed") if isinstance(result, dict) else result
        raw = result.get("raw") if isinstance(result, dict) else None
        parsing_error = result.get("parsing_error") if isinstance(result, dict) else None

        if parsed is None:
            raw_content = getattr(raw, "content", None)
            raw_tool_calls = getattr(raw, "tool_calls", None)
            logger.warning(
                "LLM failed to produce a parseable question | role='%s' | parsing_error=%r | raw_content=%r | raw_tool_calls=%r",
                role, parsing_error, raw_content, raw_tool_calls,
            )
            return f"As a {role} candidate, tell me about your experience with {description}."

        return parsed.question