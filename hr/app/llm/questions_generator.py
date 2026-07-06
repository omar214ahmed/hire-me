import time
import logging
from .chains import Chains

logger = logging.getLogger("hr_interview.question_generator")

class QuestionsGenerator:
    def __init__(self, chains: Chains):
        self.question_generator = chains.question_chain

    def generate(self, role: str, description: str, previous_questions: list[str] | None = None) -> str:
        prev_qs = previous_questions or []
        prev_text = "\n".join(f"- {q}" for q in prev_qs) if prev_qs else "None"
        logger.info("Previous questions fed to prompt:\n%s", prev_text)

        logger.info(
            "Generating question for role='%s' | description='%s' | previous_questions=%d",
            role, description, len(prev_qs),
        )
        start = time.perf_counter()

        result = self.question_generator.invoke({
            "role": role,
            "description": description,
            "previous_questions": prev_text
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

        if result is None:
            logger.warning("LLM returned None; using fallback question for role='%s'", role)
            return f"As a {role} candidate, tell me about your experience with {description}."

        new_question = result.question.strip()
        prev_lower = [q.strip().lower() for q in prev_qs]
        if new_question.lower() in prev_lower:
            logger.warning(
                "LLM returned a duplicate question for role='%s'; using fallback",
                role,
            )
            return f"As a {role} candidate, tell me about your experience with {description}."

        return new_question