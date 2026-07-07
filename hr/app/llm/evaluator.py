import logging
from schemas import EvaluationSchema
from .chains import Chains

logger = logging.getLogger("hr_interview.evaluator")

class Evaluator:
    def __init__(self, chains: Chains):
        self.evaluator = chains.evaluation_chain

    def evaluate_answer(self, question: str, answer: str, category: str) -> EvaluationSchema:
        try:
            result = self.evaluator.invoke({
                "question": question,
                "answer": answer,
                "category": category
            })
        except Exception as e:
            logger.error("LLM evaluation chain failed: %s", e)
            return EvaluationSchema(score=0, feedback="Evaluation chain error")

        # include_raw=True -> result is {"raw": ..., "parsed": ..., "parsing_error": ...},
        # not the parsed EvaluationSchema directly.
        parsed = result.get("parsed") if isinstance(result, dict) else result
        raw = result.get("raw") if isinstance(result, dict) else None
        parsing_error = result.get("parsing_error") if isinstance(result, dict) else None

        if parsed is None:
            raw_content = getattr(raw, "content", None)
            logger.warning(
                "LLM failed to produce a parseable evaluation | parsing_error=%r | raw_content=%r",
                parsing_error, raw_content,
            )
            return EvaluationSchema(score=0, feedback="Incomplete evaluation returned")

        if parsed.score is None:
            logger.warning("LLM returned evaluation with score=None")
            return EvaluationSchema(score=0, feedback="Incomplete evaluation returned")

        return EvaluationSchema(
            score=parsed.score,
            feedback=parsed.feedback
        )