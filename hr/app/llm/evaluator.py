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

        if result is None:
            return EvaluationSchema(score=0, feedback="No result from evaluator")

        if result.score is None:
            logger.warning("LLM returned evaluation with score=None")
            return EvaluationSchema(score=0, feedback="Incomplete evaluation returned")

        return EvaluationSchema(
            score=result.score,
            feedback=result.feedback
        )




        