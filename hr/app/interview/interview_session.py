import time
import logging
from llm.transcript import Transcript
from llm import QuestionsGenerator, Evaluator, ClassificationQuestion

logger = logging.getLogger("hr_interview.session")

class InterviewSession:
    def __init__(
        self,
        transcript: Transcript,
        generator: QuestionsGenerator,
        classifier: ClassificationQuestion,
        evaluator: Evaluator
    ):
        self.transcript = transcript
        self.generator = generator
        self.classifier = classifier
        self.evaluator = evaluator

        self.current_question = None
        self.current_category = None
        self.question_history: list[str] = []

    # =========================
    # 1. Generate Question
    # =========================
    def generate_question(self, role: str, skills: str) -> str:
        start = time.perf_counter()
        logger.info("generate_question started | role='%s' | history_size=%d", role, len(self.question_history))

        result = self.generator.generate(
            role=role,
            description=skills,
            previous_questions=self.question_history
        )

        elapsed = time.perf_counter() - start
        threshold_warn = 120.0
        if elapsed >= threshold_warn:
            logger.warning(
                "SLOW | generate_question took %.2f seconds (>= %.0f s) | role='%s'",
                elapsed, threshold_warn, role,
            )
        else:
            logger.info(
                "generate_question completed in %.2f seconds | role='%s'", elapsed, role,
            )

        self.current_question = result
        self.question_history.append(result)
        return result
    
    def classify_current_question(self):
        start = time.perf_counter()
        logger.info("classify_current_question started")

        result = self.classifier.classify(self.current_question)

        elapsed = time.perf_counter() - start
        logger.info("classify_current_question completed in %.2f seconds | category='%s'", elapsed, result)

        self.current_category = result
        return result


        

    def evaluate_answer(self, audio_bytes: bytes, suffix: str = ".wav") -> dict:

        if not self.current_question:
            return {
                "status": "error",
                "message": "No active question"
            }

        try:
            transcript = self.transcript.transcribe(audio_bytes, suffix)
        except Exception as e:
            return {
                "question": self.current_question,
                "category": self.current_category,
                "transcript": "",
                "evaluation": {
                    "status": "error",
                    "message": f"Audio transcription failed: {str(e)}"
                }
            }

        if not transcript or len(transcript.split()) < 3:
            return {
                "question": self.current_question,
                "category": self.current_category,
                "transcript": transcript,
                "evaluation": {
                    "status": "invalid_answer",
                    "message": "Answer too short"
                }
            }

        evaluation = self.evaluator.evaluate_answer(
            question=self.current_question,
            category=self.current_category,
            answer=transcript
            )

        question_snapshot = self.current_question
        category_snapshot = self.current_category
        self.current_question = None
        self.current_category = None

        return {
            "question": question_snapshot,
            "category": category_snapshot,
            "transcript": transcript,
            "evaluation": evaluation.model_dump()
        }
    


    def finish(self, results: list) -> dict:
        valid_scores = [
            r["evaluation"]["score"]
            for r in results
            if r.get("evaluation", {}).get("score") is not None
        ]

        if not valid_scores:
            return {
                "total_questions": len(results),
                "evaluated": 0,
                "final_score": "0/10",
                "results": results
            }

        final_score = round(sum(valid_scores) / len(valid_scores), 2)

        return {
            "total_questions": len(results),
            "evaluated": len(valid_scores),
            "final_score": f"{final_score}/10",
            "results": results
        }
    


