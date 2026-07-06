# chains.py
from . import prompts
from schemas import ClassificationSchema, EvaluationSchema, QuestionSchema


class Chains:
    def __init__(self, llm):
        self.question_chain = prompts.question_prompt | llm.with_structured_output(QuestionSchema)
        self.classification_chain = prompts.classification_prompt | llm.with_structured_output(ClassificationSchema)
        self.evaluation_chain = prompts.evaluation_prompt | llm.with_structured_output(EvaluationSchema)
