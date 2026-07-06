# chains.py
from . import prompts
from schemas import EvaluationSchema

class Chains:
    def __init__(self, llm):
        self.question_chain = prompts.question_prompt | llm | prompts.question_parser
        self.classification_chain = prompts.classification_prompt | llm | prompts.classification_parser
        self.evaluation_chain = prompts.evaluation_prompt | llm | prompts.evaluation_parser
