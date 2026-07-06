# chains.py
from . import prompts
from schemas import ClassificationSchema, EvaluationSchema, QuestionSchema


class Chains:
    def __init__(self, llm):
        self.question_chain = prompts.question_prompt | llm.with_structured_output(QuestionSchema)
        self.classification_chain = prompts.classification_prompt | llm.with_structured_output(ClassificationSchema)
        # Was previously `prompts.evaluation_prompt | llm | prompts.evaluation_parser`
        # (a PydanticOutputParser). That approach pastes the raw JSON Schema
        # into the prompt as "format instructions", and small local models
        # (e.g. qwen2.5:3b via Ollama) sometimes echo that schema back
        # verbatim instead of filling it in, which fails Pydantic validation
        # ("Field required" for every field). with_structured_output instead
        # gets Ollama to constrain generation to the schema directly, the
        # same mechanism already used successfully for the other two chains.
        self.evaluation_chain = prompts.evaluation_prompt | llm.with_structured_output(EvaluationSchema)