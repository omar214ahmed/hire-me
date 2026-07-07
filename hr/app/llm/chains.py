# chains.py
from . import prompts
from schemas import ClassificationSchema, EvaluationSchema, QuestionSchema

class Chains:
    def __init__(self, llm, embeddings=None):
        # method="json_mode" everywhere: this model writes plain JSON into
        # message content rather than making real tool calls, and
        # PydanticOutputParser's verbose format_instructions caused it to
        # echo the schema back instead of filling it in. json_mode +
        # include_raw lets us log/handle failures gracefully instead of
        # throwing, same fix already applied to the question and
        # classification chains.
        self.question_chain = prompts.question_prompt | llm.with_structured_output(
            QuestionSchema, include_raw=True, method="json_mode"
        )
        self.classification_chain = prompts.classification_prompt | llm.with_structured_output(
            ClassificationSchema, method="json_mode"
        )
        self.evaluation_chain = prompts.evaluation_prompt | llm.with_structured_output(
            EvaluationSchema, include_raw=True, method="json_mode"
        )
        # Separate embedding model, used by QuestionsGenerator's
        # uniqueness guard to catch semantically-similar (not just
        # textually-identical) questions. Optional so existing callers/
        # tests that build Chains with just an llm keep working.
        self.embeddings = embeddings