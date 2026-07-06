from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from schemas import EvaluationSchema #despi

question_prompt = ChatPromptTemplate.from_template("""
You are a senior technical interviewer.  

Your task is to generate ONE high-quality interview question.

Rules:
- The question must match the given role and skills ONLY
- Do NOT add explanations
- Do NOT include multiple questions
- Avoid generic questions
- Avoid trivia or vague questions
- The question must be realistic in a real interview setting
- Difficulty must be medium by default
- Do NOT repeat or generate a question similar to any of the previous questions
- Ask about concepts and understanding only, NOT about writing actual implementation code

Role:
{role}

Skills:
{description}

Previous questions (DO NOT repeat these):
{previous_questions}

Return ONLY valid JSON:

{{
  "question": "..."
}}
""")


classification_prompt = ChatPromptTemplate.from_template("""
You are an interview question classifier.

Classify the question into exactly ONE category.

Allowed categories:
- technical
- problem_solving
- behavioral

Rules:
- Return ONLY one category string
- No explanation
- No extra text

Question:
{question}

Return ONLY valid JSON with one of these exact values:
{{"category": "technical"}}
{{"category": "problem_solving"}}
{{"category": "behavioral"}}
""")


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas import EvaluationSchema

template_message = """
You are an expert technical interviewer.

Evaluate the candidate answer carefully.

Question:
{question}

Answer:
{answer}

Category:
{category}

Scoring Rubric (0-10):
- 0-2: Wrong, irrelevant, or no meaningful answer.
- 3-4: Very limited understanding, major mistakes, or vague with no examples.
- 5-6: Partially correct but misses key concepts or lacks depth.
- 7-8: Mostly correct with minor gaps, or good example with clear reasoning.
- 9-10: Accurate, complete, strong understanding, or structured answer with measurable impact.

Evaluation Steps:
1. Check if the answer addresses the question.
2. Check technical correctness and completeness.
3. Check clarity and communication.
4. Assign a final score from 0 to 10.

Important Rules:
- Never give high scores for vague answers.
- If the answer is unrelated, score <= 2.
- If the answer is partially correct, score between 4 and 6.
- Explain briefly why the score was assigned in the "feedback" field.
- Only set "status" and "message" if the answer is empty, nonsensical, or cannot be evaluated. Otherwise leave them null.

Output format:
{format_instructions}
"""

evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationSchema)

evaluation_prompt = PromptTemplate(
    template=template_message,
    input_variables=["question", "answer", "category"],
    partial_variables={"format_instructions": evaluation_parser.get_format_instructions()},
)



