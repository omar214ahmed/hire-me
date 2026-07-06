from typing import Optional
from pydantic import BaseModel
from .evaluation_schema import EvaluationSchema


class CreateSessionRequest(BaseModel):
    role: str
    skills: str


class SessionResponse(BaseModel):
    id: str
    role: str
    skills: str
    job_id: Optional[str] = None
    current_question: Optional[str] = None
    current_category: Optional[str] = None
    question_history: list[str] = []
    results_count: int = 0


class QuestionResponse(BaseModel):
    question: str
    category: str


class AnswerResponse(BaseModel):
    question: str
    category: Optional[str] = None
    transcript: str
    evaluation: EvaluationSchema


class SummaryResponse(BaseModel):
    total_questions: int
    evaluated: int
    final_score: str
    results: list[AnswerResponse] = []
