from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    description: str = Field(..., min_length=20, description="Raw job description text")


class JobResponse(BaseModel):
    job_id: str
    extracted: Dict[str, List[str]]
    query: str
    created_at: str


class JobHRView(BaseModel):
    """
    Minimal, stable shape exposed to other services (currently the HR
    system). Deliberately excludes internal fields such as jd_text,
    jd_embedding, query, extracted's full label set, etc. — anything the
    HR service doesn't need to build an interview prompt. Keeping this as
    its own model (rather than reusing JobResponse) means the ATS's
    internal `extracted` schema can evolve freely without breaking the
    HR integration's request/response contract.
    """
    id: str
    job_title: Optional[str] = None
    hard_skills: List[str] = Field(default_factory=list)


class MatchRequest(BaseModel):
    candidate_ids: Optional[List[str]] = Field(
        default=None,
        description="Match against only these candidate ids, skipping the "
                     "DB-wide cosine shortlist step. Omit to shortlist "
                     "against every stored candidate.",
    )
    shortlist_limit: int = Field(
        default=150, ge=1, le=10000,
        description="How many candidates the cosine-similarity step pulls "
                     "from the database (highest score first) before "
                     "reranking. Ignored if candidate_ids is set.",
    )
    top_n: int = Field(
        default=20, ge=1, le=10000,
        description="How many final, cross-encoder-reranked results to return.",
    )


class ShortlistItem(BaseModel):
    candidate_id: str
    filename: str
    semantic_score: float


class ShortlistResponse(BaseModel):
    job_id: str
    count: int
    shortlist: List[ShortlistItem]


class CandidateScore(BaseModel):
    cv_id: str
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None
    semantic_score: float
    hard_match: Dict


class MatchResponse(BaseModel):
    job_id: str
    shortlisted: int
    results: List[CandidateScore]
