from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    description: str = Field(..., min_length=20, description="Raw job description text")


class ExtractedField(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0


class ExtractedListField(BaseModel):
    value: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class JDExtraction(BaseModel):
    job_title: ExtractedField = Field(default_factory=ExtractedField)
    years_experience: ExtractedField = Field(default_factory=ExtractedField)
    hard_skills: ExtractedListField = Field(default_factory=ExtractedListField)
    soft_skills: ExtractedListField = Field(default_factory=ExtractedListField)
    nice_to_have_skills: ExtractedListField = Field(default_factory=ExtractedListField)
    education_degree: ExtractedField = Field(default_factory=ExtractedField)
    field_of_study: ExtractedField = Field(default_factory=ExtractedField)
    languages: ExtractedListField = Field(default_factory=ExtractedListField)
    work_location: ExtractedField = Field(default_factory=ExtractedField)
    job_type: ExtractedField = Field(default_factory=ExtractedField)
    benefits: ExtractedListField = Field(default_factory=ExtractedListField)

    extraction_method: str = "unknown"
    needs_review: bool = False
    routing_reason: Optional[str] = None
    schema_version: int = 1

    def low_confidence_field_count(self, threshold: float) -> int:
        count = 0
        for field_name in (
            "job_title", "years_experience", "education_degree",
            "field_of_study", "work_location", "job_type",
        ):
            field: ExtractedField = getattr(self, field_name)
            if field.value and field.confidence < threshold:
                count += 1
        for field_name in ("hard_skills", "soft_skills", "nice_to_have_skills", "languages", "benefits"):
            list_field: ExtractedListField = getattr(self, field_name)
            if list_field.value and list_field.confidence < threshold:
                count += 1
        return count

    def to_legacy_dict(self) -> Dict[str, List[str]]:
        def _one(field: ExtractedField) -> List[str]:
            return [field.value] if field.value else []

        return {
            "required job title": _one(self.job_title),
            "required years of experience": _one(self.years_experience),
            "required hard skill": sorted(set(self.hard_skills.value)),
            "required soft skill": sorted(set(self.soft_skills.value)),
            "nice_to_have_skill": sorted(set(self.nice_to_have_skills.value)),
            "required education degree": _one(self.education_degree),
            "required field of study": _one(self.field_of_study),
            "required spoken language": sorted(set(self.languages.value)),
            "work_location": _one(self.work_location),
            "job_type": _one(self.job_type),
            "benefits": sorted(set(self.benefits.value)),
        }


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
