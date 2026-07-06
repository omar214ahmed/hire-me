"""
Unit tests for the storage/model-agnostic parts of the pipeline —
these run with zero external dependencies (no Postgres, no Redis, no ML
models), so they're safe to run in CI on every commit.

Run with:  pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.matcher import hard_match, normalize_degree
from pipeline import storage
from pipeline.cv_processor import (
    parse_cv,
    extract_email,
    extract_phone,
    extract_years_of_experience,
    extract_degrees,
    split_cv_sections,
)
from pipeline.jd_processor import (
    build_jd_query,
    split_jd_sections,
    _extract_soft_skills_fallback,
    _extract_nice_to_have_skills_fallback,
    _extract_location_fallback,
)
from pipeline.ranker import attach_explainability, semantic_scores, combine_scores


def test_normalize_degree_variants():
    assert normalize_degree(["bsc", "msc"]) == {"bachelor", "master"}
    assert normalize_degree(["PhD"]) == {"phd"}
    assert normalize_degree([]) == set()


def test_hard_match_perfect_score():
    jd = {
        "required hard skill": ["python", "sql"],
        "required years of experience": ["5 years"],
        "required education degree": ["bachelor"],
        "required spoken language": ["english"],
    }
    cv = {
        "skills": ["python", "sql", "docker"],
        "years_of_experience": 6,
        "degrees": ["bachelor"],
        "languages": ["english"],
    }
    result = hard_match(jd, cv)
    assert result["total"] == 1.0
    assert result["breakdown"]["skills"]["score"] == 1.0


def test_hard_match_partial_skills():
    jd = {"required hard skill": ["python", "sql", "aws"], "required years of experience": [],
          "required education degree": [], "required spoken language": []}
    cv = {"skills": ["python"], "years_of_experience": 0, "degrees": [], "languages": []}
    result = hard_match(jd, cv)
    # 1/3 skills matched, weighted 0.4 -> contributes ~0.133; other categories
    # default to 1.0 when JD doesn't specify a requirement.
    assert abs(result["breakdown"]["skills"]["score"] - 0.33) < 0.01
    assert result["breakdown"]["skills"]["missing"] == ["aws", "sql"]


def test_hard_match_empty_jd_requirements_scores_full():
    # documents current (debatable) behavior flagged earlier: an empty JD
    # requirement list scores 1.0 rather than being excluded/neutral.
    jd = {"required hard skill": [], "required years of experience": [],
          "required education degree": [], "required spoken language": []}
    cv = {"skills": [], "years_of_experience": 0, "degrees": [], "languages": []}
    result = hard_match(jd, cv)
    assert result["total"] == 1.0


def test_extract_email():
    assert extract_email("Contact me at jane.doe@example.com please") == "jane.doe@example.com"
    assert extract_email("no email here") is None


def test_extract_phone():
    assert extract_phone("Call +1 (555) 123-4567 anytime") is not None


def test_extract_years_of_experience_sums_ranges():
    text = "Company A: 2015 to 2018\nCompany B: 2019 to Present"
    years = extract_years_of_experience(text)
    assert years >= 3  # 2015-2018 = 3; 2019-present depends on current year


def test_extract_degrees():
    assert "bsc" in extract_degrees("Holds a BSc in Computer Science")
    assert extract_degrees("no degree mentioned") == []


def test_split_cv_sections_detects_headers():
    text = "John Doe\n\nExperience\nSoftware Engineer at Foo\n\nEducation\nBSc Computer Science"
    sections = split_cv_sections(text)
    assert "experience" in sections
    assert "education" in sections
    assert "Software Engineer" in sections["experience"]


def test_parse_cv_end_to_end():
    text = (
        "Jane Smith\njane@example.com\n+1 555 000 1111\n\n"
        "Experience\n2019 to Present at Acme Corp\n\n"
        "Education\nBSc Computer Science\n\n"
        "Skills\nPython, SQL, Docker\n\n"
        "Languages\nEnglish, French\n"
    )
    parsed = parse_cv(cv_id="test123", text=text)
    assert parsed["id"] == "test123"
    assert parsed["email"] == "jane@example.com"
    assert "python" in parsed["skills"]
    assert "bsc" in parsed["degrees"]


def test_build_jd_query_formats_fields():
    extracted = {
        "required hard skill": ["python", "sql"],
        "required job title": ["backend engineer"],
        "required years of experience": ["5 years"],
        "required education degree": ["bachelor"],
    }
    query = build_jd_query(extracted)
    assert "Skills: python, sql" in query
    assert "Role: backend engineer" in query


def test_build_jd_query_handles_missing_fields():
    assert build_jd_query({}) == ""


def test_combine_scores_penalizes_weak_hard_match():
    weak = combine_scores(0.95, 0.5, has_hard_requirements=True)
    strong = combine_scores(0.95, 1.0, has_hard_requirements=True)
    assert weak < 0.95
    assert strong == 0.95


def test_extract_soft_skills_fallback_from_soft_skills_section():
    sections = {"soft_skills": "Communication\nTeamwork\nProblem solving"}
    extracted = _extract_soft_skills_fallback(sections)
    assert "Communication" in extracted
    assert "Teamwork" in extracted
    assert "Problem Solving" in extracted


def test_extract_nice_to_have_skills_fallback_from_nice_to_have_section():
    sections = {"nice_to_have": "AWS\nKubernetes\nLangChain"}
    extracted = _extract_nice_to_have_skills_fallback(sections)
    assert "AWS" in extracted
    assert "Kubernetes" in extracted
    assert "LangChain" in extracted


def test_extract_location_fallback_from_job_meta_section():
    sections = {"job_meta": "Location: Remote - London\nEmployment Type: Full-time"}
    extracted = _extract_location_fallback(sections)
    assert "Remote - London" in extracted


def test_split_jd_sections_detects_headers():
    text = (
        "About the role\nWe are hiring.\n\n"
        "Requirements\nPython, SQL, 5 years experience\n\n"
        "Education\nBSc in Computer Science"
    )
    sections = split_jd_sections(text)
    assert "requirements" in sections
    assert "Python" in sections["requirements"]
    assert "education" in sections
    assert "BSc" in sections["education"]


def test_build_jd_query_includes_requirements_section():
    extracted = {"required hard skill": ["python"]}
    sections = {"requirements": "Must know Python and SQL"}
    query = build_jd_query(extracted, sections=sections)
    assert "Requirements: Must know Python and SQL" in query


def test_semantic_scores_orders_by_cosine_similarity():
    jd_vec = [1.0, 0.0]
    cv_vecs = [[1.0, 0.0], [0.0, 1.0]]
    scores = semantic_scores(jd_vec, cv_vecs)
    assert scores[0] > scores[1]


def test_attach_explainability_keeps_semantic_score_and_adds_hard_match():
    jd_extracted = {
        "required hard skill": ["python"],
        "required years of experience": [],
        "required education degree": [],
        "required spoken language": [],
    }
    candidate_records = [{
        "parsed": {
            "id": "cand1",
            "skills": ["python", "sql"],
            "years_of_experience": 3,
            "degrees": [],
            "languages": [],
            "sections": {"skills": "python, sql", "experience": "", "education": ""},
        },
        "semantic_score": 0.87,
    }]
    result = attach_explainability(jd_extracted, candidate_records)
    assert result[0]["cv_id"] == "cand1"
    assert result[0]["semantic_score"] == 0.87
    assert result[0]["hard_match"]["breakdown"]["skills"]["score"] == 1.0
    # ordering/filtering is NOT this function's job — it's cosine similarity
    # (done in the DB) that decided which candidates made it this far.


def test_storage_close_pool_is_available():
    assert hasattr(storage, "close_pool")


def test_candidate_text_key_normalizes_whitespace_and_case():
    parsed = {"full_text": "  Python   Developer\nExperience  "}
    assert storage._candidate_text_key(parsed) == "python developer experience"


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
