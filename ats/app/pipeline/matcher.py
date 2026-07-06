"""
Step 3 of the pipeline: hard label matching between JD requirements
(from jd_processor) and CV facts (from cv_processor). Same weighting
scheme as the notebook: skills 40%, experience 25%, education 20%,
languages 15%.
"""

import re
from typing import Dict

DEGREE_NORMALIZE = {
    "bachelor": ["bsc", "b.sc", "bs", "bachelor", "b.eng", "beng"],
    "master": ["msc", "m.sc", "ms", "master", "m.eng", "meng", "mba"],
    "phd": ["phd", "ph.d", "doctorate"],
}

DEGREE_RANK = {"bachelor": 1, "master": 2, "phd": 3}

WEIGHTS = {
    "skills": 0.40,
    "experience": 0.25,
    "education": 0.20,
    "languages": 0.15,
}


def normalize_degree(degree_list):
    normalized = set()
    for d in degree_list:
        for level, variants in DEGREE_NORMALIZE.items():
            if any(v in d.lower() for v in variants):
                normalized.add(level)
    return normalized


def hard_match(jd_extracted: Dict, cv_parsed: Dict) -> Dict:
    score_breakdown = {}

    # 1. Skills
    jd_skills = set(jd_extracted.get("required hard skill", []))
    cv_skills = set(cv_parsed["skills"])
    if jd_skills:
        matched = jd_skills & cv_skills
        missing = jd_skills - cv_skills
        skills_score = len(matched) / len(jd_skills)
    else:
        matched, missing, skills_score = set(), set(), 1.0
    score_breakdown["skills"] = {
        "score": round(skills_score, 2),
        "matched": sorted(matched),
        "missing": sorted(missing),
        "weight": WEIGHTS["skills"],
    }

    # 2. Experience
    jd_exp_entities = jd_extracted.get("required years of experience", [])
    jd_years = 0
    for e in jd_exp_entities:
        nums = re.findall(r"\d+", e)
        if nums:
            jd_years = int(nums[0])
            break
    cv_years = cv_parsed["years_of_experience"]
    exp_score = min(cv_years / jd_years, 1.0) if jd_years > 0 else 1.0
    score_breakdown["experience"] = {
        "score": round(exp_score, 2),
        "cv_years": cv_years,
        "required": jd_years,
        "weight": WEIGHTS["experience"],
    }

    # 3. Education
    jd_degrees = normalize_degree(jd_extracted.get("required education degree", []))
    cv_degrees = normalize_degree(cv_parsed["degrees"])
    jd_rank = max((DEGREE_RANK.get(d, 0) for d in jd_degrees), default=0)
    cv_rank = max((DEGREE_RANK.get(d, 0) for d in cv_degrees), default=0)
    edu_score = 1.0 if cv_rank >= jd_rank else (cv_rank / jd_rank if jd_rank else 1.0)
    score_breakdown["education"] = {
        "score": round(edu_score, 2),
        "cv": sorted(cv_degrees),
        "required": sorted(jd_degrees),
        "weight": WEIGHTS["education"],
    }

    # 4. Languages
    jd_langs = set(jd_extracted.get("required spoken language", []))
    cv_langs = set(cv_parsed["languages"])
    lang_score = (len(jd_langs & cv_langs) / len(jd_langs)) if jd_langs else 1.0
    score_breakdown["languages"] = {
        "score": round(lang_score, 2),
        "weight": WEIGHTS["languages"],
    }

    total = sum(
        score_breakdown[f]["score"] * score_breakdown[f]["weight"]
        for f in score_breakdown
    )

    return {"total": round(total, 3), "breakdown": score_breakdown}
