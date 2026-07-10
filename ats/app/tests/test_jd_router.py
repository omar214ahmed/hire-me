"""
Tests for pipeline/jd_router.py - the legacy-vs-LLM routing heuristics.

These are the concrete edge cases from the architecture review: emoji
headers, long unstructured prose, missing headers, and non-English JDs
should all route to "llm". Short, well-headered, English JDs should keep
using the cheap "legacy" path so cost/latency don't regress for the easy
majority of postings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.jd_router import choose_extraction_path


WELL_STRUCTURED_ENGLISH_JD = """Data Scientist

Requirements
5+ years of experience
Strong background in statistics

Skills
Python, R

Nice to Have
AWS certification

Employment Type
Full-time
"""

EMOJI_HEADER_JD = """🚀 About Us
We are a fast growing startup building the future of logistics.

🎯 What You'll Do
Ship product fast, own your area end to end.

🔥 What You Need
5+ years of Python, strong communication skills, and a growth mindset.

✨ Nice to Have
Experience with Kubernetes.
"""

NO_HEADERS_LONG_PARAGRAPH_JD = (
    "We're looking for someone who has spent the last several years "
    "building distributed systems in Python and Go, has a track record "
    "of leading small teams, communicates clearly with both engineers "
    "and non-technical stakeholders, holds a degree in computer science "
    "or a related field, and is comfortable working in a fast-paced, "
    "ambiguous environment where priorities can shift week to week "
    "depending on what customers need most urgently from the platform."
)

SPANISH_JD = (
    "Buscamos un Ingeniero de Software con experiencia solida en Python "
    "y bases de datos SQL. El candidato ideal debe tener conocimientos "
    "de machine learning, buenas habilidades de comunicacion y trabajo "
    "en equipo, ademas de experiencia previa en entornos de nube como "
    "AWS o GCP para desplegar sistemas a gran escala todos los dias."
)

LONG_JD_OVER_CHAR_LIMIT = "Requirements\n" + ("Python experience required. " * 200)


def test_well_structured_english_jd_routes_to_legacy():
    decision = choose_extraction_path(WELL_STRUCTURED_ENGLISH_JD)
    assert decision.path == "legacy"


def test_emoji_header_jd_routes_to_llm():
    decision = choose_extraction_path(EMOJI_HEADER_JD)
    assert decision.path == "llm"
    assert decision.emoji_count > 0


def test_no_headers_long_paragraph_jd_routes_to_llm():
    decision = choose_extraction_path(NO_HEADERS_LONG_PARAGRAPH_JD)
    assert decision.path == "llm"
    assert decision.recognized_headers == 0


def test_spanish_jd_routes_to_llm():
    decision = choose_extraction_path(SPANISH_JD)
    assert decision.path == "llm"
    assert decision.non_english_ratio > 0.5


def test_overly_long_jd_routes_to_llm_regardless_of_headers():
    decision = choose_extraction_path(LONG_JD_OVER_CHAR_LIMIT, max_legacy_chars=1500)
    assert decision.path == "llm"


def test_short_jd_below_word_threshold_does_not_get_penalized_for_language():
    """A very short JD shouldn't get flagged as "non-English" purely
    because there aren't enough words to judge - avoids a short, valid
    English JD being pushed to the (slower, costlier) LLM path for no
    real reason."""
    decision = choose_extraction_path("Data Analyst\n\nRequirements\nSQL\n\nSkills\nExcel")
    assert decision.non_english_ratio == 0.0
