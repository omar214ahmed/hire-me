"""
Regression test for the job-title-not-detected bug.

Root cause: `_ner_text()` built the GLiNER input only from the
requirements/skills/experience/education/languages sections. Real JDs
almost always state the title in the first line(s) (which
`split_jd_sections` buckets into "header") or in an "About the role"
blurb ("summary") — neither of which was included, so GLiNER never saw
the title text and "required job title" came back empty.

Fix: `_ner_text()` now also includes "header" and "summary".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.jd_processor import _ner_text, split_jd_sections


def test_ner_text_includes_header_with_job_title():
    jd_text = (
        "Senior Data Engineer\n\n"
        "Acme Corp is looking for a Senior Data Engineer to join our team.\n\n"
        "Requirements\n"
        "5+ years experience with Python and SQL\n"
        "Experience with Airflow and Spark\n"
    )
    sections = split_jd_sections(jd_text)
    ner_input = _ner_text(jd_text, sections)

    # Before the fix, "Senior Data Engineer" (the header line) was
    # dropped entirely from the NER input.
    assert "Senior Data Engineer" in ner_input
    assert "Python" in ner_input


def test_ner_text_includes_summary_with_job_title():
    jd_text = (
        "About the role\n"
        "We are hiring a Backend Engineer to help scale our platform.\n\n"
        "Skills\n"
        "Python, Django, PostgreSQL\n"
    )
    sections = split_jd_sections(jd_text)
    ner_input = _ner_text(jd_text, sections)

    assert "Backend Engineer" in ner_input
    assert "Django" in ner_input


def test_ner_text_still_falls_back_to_full_text_when_sections_too_short():
    jd_text = "Data Analyst wanted."
    sections = split_jd_sections(jd_text)
    ner_input = _ner_text(jd_text, sections)
    assert ner_input == jd_text
