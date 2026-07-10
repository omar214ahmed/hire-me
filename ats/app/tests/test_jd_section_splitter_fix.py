"""
Regression tests for the JD section-splitter bugs reported against
`split_jd_sections` / `_ner_text` in pipeline/jd_processor.py.

Bug 1 (header coverage): JD_SECTION_PATTERNS required the *entire*
header line to equal one bare keyword ("requirements", "skills", ...).
Real-world headers almost always qualify that keyword ("Required
Qualifications", "Technical Skills", "Soft Skills", "Experience Level",
"Employment Type", ...), so those lines never matched and silently got
appended as content to whatever section happened to be active before
them.

Bug 2 (content loss on repeat, the more serious one): whenever a header
matched a section key that had *already* been used earlier in the
document (e.g. "Technical Skills" -> skills, then later "Tools" also ->
skills), the old code did `sections[current] = []`, which discarded
everything collected under that key so far. On a JD with several skill
sub-headings (Programming / Machine Learning / Data Analysis /
Libraries / Tools), only the content between the *last* two skill
headers survived - everything else vanished.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.jd_processor import _ner_text, split_jd_sections

REAL_WORLD_JD = """Data Scientist

About the role
We build ML systems that power fraud detection.

Required Qualifications
5+ years of experience
Strong background in statistics

Technical Skills
Programming
Python, R
Machine Learning
TensorFlow, PyTorch
Data Analysis
Pandas, NumPy
Libraries
scikit-learn
Tools
Git, Docker

Preferred Qualifications
Experience with Kubernetes is a plus

Soft Skills
Communication, teamwork

Nice to Have
AWS certification

What You'll Learn
Distributed systems

Employment Type
Full-time

Experience Level
Mid-level
"""


def test_previously_unmatched_headers_are_now_recognized():
    """Every heading style from the bug report should now be detected as
    a header (i.e. not appear verbatim as a stray line inside another
    section's text) and routed somewhere sensible."""
    sections = split_jd_sections(REAL_WORLD_JD)

    # None of these header lines should leak into the body of a section
    # as raw text - they should have been consumed as headers.
    header_lines = [
        "Required Qualifications",
        "Preferred Qualifications",
        "Technical Skills",
        "Soft Skills",
        "Nice to Have",
        "Employment Type",
        "Experience Level",
        "What You'll Learn",
    ]
    all_text = "\n".join(sections.values())
    for header in header_lines:
        assert header not in all_text, f"{header!r} was not recognized as a header"


def test_required_vs_preferred_qualifications_are_split_correctly():
    """'Required Qualifications' must land in requirements; 'Preferred
    Qualifications' must land in a separate nice_to_have bucket instead
    of being merged into hard requirements."""
    sections = split_jd_sections(REAL_WORLD_JD)

    assert "5+ years of experience" in sections["requirements"]
    assert "Strong background in statistics" in sections["requirements"]

    assert "Kubernetes" in sections["nice_to_have"]
    assert "AWS certification" in sections["nice_to_have"]
    # Preferred content must NOT bleed into the hard-requirements bucket.
    assert "Kubernetes" not in sections["requirements"]


def test_no_content_lost_across_repeated_skill_subheadings():
    """Regression test for the content-loss bug: every skill sub-heading
    (Programming / Machine Learning / Data Analysis / Libraries / Tools)
    maps to the same 'skills' canonical key. Before the fix, each repeat
    match reset the accumulator and wiped everything gathered so far -
    only the last sub-section (Tools) survived."""
    sections = split_jd_sections(REAL_WORLD_JD)
    skills_text = sections["skills"]

    for expected in ["Python, R", "TensorFlow, PyTorch", "Pandas, NumPy", "scikit-learn", "Git, Docker"]:
        assert expected in skills_text, f"{expected!r} was lost from the skills section"


def test_soft_skills_and_employment_metadata_dont_pollute_other_sections():
    sections = split_jd_sections(REAL_WORLD_JD)

    assert "Communication, teamwork" in sections["soft_skills"]
    assert "Full-time" in sections["job_meta"]
    assert "Mid-level" in sections["job_meta"]
    # Metadata shouldn't leak into hard requirements or experience text.
    assert "Full-time" not in sections["requirements"]
    assert "Mid-level" not in sections.get("experience", "")


def test_repeated_header_appends_instead_of_overwriting():
    """Minimal, isolated repro of the reset bug: two headers mapping to
    the same key, with distinct content in between, must both survive."""
    jd_text = (
        "Requirements\n"
        "Bachelor's degree required\n\n"
        "Nice to Have\n"
        "A master's degree is a plus\n\n"
        "Additional Requirements\n"
        "Must be willing to relocate\n"
    )
    sections = split_jd_sections(jd_text)

    assert "Bachelor's degree required" in sections["requirements"]
    assert "Must be willing to relocate" in sections["requirements"]
    assert "master's degree is a plus" in sections["nice_to_have"]


def test_ner_text_includes_nice_to_have_and_job_meta():
    """The new buckets must actually reach GLiNER, not just exist in the
    sections dict, or the "preferred technology tool" / "job type" /
    "location" labels will keep coming back empty on real JDs."""
    sections = split_jd_sections(REAL_WORLD_JD)
    ner_input = _ner_text(REAL_WORLD_JD, sections)

    assert "Kubernetes" in ner_input
    assert "Full-time" in ner_input


def test_bare_qualifications_header_still_maps_to_requirements():
    """Sanity check: an unqualified 'Qualifications' header (no
    'required'/'preferred' prefix) should still map to requirements,
    not accidentally get swallowed by the new nice_to_have pattern."""
    jd_text = "Qualifications\nMust know SQL\n"
    sections = split_jd_sections(jd_text)
    assert "Must know SQL" in sections["requirements"]
