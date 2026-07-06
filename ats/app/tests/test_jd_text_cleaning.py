"""
Regression tests for `clean_jd_text` in pipeline/jd_processor.py.

Real-world JD payloads sent to the API are often mangled before they ever
reach `split_jd_sections`:
  - literal escape sequences as text ("\\n", "\\t") instead of real
    newlines/tabs - e.g. from double-encoded JSON or copy/pasting from a
    terminal/log. Before this fix, `split_jd_sections` (which splits on
    real "\n" characters) saw ZERO line breaks in that case and the whole
    JD collapsed into a single "header" bucket - no section was ever
    detected.
  - Windows "\r\n" line endings.
  - non-breaking spaces / zero-width characters / BOM marks pasted from
    Word or a web page.
  - leftover HTML (<p>, <br>, <li>, &nbsp;, &amp;, ...) from a scraped or
    HTML-sourced JD.
  - exotic bullet glyphs (•, ‣, ▪, ...) in front of otherwise short
    header-like lines.

`clean_jd_text` normalizes all of that BEFORE section splitting runs, and
is intentionally implemented with its own separate regex constants so
none of this cleaning logic ends up mixed into JD_SECTION_PATTERNS
(in particular the "skills" pattern).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.jd_processor import clean_jd_text, split_jd_sections


def test_literal_backslash_n_is_converted_to_real_newline():
    raw = "Data Scientist\\n\\nRequirements\\n5+ years Python\\n"
    cleaned = clean_jd_text(raw)
    assert "\\n" not in cleaned
    assert "\n" in cleaned
    assert cleaned == "Data Scientist\n\nRequirements\n5+ years Python"


def test_literal_backslash_t_becomes_space():
    raw = "Requirements\\nPython\\tSQL\\tDocker"
    cleaned = clean_jd_text(raw)
    assert "\\t" not in cleaned
    assert "Python SQL Docker" in cleaned


def test_windows_line_endings_normalized():
    raw = "Data Scientist\r\n\r\nRequirements\r\nPython\r\n"
    cleaned = clean_jd_text(raw)
    assert "\r" not in cleaned
    assert cleaned == "Data Scientist\n\nRequirements\nPython"


def test_non_breaking_space_and_zero_width_chars_stripped():
    raw = "Requirements\n5+\u00a0years\u200b Python"
    cleaned = clean_jd_text(raw)
    assert "\u00a0" not in cleaned
    assert "\u200b" not in cleaned
    assert "5+ years Python" in cleaned


def test_html_entities_and_tags_are_cleaned():
    raw = "<p>Skills</p><br>Docker &amp; Kubernetes&nbsp;tools"
    cleaned = clean_jd_text(raw)
    assert "<p>" not in cleaned
    assert "&amp;" not in cleaned
    assert "&nbsp;" not in cleaned
    assert "Skills" in cleaned
    assert "Docker & Kubernetes tools" in cleaned


def test_bullet_glyphs_stripped_from_line_start():
    raw = "Requirements\n\u2022 Python\n\u25aa SQL\n- Docker"
    cleaned = clean_jd_text(raw)
    for line in cleaned.split("\n"):
        assert not line.startswith(("\u2022", "\u25aa", "- "))
    assert "Python" in cleaned and "SQL" in cleaned and "Docker" in cleaned


def test_excess_blank_lines_and_spaces_collapsed():
    raw = "Data Scientist\n\n\n\nRequirements\nPython   and    SQL"
    cleaned = clean_jd_text(raw)
    assert "\n\n\n" not in cleaned
    assert "Python and SQL" in cleaned


def test_clean_jd_text_is_idempotent():
    raw = "Data Scientist\\n\\nRequirements\\nPython\\tSQL"
    once = clean_jd_text(raw)
    twice = clean_jd_text(once)
    assert once == twice


def test_empty_and_none_safe():
    assert clean_jd_text("") == ""


def test_split_jd_sections_no_longer_collapses_on_escaped_newlines():
    """End-to-end regression: a JD sent with literal '\\n' text used to
    produce a single 'header' bucket containing the whole document,
    because split_jd_sections never saw a real newline to split on."""
    raw = (
        "Data Scientist\\n\\n"
        "About the role\\nWe build ML systems.\\n\\n"
        "Requirements\\n5+ years Python\\tSQL   experience\\n"
        "\u00a0\u2022 Nice to have: AWS\\n\\n\\n"
        "<p>Skills</p><br>Docker, Kubernetes&nbsp;&amp; Terraform"
    )
    sections = split_jd_sections(raw)

    assert "Data Scientist" in sections["header"]
    assert "Python" in sections["requirements"]
    assert "SQL" in sections["requirements"]
    assert "Docker" in sections["skills"]
    assert "Kubernetes" in sections["skills"]
    # Regression guard: before the fix, everything ended up crammed into
    # "header" as one unsplit blob.
    assert "Docker" not in sections["header"]
    assert "Requirements" not in sections["header"]
