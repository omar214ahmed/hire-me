"""
End-to-end tests for pipeline.jd_processor.extract_from_jd_v2 - the new
primary entry point that routes each JD to the legacy or LLM path and
returns a confidence-scored JDExtraction.

GLiNER and the Anthropic client are both stubbed out (via monkeypatch) so
these tests run fast, deterministically, and without real model weights
or network/API-key requirements - exactly the point of separating
routing/confidence logic from the model calls themselves.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import jd_processor
from pipeline.models import ModelRegistry
from pipeline.schemas import JDExtraction


class _FakeNER:
    """Deterministic stand-in for GLiNER: looks for a small fixed set of
    literal substrings per label, instead of doing real zero-shot NER.
    Enough to exercise _bucket_entities / the legacy path's control flow
    without needing real model weights."""

    def predict_entities(self, text, labels, threshold=0.3, flat_ner=True):
        out = []
        if "job title" in labels:
            for title in ("Data Scientist", "Senior Data Engineer"):
                if title in text:
                    out.append({"label": "job title", "text": title, "score": 0.9})
        if "programming language or technical skill" in labels:
            for tok in ("Python", "SQL", "R"):
                if tok in text:
                    out.append({
                        "label": "programming language or technical skill",
                        "text": tok, "score": 0.8,
                    })
        return out


@pytest.fixture(autouse=True)
def _stub_ner(monkeypatch):
    monkeypatch.setattr(ModelRegistry, "ner", classmethod(lambda cls: _FakeNER()))
    yield


def _install_fake_anthropic(monkeypatch, tool_input):
    fake_module = types.ModuleType("anthropic")

    class FakeBlock:
        def __init__(self, type_, input_):
            self.type = type_
            self.input = input_

    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeMessages:
        def create(self, **kwargs):
            return FakeResponse([FakeBlock("tool_use", tool_input)])

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    fake_module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)


WELL_STRUCTURED_JD = (
    "Data Scientist\n\n"
    "Requirements\n5+ years Python SQL experience\n\n"
    "Skills\nPython, SQL\n\n"
    "Employment Type\nFull-time\n"
)

EMOJI_JD = (
    "🚀 About Us\nWe build the future of logistics.\n\n"
    "🎯 What You Will Do\nShip product fast.\n\n"
    "🔥 What You Need\n5+ years Python, strong communication.\n"
)


def test_well_structured_jd_uses_legacy_path(monkeypatch):
    from helpers.config import get_settings
    get_settings.cache_clear()
    result = jd_processor.extract_from_jd_v2(WELL_STRUCTURED_JD)

    assert isinstance(result, JDExtraction)
    assert result.extraction_method == "legacy_regex"
    assert result.job_title.value == "Data Scientist"
    assert not result.needs_review
    get_settings.cache_clear()


def test_emoji_jd_without_api_key_falls_back_to_legacy(monkeypatch):
    """No ANTHROPIC_API_KEY configured -> router picks "llm", but the LLM
    path is unavailable, so extract_from_jd_v2 must fall back to legacy
    instead of raising - the request must still succeed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from helpers.config import get_settings
    get_settings.cache_clear()

    result = jd_processor.extract_from_jd_v2(EMOJI_JD)
    assert result.extraction_method == "legacy_regex"
    assert "LLM path failed" in (result.routing_reason or "")
    get_settings.cache_clear()


def test_emoji_jd_with_api_key_uses_llm_path(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    from helpers.config import get_settings
    get_settings.cache_clear()

    tool_input = {
        "job_title": {"value": "Logistics Engineer", "confidence": 0.9},
        "years_experience": {"value": None, "confidence": 0.0},
        "hard_skills": {"value": ["Python"], "confidence": 0.85},
        "soft_skills": {"value": ["communication"], "confidence": 0.7},
        "nice_to_have_skills": {"value": [], "confidence": 0.0},
        "education_degree": {"value": None, "confidence": 0.0},
        "field_of_study": {"value": None, "confidence": 0.0},
        "languages": {"value": [], "confidence": 0.0},
        "work_location": {"value": None, "confidence": 0.0},
        "job_type": {"value": None, "confidence": 0.0},
    }
    _install_fake_anthropic(monkeypatch, tool_input)

    result = jd_processor.extract_from_jd_v2(EMOJI_JD)
    assert result.extraction_method == "llm_structured"
    assert result.job_title.value == "Logistics Engineer"
    get_settings.cache_clear()


def test_forced_legacy_mode_never_calls_llm(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("JD_EXTRACTION_MODE", "legacy")
    from helpers.config import get_settings
    get_settings.cache_clear()

    # No fake anthropic installed at all - if the code tried to call the
    # LLM path, importing/calling it would fail loudly (ModuleNotFoundError
    # or similar), proving forced "legacy" mode genuinely skips it.
    result = jd_processor.extract_from_jd_v2(EMOJI_JD)
    assert result.extraction_method == "legacy_regex"
    assert "forced" in (result.routing_reason or "")
    get_settings.cache_clear()


def test_missing_job_title_flags_needs_review(monkeypatch):
    from helpers.config import get_settings
    get_settings.cache_clear()

    result = jd_processor.extract_from_jd_v2("Requirements\nMust know SQL\n")
    assert result.job_title.value is None
    assert result.needs_review is True
    get_settings.cache_clear()


def test_with_sections_v2_projects_to_legacy_shape_for_downstream_compat(monkeypatch):
    from helpers.config import get_settings
    get_settings.cache_clear()

    payload = jd_processor.extract_from_jd_with_sections_v2(WELL_STRUCTURED_JD)
    assert set(payload.keys()) == {"extracted", "sections", "jd_extraction"}
    assert payload["extracted"]["required job title"] == ["Data Scientist"]
    assert isinstance(payload["jd_extraction"], JDExtraction)
    get_settings.cache_clear()
