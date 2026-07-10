"""
Tests for pipeline/jd_llm_extractor.py.

Split into two groups:
  1. Pure parsing tests (parse_tool_result) - no network, no mocking,
     just JSON-in / JDExtraction-out. These are the tests that should
     catch a schema drift between the tool definition and JDExtraction.
  2. End-to-end call tests with a stubbed `anthropic` client - verifies
     is_available()/extract_from_jd_llm's control flow (missing key,
     network failure, malformed tool result) without ever making a real
     network call.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import jd_llm_extractor
from pipeline.schemas import JDExtraction


VALID_TOOL_INPUT = {
    "job_title": {"value": "Backend Engineer", "confidence": 0.9},
    "years_experience": {"value": "3+ years", "confidence": 0.8},
    "hard_skills": {"value": ["Go", "Kubernetes"], "confidence": 0.85},
    "soft_skills": {"value": [], "confidence": 0.0},
    "nice_to_have_skills": {"value": ["Rust"], "confidence": 0.5},
    "education_degree": {"value": None, "confidence": 0.0},
    "field_of_study": {"value": None, "confidence": 0.0},
    "languages": {"value": ["English"], "confidence": 0.7},
    "work_location": {"value": "Berlin, Germany", "confidence": 0.9},
    "job_type": {"value": "full-time", "confidence": 0.9},
}


def test_parse_tool_result_maps_all_fields():
    result = jd_llm_extractor.parse_tool_result(VALID_TOOL_INPUT)
    assert isinstance(result, JDExtraction)
    assert result.job_title.value == "Backend Engineer"
    assert result.job_title.confidence == 0.9
    assert result.hard_skills.value == ["Go", "Kubernetes"]
    assert result.extraction_method == "llm_structured"


def test_parse_tool_result_handles_missing_confidence_gracefully():
    tool_input = dict(VALID_TOOL_INPUT)
    tool_input["job_title"] = {"value": "Data Analyst"}  # no confidence key
    result = jd_llm_extractor.parse_tool_result(tool_input)
    assert result.job_title.value == "Data Analyst"
    assert result.job_title.confidence == 0.0


def test_parse_tool_result_clamps_out_of_range_confidence():
    tool_input = dict(VALID_TOOL_INPUT)
    tool_input["job_title"] = {"value": "Data Analyst", "confidence": 5.0}
    result = jd_llm_extractor.parse_tool_result(tool_input)
    assert result.job_title.confidence == 1.0


def test_parse_tool_result_treats_blank_string_value_as_none():
    tool_input = dict(VALID_TOOL_INPUT)
    tool_input["work_location"] = {"value": "   ", "confidence": 0.9}
    result = jd_llm_extractor.parse_tool_result(tool_input)
    assert result.work_location.value is None


def test_is_available_false_without_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from helpers.config import get_settings
    get_settings.cache_clear()
    assert jd_llm_extractor.is_available() is False
    get_settings.cache_clear()


def test_extract_from_jd_llm_raises_when_not_available(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from helpers.config import get_settings
    get_settings.cache_clear()
    with pytest.raises(jd_llm_extractor.JDExtractionError):
        jd_llm_extractor.extract_from_jd_llm("some jd text")
    get_settings.cache_clear()


def _install_fake_anthropic(monkeypatch, tool_input=None, raise_on_create=None):
    """Installs a fake `anthropic` module in sys.modules so
    jd_llm_extractor's lazy `import anthropic` picks it up, without any
    real network access or the real package installed."""
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
            if raise_on_create:
                raise raise_on_create
            return FakeResponse([FakeBlock("tool_use", tool_input or VALID_TOOL_INPUT)])

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    fake_module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)


def test_extract_from_jd_llm_happy_path(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    from helpers.config import get_settings
    get_settings.cache_clear()
    _install_fake_anthropic(monkeypatch)

    result = jd_llm_extractor.extract_from_jd_llm("some JD text with emoji headers")
    assert result.job_title.value == "Backend Engineer"
    assert result.extraction_method == "llm_structured"
    get_settings.cache_clear()


def test_extract_from_jd_llm_raises_on_network_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    from helpers.config import get_settings
    get_settings.cache_clear()
    _install_fake_anthropic(monkeypatch, raise_on_create=ConnectionError("boom"))

    with pytest.raises(jd_llm_extractor.JDExtractionError):
        jd_llm_extractor.extract_from_jd_llm("some jd text")
    get_settings.cache_clear()


def test_extract_from_jd_llm_raises_when_no_tool_use_block(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    from helpers.config import get_settings
    get_settings.cache_clear()

    fake_module = types.ModuleType("anthropic")

    class FakeResponse:
        content = []  # no tool_use block at all

    class FakeMessages:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    fake_module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    with pytest.raises(jd_llm_extractor.JDExtractionError):
        jd_llm_extractor.extract_from_jd_llm("some jd text")
    get_settings.cache_clear()
