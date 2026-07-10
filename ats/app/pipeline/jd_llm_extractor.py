"""
Structured-output LLM extraction path.

This is the module that actually fixes the failure modes the legacy
chunk -> classify -> merge -> GLiNER pipeline structurally cannot: emoji
section headers, missing/inconsistent headers, long mixed-topic
paragraphs, and non-English JDs. None of those are layout problems a
better regex can solve - they're comprehension problems, so this module
hands the whole cleaned JD to a model in a single call and asks for a
strict, schema-validated JSON object back, instead of pre-slicing the
text into sections and hoping the slice boundaries were right.

Design choices, and the failure mode each one prevents:

  - ONE call for the whole document (no chunking) -> a paragraph mixing
    a requirement, a skill, and a soft trait doesn't have to be forced
    into one bucket by an external chunker; the model reads it once and
    places each claim in the right field itself.
  - Tool-use / strict JSON schema, not free-form prose -> guarantees a
    parseable, typed result every time instead of an LLM essay that
    still needs its own extraction step.
  - Per-field confidence, explicitly requested -> lets the pipeline
    distinguish "field genuinely absent from this JD" from "the model
    guessed and isn't sure", which the legacy pipeline could never
    represent at all (it always emits *something*).
  - Explicit "leave null if unsure" instruction -> prevents confident-
    sounding hallucination of plausible-but-wrong values, which is the
    single biggest risk of moving extraction to an LLM.
  - No network/model call happens at import time, and `is_available()`
    is checked before every real call -> the module degrades to "not
    available" (caller falls back to the legacy path) rather than
    crashing a request when no API key is configured, e.g. in local dev
    or CI.
"""

import json
import logging
from typing import Optional

from helpers.config import get_settings
from pipeline.schemas import ExtractedField, ExtractedListField, JDExtraction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema: the model must call this "tool" with these exact arguments.
# Keeping the field set identical to JDExtraction means adding a new field
# to the pipeline is a two-line change (one here, one in the schema) and
# nothing else - no new regex family, no new section label, no new
# chunk-classification cue list.
# ---------------------------------------------------------------------------
_SCALAR_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
    },
    "required": ["value", "confidence"],
}

_LIST_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["value", "confidence"],
}

_EXTRACTION_TOOL = {
    "name": "record_jd_extraction",
    "description": (
        "Record the structured fields extracted from a job description. "
        "Every field must include a confidence score between 0 and 1. "
        "If a field is not present in the JD or you are not confident, "
        "set its value to null (or an empty list for list fields) and "
        "give it a low confidence rather than guessing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "job_title": _SCALAR_FIELD_SCHEMA,
            "years_experience": _SCALAR_FIELD_SCHEMA,
            "education_degree": _SCALAR_FIELD_SCHEMA,
            "field_of_study": _SCALAR_FIELD_SCHEMA,
            "work_location": _SCALAR_FIELD_SCHEMA,
            "job_type": _SCALAR_FIELD_SCHEMA,
            "hard_skills": _LIST_FIELD_SCHEMA,
            "soft_skills": _LIST_FIELD_SCHEMA,
            "nice_to_have_skills": _LIST_FIELD_SCHEMA,
            "languages": _LIST_FIELD_SCHEMA,
            "benefits": _LIST_FIELD_SCHEMA,
        },
        "required": [
            "job_title", "years_experience", "hard_skills", "soft_skills",
            "nice_to_have_skills", "education_degree", "field_of_study",
            "languages", "work_location", "job_type", "benefits",
        ],
    },
}

_SYSTEM_PROMPT = """You are an information-extraction engine for an Applicant \
Tracking System. You will be given the full raw text of a single job \
description. It may:
  - use emojis as bullets or section headers (e.g. "🚀 What you'll do")
  - use Markdown, HTML remnants, or completely unstructured prose
  - have inconsistent, missing, or unconventional section headers
  - be written in any language, or mix languages
  - contain long paragraphs that mix requirements, skills, and soft \
traits in the same sentence
  - contain duplicated information (the same requirement stated twice \
in different sections)

Read the ENTIRE document for meaning. Do not rely on section header \
keywords - infer what each field is from context and phrasing, in \
whatever language the JD is written in. Deduplicate repeated \
information (report each distinct skill/requirement once).

Extract any employee benefits or perks the posting offers (health/dental \
insurance, paid time off, parental leave, stipends, equity, remote-work \
policy, etc.) into "benefits" - one distinct entry per benefit, in your \
own concise phrasing. Benefits are not requirements or skills, and must \
never appear in those fields.

Distinguish REQUIRED items from NICE-TO-HAVE / preferred / bonus items \
based on the actual wording used ("must have" / "required" vs. \
"nice to have" / "preferred" / "a plus" / "bonus"), not on which section \
they physically sit in.

Call the record_jd_extraction tool exactly once with your result. For \
any field you cannot find or are not confident about, use null (or an \
empty list) and a low confidence score - never invent a plausible-\
sounding value."""


def is_available() -> bool:
    """Whether the LLM path can actually be used right now. Checked by
    the caller before every request so a missing/unset API key degrades
    to the legacy path instead of raising mid-request."""
    settings = get_settings()
    return bool(getattr(settings, "ANTHROPIC_API_KEY", "") or "")


def _get_client():
    """Lazy import + construction - keeps `anthropic` an optional
    dependency for deployments that only ever use the legacy path, and
    keeps this out of module import time so tests that never call the
    LLM path don't need the package installed or network access."""
    import anthropic

    settings = get_settings()
    return anthropic.Anthropic(
        api_key=settings.ANTHROPIC_API_KEY,
        timeout=settings.JD_LLM_TIMEOUT_SECONDS,
    )


def _extract_tool_input(response) -> Optional[dict]:
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if block_type == "tool_use":
            block_input = getattr(block, "input", None) if not isinstance(block, dict) else block.get("input")
            return block_input
    return None


def _build_field(raw: Optional[dict]) -> ExtractedField:
    if not isinstance(raw, dict):
        return ExtractedField()
    value = raw.get("value")
    if isinstance(value, str) and not value.strip():
        value = None
    confidence = raw.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    return ExtractedField(value=value, confidence=max(0.0, min(1.0, confidence)))


def _build_list_field(raw: Optional[dict]) -> ExtractedListField:
    if not isinstance(raw, dict):
        return ExtractedListField()
    value = raw.get("value")
    if not isinstance(value, list):
        value = []
    value = [str(v).strip() for v in value if str(v).strip()]
    confidence = raw.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    return ExtractedListField(value=value, confidence=max(0.0, min(1.0, confidence)))


def parse_tool_result(tool_input: dict) -> JDExtraction:
    """Pure function, deliberately separated from the network call, so
    the JSON->JDExtraction mapping can be unit-tested with canned tool
    outputs and never needs a real (or mocked) API call to verify."""
    return JDExtraction(
        job_title=_build_field(tool_input.get("job_title")),
        years_experience=_build_field(tool_input.get("years_experience")),
        hard_skills=_build_list_field(tool_input.get("hard_skills")),
        soft_skills=_build_list_field(tool_input.get("soft_skills")),
        nice_to_have_skills=_build_list_field(tool_input.get("nice_to_have_skills")),
        education_degree=_build_field(tool_input.get("education_degree")),
        field_of_study=_build_field(tool_input.get("field_of_study")),
        languages=_build_list_field(tool_input.get("languages")),
        work_location=_build_field(tool_input.get("work_location")),
        job_type=_build_field(tool_input.get("job_type")),
        benefits=_build_list_field(tool_input.get("benefits")),
        extraction_method="llm_structured",
    )


class JDExtractionError(RuntimeError):
    """Raised when the LLM path fails outright (network error, malformed
    response, no tool call returned, ...). Callers (jd_processor) catch
    this and fall back to the legacy path rather than failing the whole
    request - see extract_from_jd_v2's try/except."""


def extract_from_jd_llm(jd_text: str) -> JDExtraction:
    """Single structured-output call over the whole (lightly normalized)
    JD text. Raises JDExtractionError on any failure - callers decide
    the fallback behavior; this function's only job is "call the model,
    return a validated JDExtraction, or raise"."""
    if not is_available():
        raise JDExtractionError("ANTHROPIC_API_KEY not configured")

    settings = get_settings()
    client = _get_client()

    try:
        response = client.messages.create(
            model=settings.JD_LLM_MODEL,
            max_tokens=settings.JD_LLM_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            tools=[_EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "record_jd_extraction"},
            messages=[{"role": "user", "content": jd_text}],
        )
    except Exception as exc:  # network, auth, rate-limit, etc.
        raise JDExtractionError(f"LLM call failed: {exc}") from exc

    tool_input = _extract_tool_input(response)
    if tool_input is None:
        raise JDExtractionError("Model did not return a tool_use block")

    try:
        return parse_tool_result(tool_input)
    except Exception as exc:
        raise JDExtractionError(f"Malformed tool result: {exc}") from exc
