"""
Step 1 of the pipeline: turn a raw Job Description string into structured,
labeled requirements using GLiNER (zero-shot NER).

Mirrors the notebook's `extract_from_jd`, but the labels are kept generic
("job title", "years of experience", ...) so the same module can be reused
if you swap GLiNER models later.

Sectioning (chunk -> classify -> merge) now lives in `jd_chunker.py` - see
that module's docstring for why. This file re-exports `clean_jd_text` and
`split_jd_sections` from there so nothing downstream (routers, tests) needs
to change, and focuses purely on what happens AFTER sections exist: running
GLiNER per section, deterministic fallbacks, and bucketing entities into the
canonical output keys `matcher.hard_match` compares "like with like" on.
"""

import logging
import re
from typing import Dict, List

from pipeline.models import ModelRegistry
from pipeline.jd_chunker import (
    clean_jd_text,
    split_jd_sections,
    JD_SECTION_PATTERNS,
)

logger = logging.getLogger(__name__)

JD_LABELS = [
    "job title",
    "years of experience",
    "programming language or technical skill",
    "soft skill or personality trait",
    "education degree",
    "field of study",
    "spoken or written language",
    "preferred technology tool listed as a plus or optional",
    "city country or region where job is based",
    "job type or work arrangement",
]

# Map GLiNER's natural-language labels -> the canonical keys used
# downstream by the hard matcher (kept identical to the notebook's keys).
LABEL_KEY_MAP = {
    "job title": "required job title",
    "years of experience": "required years of experience",
    "programming language or technical skill": "required hard skill",
    "soft skill or personality trait": "required soft skill",
    "education degree": "required education degree",
    "field of study": "required field of study",
    "spoken or written language": "required spoken language",
    "preferred technology tool listed as a plus or optional": "nice_to_have_skill",
    "city country or region where job is based": "work_location",
    "job type or work arrangement": "job_type",
}


# ---------------------------------------------------------------------------
# Deterministic post-processing (after GLiNER, before the final bucketing)
# ---------------------------------------------------------------------------
# GLiNER's own judgement on "is this required or optional" is inherently
# fuzzy, and real JDs often say the quiet part out loud right next to the
# item itself - e.g. a "Libraries" list like:
#     Scikit-learn, TensorFlow (preferred), PyTorch (preferred), XGBoost (preferred)
# where only the parenthetical actually marks which ones are optional.
# Nothing here touches JD_SECTION_PATTERNS - this only re-checks entities
# GLiNER already returned, using the literal JD text as ground truth.
_OPTIONAL_INLINE_MARKER = re.compile(
    r"\(\s*(?:preferred|optional|nice[\s-]to[\s-]have|bonus|a\s+plus|desired)\s*\)",
    re.IGNORECASE,
)

# Closed, well-known vocabulary for employment type / work arrangement.
# Used as a deterministic safety net alongside GLiNER's "job type or work
# arrangement" label - a short field like "Full-time" is exactly the kind
# of thing a general-purpose zero-shot label can miss, and there's no
# reason to leave "job_type" empty when the literal word is right there
# in the job_meta section.
_JOB_TYPE_VOCAB = re.compile(
    r"\b(full[\s-]?time|part[\s-]?time|contract(?:or)?|freelance|"
    r"intern(?:ship)?|temporary|temp|permanent|remote|hybrid|on[\s-]?site)\b",
    re.IGNORECASE,
)

def _looks_optional_inline(entity_text: str, ner_input: str) -> bool:
    """True if entity_text is immediately followed (within a short
    window, allowing for punctuation) by an inline optional marker in
    the source text, e.g. "TensorFlow (preferred)". Checks every
    occurrence of entity_text in the text, since the same skill can be
    mentioned more than once (marked in one place, unmarked in another)."""
    if not entity_text:
        return False
    haystack = ner_input.lower()
    needle = entity_text.lower()
    start = haystack.find(needle)
    while start != -1:
        window = ner_input[start : start + len(needle) + 20]
        if _OPTIONAL_INLINE_MARKER.search(window):
            return True
        start = haystack.find(needle, start + 1)
    return False


def _came_from_nice_to_have_section(entity_text: str, sections: Dict[str, str]) -> bool:
    """Coarse provenance check: the entity's text shows up in the
    nice_to_have section but nowhere in requirements/skills. Used as a
    secondary signal alongside the inline marker above."""
    if not entity_text:
        return False
    text = entity_text.lower()
    nice = sections.get("nice_to_have", "").lower()
    hard = (sections.get("requirements", "") + " " + sections.get("skills", "")).lower()
    return text in nice and text not in hard


def _is_actually_optional(entity_text: str, ner_input: str, sections: Dict[str, str]) -> bool:
    return _looks_optional_inline(entity_text, ner_input) or _came_from_nice_to_have_section(
        entity_text, sections
    )


def _looks_like_soft_skill(text: str) -> bool:
    lowered = _normalize_text(text).lower()
    soft_skill_markers = {
        "attention",
        "detail",
        "problem",
        "solving",
        "critical",
        "thinking",
        "communication",
        "collaboration",
        "team",
        "leadership",
        "analytical",
        "curiosity",
        "willingness",
        "learning",
        "adaptability",
        "interpersonal",
        "empathy",
        "management",
        "time",
        "ownership",
        "accountability",
    }
    tokens = set(re.split(r"[^a-z0-9]+", lowered))
    return bool(tokens & soft_skill_markers) or re.search(
        r"\b(problem|solve|communicat|collaborat|critical|thinking|attention|detail|analytical|team|leadership|curiosity|willing|adapt|empathy|interpersonal)\b",
        lowered,
    )


def _came_from_soft_skills_section(entity_text: str, sections: Dict[str, str]) -> bool:
    if not entity_text:
        return False
    text = entity_text.lower()
    soft_skills = sections.get("soft_skills", "").lower()
    if text in soft_skills:
        return True
    if _looks_like_soft_skill(entity_text):
        return "soft skills" in sections.get("soft_skills", "").lower() or "soft" in sections.get("soft_skills", "").lower()
    return False


def _job_type_fallback(sections: Dict[str, str]) -> List[str]:
    """Deterministic supplement for job_type: scans the job_meta section
    text directly for standard employment-type wording. Only adds
    matches - never removes anything GLiNER already found - so this
    can only fill genuine gaps, not override the model."""
    text = sections.get("job_meta", "")
    return [m.group(0).lower() for m in _JOB_TYPE_VOCAB.finditer(text)]


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _canonicalize_skill(value: str) -> str:
    text = _normalize_text(value)
    if not text:
        return ""

    cleaned = text.lower()
    cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", cleaned)
    cleaned = re.sub(r"[.!,;:]+$", "", cleaned)

    generic_tokens = {
        "strong",
        "good",
        "basic",
        "solid",
        "working",
        "hands-on",
        "practical",
        "familiarity",
        "experience",
        "knowledge",
        "understanding",
        "proficiency",
        "comfort",
        "interest",
        "ability",
        "skills",
        "skill",
        "programming",
        "technical",
        "version",
        "control",
        "preferred",
        "optional",
        "bonus",
        "desired",
        "plus",
        "with",
        "in",
        "of",
        "for",
        "using",
        "and",
        "or",
        "to",
        "the",
        "a",
        "an",
        "is",
        "are",
        "be",
    }
    soft_skill_tokens = {
        "attention",
        "detail",
        "problem",
        "solving",
        "critical",
        "thinking",
        "communication",
        "teamwork",
        "collaboration",
        "management",
        "analytical",
        "willingness",
        "curiosity",
        "learning",
        "time",
    }

    tokens = [tok for tok in re.split(r"[^a-z0-9]+", cleaned) if tok]
    tokens = [tok for tok in tokens if tok not in generic_tokens]

    if not tokens:
        return ""

    if any(tok in soft_skill_tokens for tok in tokens):
        return ""

    if len(tokens) > 2:
        return ""

    normalized = []
    for tok in tokens:
        if re.fullmatch(r"[a-z]{1,4}", tok):
            normalized.append(tok.upper())
        else:
            normalized.append(tok.capitalize())

    return " ".join(normalized)


def _canonicalize_title(value: str) -> str:
    title = _normalize_text(value)
    return re.sub(r"^\s*(?:job|role|position)\s*[:\-]\s*", "", title, flags=re.I)


def _canonicalize_experience(value: str) -> str:
    text = _normalize_text(value)
    return text


def _canonicalize_education(value: str) -> str:
    text = _normalize_text(value)
    lowered = text.lower()
    if "bachelor" in lowered:
        return "Bachelor's degree"
    if "master" in lowered:
        return "Master's degree"
    if "phd" in lowered or "ph.d" in lowered:
        return "PhD"
    if "degree" in lowered:
        return "Degree"
    return text


def _canonicalize_field_of_study(value: str) -> str:
    text = _normalize_text(value)
    lowered = text.lower()
    for pattern in [
        r"\bcomputer science\b",
        r"\bdata science\b",
        r"\bartificial intelligence\b",
        r"\bstatistics\b",
        r"\bmathematics\b",
        r"\bsoftware engineering\b",
    ]:
        if re.search(pattern, lowered):
            return re.sub(r"\s+", " ", text).strip()
    return text


def _extract_title_fallback(sections: Dict[str, str]) -> List[str]:
    header = sections.get("header", "")
    lines = [line.strip() for line in header.splitlines() if line.strip()]
    if not lines:
        return []

    first_line = lines[0]
    if len(first_line) <= 120:
        return [_canonicalize_title(first_line)]

    return []


def _extract_experience_fallback(sections: Dict[str, str]) -> List[str]:
    text = "\n".join(
        [sections.get("requirements", ""), sections.get("job_meta", ""), sections.get("experience", "")]
    )
    if not text:
        return []

    matches = re.findall(r"(?:\b|^)(?:0\s*[\-–]\s*2|[0-9]+(?:\s*[+]|\s*\+\s*)?)(?:\s*years?)", text, flags=re.I)
    if matches:
        return [_normalize_text(matches[0])]

    return []


def _extract_skills_fallback(sections: Dict[str, str]) -> List[str]:
    text = sections.get("skills", "")
    if not text:
        return []

    skill_candidates = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 80:
            continue
        cleaned = _canonicalize_skill(line)
        if cleaned:
            skill_candidates.append(cleaned)
    return sorted(set(skill_candidates))


def _extract_soft_skills_fallback(sections: Dict[str, str]) -> List[str]:
    text = sections.get("soft_skills", "")
    if not text:
        return []

    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 80:
            continue
        line = re.sub(r"^[-•*]\s*", "", line)
        if not line:
            continue
        cleaned = re.sub(r"\s+", " ", line)
        cleaned = re.sub(r"\b(?:skill|skills|ability|abilities)\b", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,-")
        if cleaned:
            values.append(cleaned)
    return sorted(set(values))


def _extract_nice_to_have_skills_fallback(sections: Dict[str, str]) -> List[str]:
    text = sections.get("nice_to_have", "")
    if not text:
        return []

    values = []
    for raw in re.split(r"[\n,;]+", text):
        line = raw.strip()
        if not line or len(line) > 80:
            continue
        line = re.sub(r"^[-•*]\s*", "", line)
        cleaned = _canonicalize_skill(line)
        if cleaned:
            values.append(cleaned)
    return sorted(set(values))


def _extract_location_fallback(sections: Dict[str, str]) -> List[str]:
    text = sections.get("job_meta", "")
    if not text:
        return []

    match = re.search(r"location\s*[:\-]\s*(.+)", text, flags=re.I)
    if match:
        location = re.sub(r"\s+", " ", match.group(1)).strip()
        return [location] if location else []

    location_candidates = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r"\b(remote|hybrid|onsite|on-site|office|london|new york|berlin|dublin|paris|remote[- ]?[a-z]+)\b", line, flags=re.I):
            location_candidates.append(re.sub(r"\s+", " ", line).strip())
    return sorted(set(location_candidates))


def _extract_education_fallback(sections: Dict[str, str]) -> List[str]:
    text = sections.get("requirements", "") + "\n" + sections.get("education", "")
    if not text:
        return []

    degree_matches = []
    if re.search(r"\bbachelor(?:'s)?\s+degree\b", text, re.I):
        degree_matches.append("Bachelor's degree")
    elif re.search(r"\bbachelor\b", text, re.I):
        degree_matches.append("Bachelor's degree")
    if re.search(r"\bmaster(?:'s)?\s+degree\b", text, re.I):
        degree_matches.append("Master's degree")
    elif re.search(r"\bmaster\b", text, re.I):
        degree_matches.append("Master's degree")
    if re.search(r"\bphd\b|\bph\.d\b", text, re.I):
        degree_matches.append("PhD")
    if not degree_matches and re.search(r"\bdegree\b", text, re.I):
        degree_matches.append("Degree")
    return degree_matches


def _extract_field_of_study_fallback(sections: Dict[str, str]) -> List[str]:
    text = sections.get("requirements", "") + "\n" + sections.get("education", "")
    if not text:
        return []

    studies = []
    for pattern in [
        r"\bcomputer science\b",
        r"\bdata science\b",
        r"\bartificial intelligence\b",
        r"\bstatistics\b",
        r"\bmathematics\b",
        r"\bsoftware engineering\b",
    ]:
        if re.search(pattern, text, re.I):
            studies.append(re.search(pattern, text, re.I).group(0).strip())
    return studies


def _extract_job_type_fallback(sections: Dict[str, str]) -> List[str]:
    job_type = _job_type_fallback(sections)
    if job_type:
        return job_type

    meta = sections.get("job_meta", "")
    if re.search(r"\b(full[- ]time|part[- ]time|contract|internship|remote|hybrid)\b", meta, re.I):
        return [re.search(r"\b(full[- ]time|part[- ]time|contract|internship|remote|hybrid)\b", meta, re.I).group(0).lower()]
    return []


def _extract_benefits_fallback(sections: Dict[str, str]) -> List[str]:
    """Benefits/perks aren't in GLiNER's label set at all (they're not
    something a candidate's CV can be matched against), so this is the
    only source of the "benefits" field - a plain, deterministic split
    of the "benefits" section into one entry per bullet/line, cleaned up
    the same light way as the other list fallbacks."""
    text = sections.get("benefits", "")
    if not text:
        return []

    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 200:
            continue
        line = re.sub(r"^[-•*]\s*", "", line)
        cleaned = re.sub(r"\s+", " ", line).strip(" .")
        if cleaned:
            values.append(cleaned)
    return sorted(set(values))


def _bucket_entities(entities: List[Dict], ner_input: str, sections: Dict[str, str]) -> Dict[str, List[str]]:
    result = {key: [] for key in LABEL_KEY_MAP.values()}
    result["benefits"] = []
    for entity in entities:
        key = LABEL_KEY_MAP.get(entity.get("label"))
        if not key:
            continue
        text_value = entity["text"].strip()
        if key in ("required hard skill", "required soft skill") and _is_actually_optional(
            entity["text"], ner_input, sections
        ):
            key = "nice_to_have_skill"
        if key == "required hard skill" and _came_from_soft_skills_section(entity["text"], sections):
            key = "required soft skill"
        if key == "required hard skill":
            text_value = _canonicalize_skill(text_value)
            if not text_value or _looks_like_soft_skill(text_value):
                continue
        elif key == "required education degree":
            text_value = _canonicalize_education(text_value)
        elif key == "required field of study":
            text_value = _canonicalize_field_of_study(text_value)
        elif key == "required job title":
            text_value = _canonicalize_title(text_value)
        elif key == "required years of experience":
            text_value = _canonicalize_experience(text_value)
        result[key].append(text_value)

    # Deterministic fallbacks when GLiNER misses obvious JD fields.
    if not result["required job title"]:
        result["required job title"].extend(_extract_title_fallback(sections))
    if not result["required years of experience"]:
        result["required years of experience"].extend(_extract_experience_fallback(sections))
    if not result["required hard skill"]:
        result["required hard skill"].extend(_extract_skills_fallback(sections))
    if not result["required soft skill"]:
        result["required soft skill"].extend(_extract_soft_skills_fallback(sections))
    if not result["nice_to_have_skill"]:
        result["nice_to_have_skill"].extend(_extract_nice_to_have_skills_fallback(sections))
    if not result["required education degree"]:
        result["required education degree"].extend(_extract_education_fallback(sections))
    if not result["required field of study"]:
        result["required field of study"].extend(_extract_field_of_study_fallback(sections))

    if not result["work_location"]:
        result["work_location"].extend(_extract_location_fallback(sections))
    result["job_type"].extend(_extract_job_type_fallback(sections))
    if not result["benefits"]:
        result["benefits"].extend(_extract_benefits_fallback(sections))

    for key in result:
        result[key] = sorted(set(result[key]))

    return result


def _ner_text(jd_text: str, sections: Dict[str, str]) -> str:
    """
    Prefer the requirements/skills/experience/education sections for NER
    (denser signal, less boilerplate). Falls back to the full JD text if
    section-splitting didn't find anything useful (e.g. unstructured JD
    with no headers at all).

    "header" and "summary" are included on purpose (Fix 1): the job title
    almost always sits in the first line(s) of the posting or in the
    "About the role" blurb, not under "Requirements"/"Skills"/etc. Without
    them, GLiNER never sees the title text at all and "required job title"
    comes back empty on most real-world JDs.

    "nice_to_have" and "job_meta" are included on purpose (Fix 2): a
    "Preferred/Nice to Have" section is exactly where the "preferred
    technology tool listed as a plus or optional" label gets its
    signal, and a "job_meta" section (Employment Type, Experience Level,
    Location, ...) is exactly where "job type or work arrangement" and
    "city country or region" entities live. Leaving either out means
    GLiNER never sees that text at all.
    """
    focused = " \n".join(
        sections.get(key, "")
        for key in (
            "header",
            "summary",
            "requirements",
            "nice_to_have",
            "skills",
            "experience",
            "education",
            "languages",
            "job_meta",
        )
        if sections.get(key)
    )
    return focused if len(focused) > 50 else jd_text


# ---------------------------------------------------------------------------
# Per-section label routing
# ---------------------------------------------------------------------------
# Instead of concatenating every relevant section into one blob and asking
# GLiNER to search for all JD_LABELS in it, we run GLiNER once PER detected
# section, each time restricted to only the labels that section can
# plausibly contain. A JD with 7 recognized sections means (up to) 7
# separate GLiNER calls instead of 1 - each scoped to a small piece of text
# and a small label set, so the model isn't asked to look for e.g. "field
# of study" inside the "Employment Type / Location" blurb, and 2-3 labels
# is a much easier zero-shot task than all 10 at once.
#
# A section that's empty (or wasn't detected at all in this JD) is simply
# skipped - no call is made for it.
SECTION_LABEL_MAP: Dict[str, List[str]] = {
    "header": ["job title"],
    "summary": ["job title", "years of experience"],
    "requirements": [
        "years of experience",
        "programming language or technical skill",
        "soft skill or personality trait",
        "education degree",
        "field of study",
    ],
    "skills": [
        "programming language or technical skill",
    ],
    "soft_skills": [
        "soft skill or personality trait",
    ],
    "experience": [
        "years of experience",
        "programming language or technical skill",
        "soft skill or personality trait",
    ],
    "education": ["education degree", "field of study"],
    "languages": ["spoken or written language"],
    "nice_to_have": [
        "preferred technology tool listed as a plus or optional",
        "programming language or technical skill",
        "soft skill or personality trait",
    ],
    "job_meta": [
        "city country or region where job is based",
        "job type or work arrangement",
    ],
}


def _run_ner_per_section(ner_model, sections: Dict[str, str], threshold: float) -> List[Dict]:
    """Run GLiNER once per non-empty section that appears in
    SECTION_LABEL_MAP, each call scoped to only that section's relevant
    labels. Returns the combined entity list, exactly like one big call
    would have - `_bucket_entities` doesn't need to know sectioning
    happened."""
    entities: List[Dict] = []
    for section_key, labels in SECTION_LABEL_MAP.items():
        section_text = sections.get(section_key, "")
        if not section_text.strip():
            continue
        entities.extend(
            ner_model.predict_entities(
                section_text,
                labels,
                threshold=threshold,
                flat_ner=True,
            )
        )
    return entities


def _run_ner(jd_text: str, threshold: float):
    """Shared by extract_from_jd / extract_from_jd_with_sections: clean,
    split, run GLiNER, return everything the bucketing step needs.

    If the section splitter found real section boundaries beyond just
    "header" (i.e. this looks like a properly structured JD), GLiNER runs
    once PER section with a section-specific label subset - see
    SECTION_LABEL_MAP and _run_ner_per_section.

    Otherwise (a short or unstructured JD where everything landed in the
    catch-all "header" bucket, with no real headers to split on) we fall
    back to the original single whole-text pass over every label - this is
    deliberately kept as-is since it's the behavior that already performs
    well on short JDs.
    """
    jd_text = clean_jd_text(jd_text)
    sections = split_jd_sections(jd_text)
    ner_input = _ner_text(jd_text, sections)

    non_header_content_len = sum(
        len(text) for key, text in sections.items() if key != "header"
    )

    ner_model = ModelRegistry.ner()

    if non_header_content_len > 50:
        entities = _run_ner_per_section(ner_model, sections, threshold)
    else:
        entities = ner_model.predict_entities(
            ner_input,
            JD_LABELS,
            threshold=threshold,
            flat_ner=True,
        )

    return entities, ner_input, sections


def extract_from_jd(jd_text: str, threshold: float = 0.3) -> Dict[str, List[str]]:
    """
    Split the JD into sections, run GLiNER over the requirements-relevant
    sections, and bucket entities by canonical label.

    Returns a dict keyed by LABEL_KEY_MAP values, e.g.:
        {
          "required hard skill": ["python", "tensorflow", ...],
          "required years of experience": ["5 years"],
          ...
        }
    """
    entities, ner_input, sections = _run_ner(jd_text, threshold)
    return _bucket_entities(entities, ner_input, sections)


def extract_from_jd_with_sections(jd_text: str, threshold: float = 0.3) -> Dict:
    """
    Same as extract_from_jd, but also returns the section split — used by
    routers/jobs.py so the sections (esp. "requirements"/"skills") can be
    persisted and reused for the embedding text (build_jd_query) and for
    the reranker's JD-side text later.
    """
    entities, ner_input, sections = _run_ner(jd_text, threshold)
    result = _bucket_entities(entities, ner_input, sections)
    return {"extracted": result, "sections": sections}


def build_jd_query(jd_extracted: Dict[str, List[str]], sections: Dict[str, str] = None) -> str:
    """
    Build a focused query string from JD extracted fields (+ raw
    skills/requirements section text, if available) for embedding.
    """
    parts = []
    if jd_extracted.get("required hard skill"):
        parts.append("Skills: " + ", ".join(jd_extracted["required hard skill"]))
    if jd_extracted.get("required job title"):
        parts.append("Role: " + ", ".join(jd_extracted["required job title"]))
    if jd_extracted.get("required years of experience"):
        parts.append("Experience: " + ", ".join(jd_extracted["required years of experience"]))
    if jd_extracted.get("required education degree"):
        parts.append("Education: " + ", ".join(jd_extracted["required education degree"]))

    if sections:
        if sections.get("requirements"):
            parts.append("Requirements: " + sections["requirements"])
        elif sections.get("skills"):
            parts.append("Requirements: " + sections["skills"])

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Extraction v2: router (legacy vs. LLM) + confidence-gated structured output
# ---------------------------------------------------------------------------
# This is the new primary entry point. It does NOT replace the functions
# above - extract_from_jd / extract_from_jd_with_sections stay exactly as
# they are, both because existing callers/tests depend on their exact
# shape and because they ARE still used, as the "legacy" arm of the
# router, for JDs simple enough not to need an LLM call.
from pipeline import jd_router
from pipeline import jd_llm_extractor
from pipeline import jd_keyword_extractor
from pipeline.schemas import ExtractedField, ExtractedListField, JDExtraction


def _legacy_dict_to_jd_extraction(
    legacy_result: Dict[str, List[str]], routing_reason: str
) -> JDExtraction:
    """Wrap the existing extract_from_jd() output in the v2 structured
    shape. Legacy extraction has no native confidence signal, so each
    present value gets a flat, moderate confidence (0.6) - not 0.0 -
    so a JD that legitimately routed to the cheap legacy path isn't
    unfairly flagged needs_review just for using the cheaper path.
    0.0 is reserved for "the LLM path ran and was genuinely unsure."""
    def _field(key: str) -> ExtractedField:
        values = legacy_result.get(key) or []
        return ExtractedField(value=values[0] if values else None, confidence=0.6 if values else 0.0)

    def _list_field(key: str) -> ExtractedListField:
        values = legacy_result.get(key) or []
        return ExtractedListField(value=list(values), confidence=0.6 if values else 0.0)

    return JDExtraction(
        job_title=_field("required job title"),
        years_experience=_field("required years of experience"),
        hard_skills=_list_field("required hard skill"),
        soft_skills=_list_field("required soft skill"),
        nice_to_have_skills=_list_field("nice_to_have_skill"),
        education_degree=_field("required education degree"),
        field_of_study=_field("required field of study"),
        languages=_list_field("required spoken language"),
        work_location=_field("work_location"),
        job_type=_field("job_type"),
        benefits=_list_field("benefits"),
        extraction_method="legacy_regex",
        routing_reason=routing_reason,
    )


_SCALAR_FIELD_NAMES = (
    "job_title", "years_experience", "education_degree",
    "field_of_study", "work_location", "job_type",
)
_LIST_FIELD_NAMES = (
    "hard_skills", "soft_skills", "nice_to_have_skills", "languages", "benefits",
)


def _run_legacy_safely(jd_text: str, threshold: float, decision_reason: str):
    """extract_from_jd() depends on the GLiNER model actually loading
    (see pipeline/models.py + helpers/config.py's GLINER_ONNX_DIR) -
    that can fail on any machine where the configured model path isn't
    present. Previously this exception was never caught here, so a
    misconfigured/missing model crashed the whole job-creation request
    (500) instead of degrading gracefully. Returns (None, reason) on
    failure so the caller can fall through to the keyword extractor
    instead of failing the request."""
    try:
        legacy_result = extract_from_jd(jd_text, threshold=threshold)
        return (
            _legacy_dict_to_jd_extraction(legacy_result, routing_reason=decision_reason),
            decision_reason,
        )
    except Exception as exc:
        logger.warning("Legacy (GLiNER) JD extraction failed (%s); falling back further", exc)
        return None, f"{decision_reason}; legacy path failed ({exc})"


def _merge_with_keyword_fallback(result, jd_text: str, reason: str) -> JDExtraction:
    """Layer the dependency-free keyword extractor (pipeline/
    jd_keyword_extractor.py) on top of whatever the primary path
    produced: fills any scalar field that came back empty, and unions
    any list field, instead of shipping a result that's emptier than
    plain keyword matching against the JD's own text would be. This is
    what actually fixes JDs that route to the LLM path (unavailable, no
    API key) and then also hit a GLiNER load/predict failure - without
    this, such a JD would previously come back with every field
    "none detected" even when the information is plainly in the text.

    Cheap and dependency-free (pure regex), so it's safe to always run,
    not just when the primary path fails outright.
    """
    keyword_result = jd_keyword_extractor.extract_from_jd_keywords(jd_text)

    if result is None:
        keyword_result.routing_reason = f"{reason}; used keyword_fallback (primary paths unavailable)"
        return keyword_result

    filled_any = False
    for name in _SCALAR_FIELD_NAMES:
        primary_field: ExtractedField = getattr(result, name)
        fallback_field: ExtractedField = getattr(keyword_result, name)
        if not primary_field.value and fallback_field.value:
            setattr(result, name, fallback_field)
            filled_any = True

    for name in _LIST_FIELD_NAMES:
        primary_field: ExtractedListField = getattr(result, name)
        fallback_field: ExtractedListField = getattr(keyword_result, name)
        if fallback_field.value:
            merged = sorted(set(primary_field.value) | set(fallback_field.value))
            if merged != sorted(primary_field.value):
                filled_any = True
            setattr(result, name, ExtractedListField(
                value=merged,
                confidence=max(primary_field.confidence, fallback_field.confidence),
            ))

    if filled_any:
        result.extraction_method = f"{result.extraction_method}+keyword_fallback"
        result.routing_reason = f"{result.routing_reason or reason}; gaps filled by keyword_fallback"

    return result


def extract_from_jd_v2(jd_text: str, threshold: float = 0.3) -> JDExtraction:
    """
    Primary v2 entry point: routes each JD to the legacy pipeline or the
    structured-output LLM pipeline (pipeline/jd_router.py), runs it, and
    returns a confidence-scored JDExtraction.

    Never raises for extraction-quality reasons:
      - LLM path selected but unavailable (no API key) or fails outright
        (network error, malformed response) -> falls back to the legacy
        (GLiNER) path.
      - Legacy path's GLiNER model fails to load/predict (e.g. a
        misconfigured model path - see helpers/config.py) -> falls back
        to the dependency-free keyword extractor.
      - In every case, the dependency-free keyword extractor
        (pipeline/jd_keyword_extractor.py) is also layered on top to
        fill any field the primary path left empty, so a well-formed JD
        never comes back with every field blank just because the ML
        models/API happen to be unavailable in this deployment.

    A JD that comes back with several low-confidence fields is flagged
    needs_review instead of shipped silently as if it were fully
    trustworthy.
    """
    settings = None
    try:
        from helpers.config import get_settings
        settings = get_settings()
    except Exception:
        pass

    mode = getattr(settings, "JD_EXTRACTION_MODE", "auto") if settings else "auto"
    max_legacy_chars = getattr(settings, "JD_ROUTER_MAX_LEGACY_CHARS", 1500) if settings else 1500
    min_headers = getattr(settings, "JD_ROUTER_MIN_RECOGNIZED_HEADERS", 2) if settings else 2
    min_confidence = getattr(settings, "JD_MIN_CONFIDENCE", 0.5) if settings else 0.5
    max_low_conf_fields = getattr(settings, "JD_MAX_LOW_CONFIDENCE_FIELDS_BEFORE_REVIEW", 2) if settings else 2

    if mode == "legacy":
        decision_path, decision_reason = "legacy", "JD_EXTRACTION_MODE=legacy (forced)"
    elif mode == "llm":
        decision_path, decision_reason = "llm", "JD_EXTRACTION_MODE=llm (forced)"
    else:
        decision = jd_router.choose_extraction_path(
            jd_text,
            max_legacy_chars=max_legacy_chars,
            min_recognized_headers=min_headers,
        )
        decision_path, decision_reason = decision.path, decision.reason

    result = None
    reason = decision_reason

    if decision_path == "llm":
        try:
            result = jd_llm_extractor.extract_from_jd_llm(jd_text)
            result.routing_reason = decision_reason
        except jd_llm_extractor.JDExtractionError as exc:
            logger.warning("LLM JD extraction failed (%s); falling back to legacy path", exc)
            reason = f"{decision_reason}; LLM path failed ({exc}), used legacy fallback"
            result, reason = _run_legacy_safely(jd_text, threshold, reason)
    else:
        result, reason = _run_legacy_safely(jd_text, threshold, decision_reason)

    result = _merge_with_keyword_fallback(result, jd_text, reason)

    low_conf_count = result.low_confidence_field_count(min_confidence)
    if low_conf_count > max_low_conf_fields or not result.job_title.value:
        result.needs_review = True

    return result


def extract_from_jd_with_sections_v2(jd_text: str, threshold: float = 0.3) -> Dict:
    """
    v2 counterpart to extract_from_jd_with_sections: returns both the
    structured JDExtraction (projected to the legacy flat shape via
    to_legacy_dict(), for drop-in compatibility with storage/
    build_jd_query/matcher) AND the full JDExtraction (with confidence +
    needs_review) for anything that wants the richer v2 view - e.g. the
    ATS console flagging JDs for human review.
    """
    jd_extraction = extract_from_jd_v2(jd_text, threshold=threshold)

    # Sections are still useful for build_jd_query's raw-text embedding
    # input regardless of which extraction path ran, so compute them the
    # same (cheap, deterministic) way as before.
    cleaned = clean_jd_text(jd_text)
    sections = split_jd_sections(cleaned)

    return {
        "extracted": jd_extraction.to_legacy_dict(),
        "sections": sections,
        "jd_extraction": jd_extraction,
    }
