"""
Deterministic keyword/vocabulary extraction path - the "always works"
safety net.

Why this exists
----------------
Both other extraction paths have an external dependency that can be
unavailable in a given deployment:

  - jd_llm_extractor needs ANTHROPIC_API_KEY + network access.
  - the legacy path (jd_processor.extract_from_jd) needs the GLiNER
    model weights to actually load from disk (see helpers/config.py's
    GLINER_ONNX_DIR). If that path is wrong/missing/unreadable on a
    given machine, GLiNER produces zero entities and every field comes
    back empty - even though the JD text itself is perfectly good,
    well-structured English with the information sitting right there.

The legacy path already has *some* deterministic fallbacks (see the
`_extract_*_fallback` functions in jd_processor.py), but they only read
from a handful of narrowly-named sub-sections (`sections["skills"]`,
`sections["soft_skills"]`, `sections["education"]`, ...). Plenty of
real JDs - like the one that motivated this module - put everything
under one "Requirements" block with no separate "Skills:"/"Education:"
sub-header at all, so those fallbacks find nothing even though the
content is right there in plain text.

This module fixes that by matching curated vocabularies/regexes
directly against the *whole* cleaned JD text (falling back to specific
sections only when it sharpens precision, e.g. benefits/location), so
it doesn't depend on the JD happening to use a particular header
vocabulary at all. It has zero ML/network dependency, so it can be
used:

  - as a gap-filler layered on top of whatever the GLiNER or LLM path
    returned (union list fields, fill empty scalar fields) - this is
    the default wiring, see jd_processor.extract_from_jd_v2
  - standalone, if both other paths are unavailable/fail outright, so
    a job is never persisted with every field empty when the JD text
    plainly contains the information
"""

import re
from typing import Dict, List, Optional, Tuple

from pipeline.jd_chunker import clean_jd_text, split_jd_sections
from pipeline.schemas import ExtractedField, ExtractedListField, JDExtraction

# A flat, moderate confidence for anything this module finds: it's not a
# guess (every value here is a literal regex/vocab hit against the JD's
# own text), but it's not a comprehension-based read either, so it's
# scored a notch below the LLM path (which reasons about phrasing) and
# roughly level with the legacy path's own fallback confidence (0.6).
_CONFIDENCE = 0.55

# ---------------------------------------------------------------------------
# Hard skills vocabulary
# ---------------------------------------------------------------------------
_HARD_SKILL_NAMES = [
    # Languages
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Golang",
    "Rust", "Ruby", "PHP", "Kotlin", "Swift", "Scala", "R", "SQL", "Bash",
    # Web / backend frameworks
    "FastAPI", "Flask", "Django", "Node.js", "Express", "React", "Vue",
    "Angular", "Next.js", "Spring Boot", "Spring", ".NET",
    # ML / AI
    "PyTorch", "TensorFlow", "Keras", "ONNX", "OpenVINO", "onnxruntime-gpu",
    "onnxruntime", "scikit-learn", "Hugging Face", "Transformers", "PEFT",
    "LoRA", "bitsandbytes", "CUDA", "NLP", "LLM", "Pandas", "NumPy",
    "XGBoost", "MLflow",
    # Infra / cloud / devops
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform", "CI/CD",
    "Jenkins", "GitHub Actions", "GitLab CI", "Ansible", "Linux",
    # APIs / data
    "REST", "GraphQL", "gRPC", "Kafka", "RabbitMQ", "Redis", "PostgreSQL",
    "MySQL", "MongoDB", "Elasticsearch", "Spark", "Airflow",
    # Version control
    "Git", "GitHub", "GitLab",
]


def _compile_vocab(names: List[str]) -> "re.Pattern":
    """One alternation regex for a whole vocabulary, longest names first
    (so 'Node.js' wins over any shorter overlapping token), each hit
    bounded by non-alnum lookaround instead of \\b - \\b alone is
    unreliable for tokens containing punctuation like 'C++', '.NET',
    'CI/CD'."""
    escaped = sorted((re.escape(n) for n in names), key=len, reverse=True)
    pattern = r"(?<![A-Za-z0-9])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9])"
    return re.compile(pattern, re.IGNORECASE)


_HARD_SKILL_PATTERN = _compile_vocab(_HARD_SKILL_NAMES)
_HARD_SKILL_CANON = {n.lower(): n for n in _HARD_SKILL_NAMES}

# ---------------------------------------------------------------------------
# Soft skills vocabulary (phrase -> canonical display form)
# ---------------------------------------------------------------------------
_SOFT_SKILL_PHRASES: List[Tuple[str, str]] = [
    (r"curiosity", "Curiosity"),
    (r"collaborative mindset", "Collaborative mindset"),
    (r"structured and action[\s-]oriented", "Structured and action-oriented"),
    (r"action[\s-]oriented", "Action-oriented"),
    (r"comfortable (?:working )?(?:in )?ambiguity", "Comfortable with ambiguity"),
    (r"systems? thinking", "Systems thinking"),
    (r"problem[\s-]solving", "Problem solving"),
    (r"adaptability", "Adaptability"),
    (r"communication and collaboration", "Communication and collaboration"),
    (r"communication skills?", "Communication"),
    (r"\bcollaboration\b", "Collaboration"),
    (r"teamwork", "Teamwork"),
    (r"leadership", "Leadership"),
    (r"attention to detail", "Attention to detail"),
    (r"critical thinking", "Critical thinking"),
    (r"time management", "Time management"),
    (r"\bownership\b", "Ownership"),
    (r"accountability", "Accountability"),
    (r"interpersonal skills?", "Interpersonal skills"),
    (r"analytical (?:skills?|mindset)", "Analytical skills"),
    (r"willingness to learn", "Willingness to learn"),
    (r"\bempathy\b", "Empathy"),
    (r"focus on impact", "Focus on impact"),
    (r"resilien\w*", "Resilience"),
]

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
_LANGUAGE_NAMES = [
    "English", "Portuguese", "Spanish", "French", "German", "Italian",
    "Mandarin", "Chinese", "Arabic", "Japanese", "Korean", "Russian",
    "Hindi", "Dutch", "Polish", "Turkish",
]
_PROFICIENCY_RE = re.compile(
    r"\b(native|fluent|advanced|intermediate|basic|conversational|proficient)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------
_DEGREE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bph\.?d\.?\b", "PhD"),
    (r"\bdoctorate\b", "Doctorate"),
    (r"\bmaster'?s?\s*(?:degree)?\b", "Master's degree"),
    (r"\bmsc\b", "Master's degree"),
    (r"\bbachelor'?s?\s*(?:degree)?\b", "Bachelor's degree"),
    (r"\bbsc\b", "Bachelor's degree"),
    (r"\bassociate'?s?\s+degree\b", "Associate degree"),
    (r"\bdiploma\b", "Diploma"),
]

# Deliberately full phrases only (no bare "ai"/"ml" tokens) - a bare
# 2-letter acronym match against generic prose ("AI products", "AI
# services") produced exactly the kind of false positive this module
# exists to avoid (see module docstring / the bug this was built for).
_FIELD_OF_STUDY_PHRASES = [
    "computer science", "data science", "artificial intelligence",
    "machine learning", "software engineering", "electrical engineering",
    "information technology", "information systems", "statistics",
    "applied mathematics", "mathematics", "physics", "engineering",
]

# ---------------------------------------------------------------------------
# Job type / work arrangement
# ---------------------------------------------------------------------------
_JOB_TYPE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bfull[\s-]?time\b", "Full-time"),
    (r"\bpart[\s-]?time\b", "Part-time"),
    (r"\bfreelance\b", "Freelance"),
    (r"(?<!employment )\bcontract(?:or)?\b", "Contract"),
    (r"\bintern(?:ship)?\b", "Internship"),
    (r"\btemp(?:orary)?\b", "Temporary"),
    (r"\bpermanent\b", "Permanent"),
    (r"\bclt\b", "CLT (Brazilian formal employment contract)"),
    (r"\bpj\b", "PJ (Brazilian contractor arrangement)"),
    (r"\bhybrid\b", "Hybrid"),
    (r"\bon[\s-]?site\b", "On-site"),
    (r"\bremote\b", "Remote"),
]

# ---------------------------------------------------------------------------
# Years of experience
# ---------------------------------------------------------------------------
_YEARS_PATTERN = re.compile(
    r"\b\d{1,2}\s*\+?\s*(?:-|to|–)\s*\d{1,2}\s*\+?\s*years?\b"
    r"|\b\d{1,2}\s*\+\s*years?\b"
    r"|\b\d{1,2}\s*years?\s+of\s+experience\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Job title
# ---------------------------------------------------------------------------
_TITLE_ROLE_WORDS = (
    r"engineer|developer|scientist|manager|analyst|designer|architect|"
    r"specialist|intern|consultant|administrator|director|researcher|"
    r"product owner"
)
_TITLE_LINE_RE = re.compile(
    r"^(?:[A-Z][\w./+#&-]*\s*){1,6}(?:" + _TITLE_ROLE_WORDS + r")s?\b.{0,20}$",
    re.IGNORECASE,
)
_TITLE_PHRASE_RE = re.compile(
    r"(?:looking for|hiring|seeking|as an?|for an?|role of|position of)\s+an?\s+"
    r"([A-Z][\w./+#&\s-]{2,60}?(?:" + _TITLE_ROLE_WORDS + r")s?)\b",
    re.IGNORECASE,
)

# Optional-marker words that flag a bullet/clause as "nice to have"
# rather than strictly required, mirroring jd_processor's own
# _OPTIONAL_INLINE_MARKER vocabulary so the two stay conceptually
# aligned even though this module never imports from jd_processor
# (kept independent on purpose - see module docstring).
_OPTIONAL_MARKER_RE = re.compile(
    r"\b(preferred|optional|nice[\s-]to[\s-]have|bonus|a\s+plus|desired|"
    r"previous experience|familiarity with)\b",
    re.IGNORECASE,
)


def _find_skills(text: str) -> List[str]:
    hits = {m.group(0).lower() for m in _HARD_SKILL_PATTERN.finditer(text)}
    return sorted(_HARD_SKILL_CANON[h] for h in hits)


def _split_optional_vs_required(sections: Dict[str, str]) -> Tuple[str, str]:
    """Split the requirements-bearing text into (required_text,
    optional_text) line by line, using inline optional-marker words as
    the signal - this is the closest a header-free heuristic can get to
    the LLM path's "required vs nice-to-have" judgement, without a
    model call. Lines with no marker are treated as required (the
    conservative default: a plain requirements bullet is required)."""
    source = "\n".join(
        sections.get(key, "")
        for key in ("requirements", "nice_to_have", "skills", "experience")
        if sections.get(key)
    )
    required_lines, optional_lines = [], []
    for line in source.splitlines():
        if not line.strip():
            continue
        if _OPTIONAL_MARKER_RE.search(line):
            optional_lines.append(line)
        else:
            required_lines.append(line)
    return "\n".join(required_lines), "\n".join(optional_lines)


def _find_soft_skills(text: str) -> List[str]:
    values = set()
    for pattern, canon in _SOFT_SKILL_PHRASES:
        if re.search(pattern, text, re.IGNORECASE):
            values.add(canon)
    return sorted(values)


def _find_languages(text: str) -> List[str]:
    values = []
    seen = set()
    for name in _LANGUAGE_NAMES:
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
            if name.lower() in seen:
                break
            window = text[max(0, m.start() - 25): m.end() + 25]
            prof = _PROFICIENCY_RE.search(window)
            label = f"{name} ({prof.group(1).capitalize()})" if prof else name
            values.append(label)
            seen.add(name.lower())
            break
    return sorted(values)


_EDUCATION_CONTEXT_RE = re.compile(
    r"\b(degree|bachelor|master|phd|ph\.d|doctorate|diploma|education|"
    r"academic|major(?:ed|ing)?|graduate)\b",
    re.IGNORECASE,
)


def _find_education(text: str) -> Tuple[Optional[str], Optional[str]]:
    degree = None
    degree_match_end = None
    for pattern, canon in _DEGREE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            degree = canon
            degree_match_end = m.end()
            break

    # field_of_study is only meaningful paired with an education context
    # ("Bachelor's in Computer Science") - matching a phrase like
    # "machine learning" or "AI" anywhere in a tech JD's prose (e.g. "AI
    # products", "machine learning workloads") produces exactly the kind
    # of false positive this module exists to avoid (see module
    # docstring). Only search a window right after a degree match, or
    # near an explicit education-context word - never the whole document
    # unscoped.
    field_of_study = None
    if degree_match_end is not None:
        window = text[degree_match_end: degree_match_end + 80]
        for phrase in _FIELD_OF_STUDY_PHRASES:
            if re.search(r"\b" + re.escape(phrase) + r"\b", window, re.IGNORECASE):
                field_of_study = phrase.title()
                break
    if field_of_study is None:
        for phrase in _FIELD_OF_STUDY_PHRASES:
            for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", text, re.IGNORECASE):
                nearby = text[max(0, m.start() - 40): m.start()]
                if _EDUCATION_CONTEXT_RE.search(nearby):
                    field_of_study = phrase.title()
                    break
            if field_of_study:
                break

    return degree, field_of_study



def _find_years_experience(text: str) -> Optional[str]:
    m = _YEARS_PATTERN.search(text)
    return re.sub(r"\s+", " ", m.group(0)).strip() if m else None


def _find_job_type(text: str) -> List[str]:
    values = []
    seen = set()
    for pattern, canon in _JOB_TYPE_PATTERNS:
        if canon in seen:
            continue
        if re.search(pattern, text, re.IGNORECASE):
            values.append(canon)
            seen.add(canon)
    return values


def _find_work_location(text: str) -> Optional[str]:
    m = re.search(r"from anywhere in ([A-Z][A-Za-z\s]{2,40})", text)
    if m:
        return f"Remote ({m.group(1).strip()})"
    m = re.search(r"\bbased in ([A-Z][A-Za-z\s,]{2,40})", text)
    if m:
        return m.group(1).strip().rstrip(".,")
    if re.search(r"\bremote\b", text, re.IGNORECASE):
        return "Remote"
    if re.search(r"\bhybrid\b", text, re.IGNORECASE):
        return "Hybrid"
    if re.search(r"\bon[\s-]?site\b", text, re.IGNORECASE):
        return "On-site"
    return None


def _find_job_title(cleaned_text: str, sections: Dict[str, str]) -> Optional[str]:
    # 1) explicit "Job Title: X" / "Position: X" style line anywhere.
    m = re.search(
        r"(?:job\s*title|position|role|vacancy)\s*[:\-]\s*(.{3,80})",
        cleaned_text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().splitlines()[0].strip()

    # 2) a short line early in the doc that reads like a title.
    candidate_text = sections.get("header", "") or sections.get("summary", "")
    lines = [l.strip() for l in candidate_text.splitlines() if l.strip()]
    if not lines:
        lines = [l.strip() for l in cleaned_text.splitlines()[:5] if l.strip()]
    for line in lines[:5]:
        if len(line) <= 80 and _TITLE_LINE_RE.match(line):
            return line

    # 3) "looking for a <Title>" / "hiring a <Title>" phrasing anywhere.
    m = _TITLE_PHRASE_RE.search(cleaned_text)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()

    return None


def _find_benefits(sections: Dict[str, str], cleaned_text: str) -> List[str]:
    text = sections.get("benefits", "")
    values = []
    if text:
        for line in text.splitlines():
            line = line.strip()
            if not line or len(line) > 200:
                continue
            label = line.split(":", 1)[0].strip() if ":" in line[:60] else line
            cleaned = re.sub(r"\s+", " ", label).strip(" .")
            if cleaned:
                values.append(cleaned)
    return sorted(set(values))


def extract_from_jd_keywords(jd_text: str) -> JDExtraction:
    """Pure-Python, dependency-free extraction. Never raises for a
    non-empty jd_text - worst case, individual fields come back empty
    if the JD text genuinely doesn't contain that information."""
    cleaned = clean_jd_text(jd_text)
    sections = split_jd_sections(cleaned)

    required_text, optional_text = _split_optional_vs_required(sections)
    full_text = cleaned

    hard_skills = _find_skills(required_text) or _find_skills(full_text)
    nice_to_have_from_optional = _find_skills(optional_text)
    # Anything flagged optional shouldn't also sit in required.
    hard_skills = [s for s in hard_skills if s not in nice_to_have_from_optional]

    soft_skills = _find_soft_skills(full_text)
    languages = _find_languages(full_text)
    degree, field_of_study = _find_education(full_text)
    years_experience = _find_years_experience(full_text)
    job_type = _find_job_type(full_text)
    work_location = _find_work_location(full_text)
    job_title = _find_job_title(cleaned, sections)
    benefits = _find_benefits(sections, cleaned)

    def _scalar(value: Optional[str]) -> ExtractedField:
        return ExtractedField(value=value, confidence=_CONFIDENCE if value else 0.0)

    def _list(values: List[str]) -> ExtractedListField:
        return ExtractedListField(value=values, confidence=_CONFIDENCE if values else 0.0)

    return JDExtraction(
        job_title=_scalar(job_title),
        years_experience=_scalar(years_experience),
        hard_skills=_list(hard_skills),
        soft_skills=_list(soft_skills),
        nice_to_have_skills=_list(nice_to_have_from_optional),
        education_degree=_scalar(degree),
        field_of_study=_scalar(field_of_study),
        languages=_list(languages),
        work_location=_scalar(work_location),
        job_type=_scalar(", ".join(job_type) if job_type else None),
        benefits=_list(benefits),
        extraction_method="keyword_fallback",
    )
