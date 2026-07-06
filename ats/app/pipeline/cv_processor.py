"""
Step 2 of the pipeline: split a (already cleaned) CV text into sections and
pull out structured facts with regex. This is the same logic as the
notebook's `split_cv_sections` / field extractors, applied to the text
already produced by preprocessing.dispatcher / preprocessing.cleaner.
"""

import datetime
import re
from typing import Dict, List

SECTION_PATTERNS = {
    "experience": r"(work\s+)?experience|employment(\s+history)?|career\s+history|professional\s+background",
    "education": r"education(al\s+background)?|academic(\s+background)?|qualifications?|degrees?",
    "skills": r"(technical\s+)?skills?|competenc(y|ies)|technologies|tools?",
    "languages": r"languages?|spoken\s+languages?",
    "summary": r"summary|objective|profile|about",
}

DEGREE_PATTERN = r"\b(b\.?sc?|m\.?sc?|ph\.?d|bachelor|master|doctorate|b\.?eng?|m\.?eng?|mba)\b"


def split_cv_sections(text: str) -> Dict[str, str]:
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {}
    current = "header"
    sections[current] = []

    for line in lines:
        stripped = line.strip()
        matched = False
        for section, pattern in SECTION_PATTERNS.items():
            header_match = re.match(r"^(" + pattern + r")\s*(?::\s*(.*))?$", stripped.lower())
            if header_match and len(stripped) < 60:
                current = section
                sections[current] = []
                inline_content = header_match.group(2)
                if inline_content and inline_content.strip():
                    sections[current].append(inline_content.strip())
                matched = True
                break
        if not matched:
            sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def extract_email(text: str):
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    return match.group(0) if match else None


def extract_phone(text: str):
    match = re.search(r"[\+\d][\d\s\-().]{7,15}", text)
    return match.group(0).strip() if match else None


def extract_years_of_experience(text: str) -> int:
    pattern = r"(\d{4})\s*(?:to|–|-|—)\s*(\d{4}|present|current|now)"
    matches = re.findall(pattern, text.lower())
    total = 0
    current_year = datetime.datetime.now().year
    for start, end in matches:
        s = int(start)
        e = current_year if end in ("present", "current", "now") else int(end)
        total += max(0, e - s)
    return total


def extract_degrees(text: str) -> List[str]:
    matches = re.findall(DEGREE_PATTERN, text.lower())
    return sorted(set(matches))


def extract_skills_from_section(text: str) -> List[str]:
    body = re.sub(r"^.*skills?:?", "", text, flags=re.IGNORECASE | re.MULTILINE)
    items = re.split(r"[,|•·\n/]+", body)
    return sorted(set(s.strip().lower() for s in items if 1 < len(s.strip()) < 40))


def extract_languages(text: str) -> List[str]:
    items = re.split(r"[,|•·\n/]+", text)
    return sorted(set(s.strip().lower() for s in items if 1 < len(s.strip()) < 30))


def extract_inline_languages(text: str) -> List[str]:
    match = re.search(r"languages?\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        items = re.split(r"[,|]+", match.group(1))
        return [s.strip().lower() for s in items if s.strip()]
    return []


def parse_cv(cv_id: str, text: str) -> Dict:
    """
    Full CV parser: splits sections, extracts structured facts.
    `text` should already be cleaned (preprocessing.cleaner.clean_resume_text).
    """
    sections = split_cv_sections(text)

    langs = extract_languages(sections.get("languages", ""))
    if not langs:
        langs = extract_inline_languages(text)

    return {
        "id": cv_id,
        "email": extract_email(text),
        "phone": extract_phone(text),
        "years_of_experience": extract_years_of_experience(sections.get("experience", "")),
        "degrees": extract_degrees(sections.get("education", "")),
        "skills": extract_skills_from_section(sections.get("skills", "")),
        "languages": langs,
        "sections": sections,
        "full_text": text,
    }


def build_cv_text(cv_parsed: Dict) -> str:
    """Combine relevant CV sections into a single string for embedding."""
    sections = cv_parsed["sections"]
    parts = [
        sections.get("skills", ""),
        sections.get("experience", ""),
        sections.get("education", ""),
    ]
    return " ".join(p for p in parts if p)
