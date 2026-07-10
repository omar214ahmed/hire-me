"""
Extraction-path router: decides, per JD, whether the cheap legacy
(regex-header + GLiNER) pipeline is trustworthy enough to use, or whether
the JD needs the comprehension-based LLM extractor.

Why this exists (see the architecture review this was built from): the
legacy pipeline's failure mode isn't random - it's *predictable*. It fails
when a JD:
  - has few or no headers the regex vocabulary recognizes
  - is short on structure but long on prose (few paragraph breaks relative
    to length)
  - is not (primarily) in English
  - is dense with emoji used as bullets/headers (a strong signal the
    header vocabulary won't match, since headers are matched as plain
    English phrases)

None of these signals require a model call to detect - they're the same
kind of cheap, explainable heuristic the legacy classifier already uses,
just applied one level up (to the whole document, to pick a *pipeline*,
not to a paragraph, to pick a *label*). That keeps routing itself fast
and free, so the cost of the LLM path is only paid on JDs that actually
need it.

This module makes no network calls and has no ML dependency - it is
pure Python so it's trivially unit-testable and never itself a source of
flakiness.
"""

import re
from dataclasses import dataclass
from typing import Optional

from pipeline.jd_chunker import JD_SECTION_PATTERNS, clean_jd_text

# A generous, high-recall set of common English function words. This is
# NOT a language detector - it's a fast, dependency-free proxy for "is
# this text predominantly English", which is all the router needs. A
# real language ID model (e.g. fastText lid.176) is a reasonable future
# upgrade behind the same `looks_single_language_english` function
# signature; nothing else in the router needs to change if swapped in.
_ENGLISH_STOPWORDS = {
    "the", "and", "you", "your", "we", "our", "for", "with", "have", "will",
    "are", "is", "to", "of", "in", "on", "a", "an", "as", "be", "this",
    "that", "at", "or", "role", "team", "experience", "years", "work",
    "skills", "required", "about",
}

# Rough heuristic emoji ranges (not exhaustive Unicode emoji coverage,
# but covers the overwhelming majority of emoji actually used as bullets/
# headers in job postings: pictographs, symbols, transport, supplemental).
_EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)

_HEADER_LINE_MAX_LEN = 60


@dataclass
class RoutingDecision:
    path: str              # "legacy" or "llm"
    reason: str             # human-readable explanation, for logging/debugging
    recognized_headers: int
    char_count: int
    emoji_count: int
    non_english_ratio: float


def _count_recognized_headers(text: str) -> int:
    """How many lines in the JD match the legacy pipeline's own header
    vocabulary. Reuses JD_SECTION_PATTERNS directly (not a re-implementation)
    so the router's notion of "has headers the legacy path understands" can
    never drift out of sync with what the legacy path actually does."""
    count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or len(stripped) >= _HEADER_LINE_MAX_LEN:
            continue
        lowered = stripped.lower()
        for pattern in JD_SECTION_PATTERNS.values():
            if re.match(r"^(?:" + pattern + r")\s*(?::.*)?$", lowered):
                count += 1
                break
    return count


def _emoji_count(text: str) -> int:
    return len(_EMOJI_PATTERN.findall(text))


def _non_english_word_ratio(text: str) -> float:
    """Fraction of alphabetic "words" that are NOT recognized English
    stopwords AND contain non-ASCII letters or simply never match common
    English function words at all across a long stretch of text. Cheap
    proxy: JDs overwhelmingly reuse a small set of English function words
    regardless of topic/company; if a long document has almost none of
    them, it's very likely not (primarily) English."""
    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    if len(words) < 20:
        return 0.0  # too short to judge; don't penalize
    stopword_hits = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
    return 1.0 - (stopword_hits / len(words))


def choose_extraction_path(
    jd_text: str,
    max_legacy_chars: int = 1500,
    min_recognized_headers: int = 2,
    non_english_ratio_threshold: float = 0.85,
    emoji_ratio_threshold: float = 0.01,
) -> RoutingDecision:
    """Return a RoutingDecision for this JD.

    "legacy" only when ALL of the following hold - each condition maps to
    one of the concrete failure modes in the architecture review, so a
    JD that would fail the legacy pipeline for a *known* reason never
    gets routed there:
      - short enough that a wrong regex classification affects little text
      - has enough recognized headers that section detection has signal
        to work with at all
      - reads as predominantly English (regex vocabulary is English-only)
      - not emoji-dense (header regex has no emoji awareness)
    """
    text = clean_jd_text(jd_text)
    char_count = len(text)
    recognized_headers = _count_recognized_headers(text)
    emoji_count = _emoji_count(text)
    emoji_ratio = emoji_count / max(char_count, 1)
    non_english_ratio = _non_english_word_ratio(text)

    reasons = []
    if char_count > max_legacy_chars:
        reasons.append(f"length {char_count} > {max_legacy_chars}")
    if recognized_headers < min_recognized_headers:
        reasons.append(f"only {recognized_headers} recognized header(s)")
    if non_english_ratio > non_english_ratio_threshold:
        reasons.append(f"non-English word ratio {non_english_ratio:.2f}")
    if emoji_ratio > emoji_ratio_threshold:
        reasons.append(f"emoji-dense ({emoji_count} emoji)")

    if reasons:
        return RoutingDecision(
            path="llm",
            reason="; ".join(reasons),
            recognized_headers=recognized_headers,
            char_count=char_count,
            emoji_count=emoji_count,
            non_english_ratio=non_english_ratio,
        )

    return RoutingDecision(
        path="legacy",
        reason="short, well-headered, English, emoji-free",
        recognized_headers=recognized_headers,
        char_count=char_count,
        emoji_count=emoji_count,
        non_english_ratio=non_english_ratio,
    )
