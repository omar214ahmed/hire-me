"""
JD sectioning, v2: chunk -> classify -> merge.

Why this replaces the old single-pass line scanner
----------------------------------------------------
The previous `split_jd_sections` walked the JD line by line, kept ONE
"current section" pointer, and routed every line to whichever header it
last saw. That has three structural problems no amount of regex tuning
fixes:

  1. A line only ever gets ONE label - the label of whatever header came
     before it. A mixed-content line ("Remote - must relocate for
     on-site sprints") can't be *both* location and requirement.
  2. A JD with no headers at all (or headers the pattern list doesn't
     recognize) dumps everything into a single catch-all bucket, because
     there is nothing but the header regex driving classification.
  3. Content is trusted blindly just because it sits under a header -
     a stray "Location: Remote" line typed under "Requirements" (people
     don't always format JDs cleanly) stays misfiled forever.

This module fixes all three by splitting the pipeline into three
independent, testable stages:

  chunk_jd()      text -> List[Chunk]            (paragraph-sized units,
                                                    context preserved)
  classify_chunk() Chunk -> (primary, secondary)  (content scoring, not
                                                    just "trust the header")
  merge_chunks()  List[Chunk] -> Dict[str, str]   (additive - a label can
                                                    never be silently
                                                    overwritten, only
                                                    appended to)

Each stage has one job, so a future "one more edge case" fix touches one
function instead of a 700-line if/elif chain.
"""

import re
import unicodedata
import html
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Text cleaning (moved here unchanged from the old jd_processor - chunking
# is the first thing that touches raw JD text, so this is where it belongs;
# jd_processor re-exports it so nothing downstream needs to know it moved).
# ---------------------------------------------------------------------------
_LITERAL_ESCAPE_PATTERN = re.compile(r"\\r\\n|\\n|\\r|\\t")
_LITERAL_ESCAPE_MAP = {r"\r\n": "\n", r"\n": "\n", r"\r": "\n", r"\t": " "}

_HTML_BLOCK_BREAK_TAGS = re.compile(
    r"</?\s*(br|p|div|li|ul|ol|tr|table|h[1-6])\s*/?\s*>", re.IGNORECASE
)
_HTML_ANY_TAG = re.compile(r"<[^>]+>")

_BULLET_CHARS = re.compile(r"^[\s]*[•‣▪●◦▶○✓✔·∙–—*-]\s*")
_ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")


def clean_jd_text(text: str) -> str:
    """Normalize a raw JD payload before chunking. Idempotent. See the
    module docstring in the git history / README for the exact list of
    malformed inputs this guards against (literal "\\n", HTML remnants,
    NBSP/zero-width chars, stray bullets, Windows line endings, ...)."""
    if not text:
        return text

    cleaned = html.unescape(text)

    def _replace_escape(match: "re.Match") -> str:
        return _LITERAL_ESCAPE_MAP[match.group(0)]

    cleaned = _LITERAL_ESCAPE_PATTERN.sub(_replace_escape, cleaned)

    if "<" in cleaned and ">" in cleaned:
        cleaned = _HTML_BLOCK_BREAK_TAGS.sub("\n", cleaned)
        cleaned = _HTML_ANY_TAG.sub("", cleaned)

    cleaned = _ZERO_WIDTH_CHARS.sub("", cleaned)
    cleaned = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in cleaned)

    lines = cleaned.split("\n")
    lines = [_BULLET_CHARS.sub("", line) for line in lines]
    cleaned = "\n".join(lines)

    cleaned = _MULTI_SPACES.sub(" ", cleaned)
    cleaned = _MULTI_BLANK_LINES.sub("\n\n", cleaned)
    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines).strip()

    return cleaned


# ---------------------------------------------------------------------------
# Canonical labels + header vocabulary
# ---------------------------------------------------------------------------
# Same canonical keys the rest of the pipeline (SECTION_LABEL_MAP,
# _bucket_entities fallbacks, build_jd_query) already expects. Order here
# only matters for header-pattern matching precedence (first match wins),
# same rule the old implementation used - more specific patterns first so
# they don't get swallowed by a more general one later in the dict.
CANONICAL_LABELS = [
    "benefits",
    "nice_to_have",
    "job_meta",
    "requirements",
    "soft_skills",
    "experience",
    "education",
    "skills",
    "languages",
    "summary",
    "header",
]

JD_SECTION_PATTERNS = {
    # Perks/compensation-package copy ("Benefits", "Perks", "What's in it
    # for you", ...) - kept separate from job_meta (which is the raw
    # employment-type/salary/location metadata) so a JD's benefits list
    # (health plan, PTO, stipends, ...) surfaces as its own bucket instead
    # of being merged into unrelated metadata.
    "benefits": (
        r"benefits?|perks?|"
        r"what('s| is)\s+in\s+it\s+for\s+you|"
        r"why\s+(you('ll)?\s+)?(join|love)\s+(us|working\s+here)"
    ),
    "nice_to_have": (
        r"(?:preferred|desired|good[\s-]+to[\s-]+have|nice[\s-]+to[\s-]+have|"
        r"bonus|pluses?|optional|extra)\s*"
        r"(?:points?)?\s*"
        r"(?:qualifications?|requirements?|skills?)?"
    ),
    "job_meta": (
        r"employment\s+type|experience\s+level|seniority(\s+level)?|"
        r"job\s+type|work\s+arrangement|work\s+mode|"
        r"location|salary|compensation"
    ),
    "requirements": (
        r"(?:(?:required|minimum|must[\s-]have|essential|mandatory|basic|"
        r"key|other|additional)\s+)?(?:requirements?|qualifications?)"
        r"|what\s+you('ll)?\s+(need|bring)"
        r"|must\s+have"
    ),
    "soft_skills": (
        r"soft\s+skills?|interpersonal\s+skills?|personality\s+traits?"
    ),
    "experience": (
        r"(work\s+)?experience|responsibilities|duties|"
        r"(?:job|role|position)?\s*description|"
        r"what\s+you('ll)?\s+do|role\s+overview"
    ),
    "education": r"education(al\s+background)?|academic(\s+background)?|degrees?",
    "skills": (
        r"(?:technical|soft|hard|core|key|general)?\s*skills?"
        r"|tech(nology)?\s+stack|tools?"
        r"|programming(\s+languages?)?|libraries|frameworks?"
        r"|data\s+analysis|machine\s+learning|artificial\s+intelligence"
    ),
    "languages": r"languages?|spoken\s+languages?",
    "summary": (
        r"summary|about\s+(the\s+role|us|the\s+company)|overview"
        r"|what\s+you('ll)?\s+learn"
    ),
}

_HEADER_LINE_MAX_LEN = 60


def _match_header(stripped_line: str) -> Optional[Tuple[str, Optional[str]]]:
    """Return (canonical_key, inline_content_or_None) if stripped_line
    looks like a section header, else None."""
    if not stripped_line or len(stripped_line) >= _HEADER_LINE_MAX_LEN:
        return None
    lowered = stripped_line.lower()
    for section, pattern in JD_SECTION_PATTERNS.items():
        # Named group for the inline-content capture, deliberately - some
        # of the alternations above (e.g. "(work\s+)?experience") contain
        # their own capturing groups, which would silently shift a
        # positional index like group(2)/group(3) depending on which
        # pattern matched. A named group is immune to that.
        m = re.match(r"^(?:" + pattern + r")\s*(?::\s*(?P<inline>.*))?$", lowered)
        if m:
            inline = m.group("inline")
            return section, (inline.strip() if inline and inline.strip() else None)
    return None


# ---------------------------------------------------------------------------
# Stage 1: chunking
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    header_hint: str          # canonical label of the nearest preceding header ("header" if none yet)
    order: int                # document order, for stable, readable merges
    primary_label: str = ""   # filled in by classify_chunk
    secondary_labels: List[str] = field(default_factory=list)


def chunk_jd(text: str) -> List[Chunk]:
    """Split a cleaned JD into paragraph-sized chunks, each tagged with
    the nearest preceding header (its "header_hint").

    A chunk boundary is either:
      - a blank line (paragraph break), or
      - a line that itself looks like a section header (so runs of
        back-to-back sub-headings with no blank line between them, e.g.
        "Technical Skills\\nProgramming\\nPython, R\\nMachine Learning\\n...",
        still produce one chunk per sub-list instead of one giant blob).

    Chunks are built additively (append to a list) - never by resetting
    a shared per-key accumulator - so a header key that recurs later in
    the document can never wipe out content collected under it earlier.
    """
    text = clean_jd_text(text)
    lines = text.split("\n")

    chunks: List[Chunk] = []
    buffer: List[str] = []
    header_hint = "header"
    order = 0

    def flush():
        nonlocal buffer, order
        if not buffer:
            return
        if header_hint == "header" and len(buffer) > 1:
            # No header governs this block, so its lines don't share
            # established context - forcing them into one paragraph-sized
            # chunk would let one line's topic (e.g. a responsibilities
            # sentence) drag unrelated lines (e.g. a degree requirement)
            # along with it. Classify line-by-line instead. Content that
            # *does* sit under a real header is still kept as one chunk
            # (that's the actual context worth preserving).
            for line in buffer:
                stripped_line = line.strip()
                if stripped_line:
                    chunks.append(Chunk(text=stripped_line, header_hint=header_hint, order=order))
                    order += 1
        else:
            content = "\n".join(buffer).strip()
            if content:
                chunks.append(Chunk(text=content, header_hint=header_hint, order=order))
                order += 1
        buffer = []

    for raw_line in lines:
        stripped = raw_line.strip()

        if not stripped:
            flush()
            continue

        header_match = _match_header(stripped)
        if header_match:
            flush()
            header_hint, inline = header_match
            if inline:
                buffer.append(inline)
            continue

        buffer.append(raw_line)

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Stage 2: classification
# ---------------------------------------------------------------------------
# Lightweight, explainable, content-based scoring - deliberately NOT a
# giant ML call per chunk (keeps this stage fast and dependency-free).
# Each cue list is easy to extend without touching the scoring logic.
_CUES: Dict[str, List[re.Pattern]] = {
    "benefits": [
        re.compile(
            r"\b(health\s+(plan|insurance)|dental\s+(plan|insurance)|vision\s+(plan|insurance)|"
            r"life\s+insurance|paid\s+time\s+off|\bpto\b|parental\s+leave|maternity\s+leave|"
            r"paternity\s+leave|401\(?k\)?|retirement\s+plan|stock\s+options?|equity\s+grant|"
            r"gym\s+membership|wellness|wellhub|meal\s+(card|voucher|allowance)|"
            r"snack(?:\s+allowance)?|vacation\s+days?|day\s+off|telemedicine|"
            r"tuition|learning\s+(stipend|allowance|budget)|professional\s+development|"
            r"discount|perks?|benefits?|onboarding\s+kit|home\s+office\s+(allowance|stipend)|"
            r"flexible\s+spending|sign\s+language|life\s+balance)\b",
            re.I,
        ),
    ],
    "job_meta": [
        re.compile(r"\b(full[\s-]?time|part[\s-]?time|contract(?:or)?|freelance|"
                    r"intern(?:ship)?|temporary|permanent|remote|hybrid|on[\s-]?site)\b", re.I),
        re.compile(r"\b(salary|compensation|per\s+(annum|year|hour)|\$\s?\d|bonus\s+plan)\b", re.I),
        re.compile(r"\b(location|based\s+in|relocate|relocation|time\s?zone)\b", re.I),
    ],
    "education": [
        re.compile(r"\b(bachelor|master|phd|ph\.d|degree|diploma)\b", re.I),
        re.compile(r"\b(computer science|data science|statistics|mathematics|"
                    r"software engineering|artificial intelligence)\b", re.I),
    ],
    "soft_skills": [
        re.compile(r"\b(communication|collaborat\w*|teamwork|leadership|"
                    r"problem[\s-]?solv\w*|critical\s+thinking|attention\s+to\s+detail|"
                    r"analytical|adaptab\w*|interpersonal|time\s+management|"
                    r"ownership|accountability|curiosity)\b", re.I),
    ],
    "nice_to_have": [
        re.compile(r"\b(preferred|nice[\s-]to[\s-]have|bonus|a\s+plus|optional|desired|good[\s-]to[\s-]have)\b", re.I),
    ],
    "languages": [
        re.compile(r"\b(english|arabic|french|german|spanish|mandarin|fluent|"
                    r"native\s+speaker|bilingual|spoken\s+language)\b", re.I),
    ],
    "skills": [
        re.compile(r"\b(python|sql|java|c\+\+|javascript|typescript|react|node|"
                    r"docker|kubernetes|aws|azure|gcp|git|linux|spark|airflow|"
                    r"tensorflow|pytorch|pandas|numpy|scikit-learn|nlp|api|"
                    r"tech(?:nology)?\s+stack|tools?|frameworks?|libraries)\b", re.I),
        # structural cue: a short, comma-separated list of capitalized-ish
        # tokens (typical of a tools/skills enumeration).
        re.compile(r"^(?:[A-Za-z0-9+.#/ ]{2,25},\s*){1,}[A-Za-z0-9+.#/ ]{2,25}$"),
    ],
    "experience": [
        re.compile(r"\b(develop|design|build|manage|lead|own|implement|"
                    r"collaborate|maintain|analyze|deploy|responsible\s+for|"
                    r"you\s+will|you'll\s+be)\b", re.I),
    ],
    "requirements": [
        re.compile(r"\b(must|should|required|requires?|minimum|essential|mandatory)\b", re.I),
        re.compile(r"\b\d+\s*[-+]?\s*years?\b", re.I),
    ],
    "summary": [
        re.compile(r"\b(we\s+are|our\s+mission|about\s+us|founded|company\s+overview|"
                    r"join\s+our\s+team|we're\s+looking\s+for|we\s+build)\b", re.I),
    ],
}

_OVERRIDE_MARGIN = 1        # how much a content score must beat the header-hint's own score by
_CONTENT_MIN_SCORE = 2      # minimum score to be trusted as an override at all
_SECONDARY_MIN_SCORE = 2    # minimum score for a label to be added as secondary


def _score_chunk(text: str) -> Dict[str, Tuple[int, int]]:
    """Score = (distinct cue patterns matched, total raw hits).

    The primary score is the count of distinct cue *patterns* that
    matched (not raw hits) - a phrase like "Bachelor's degree" hitting
    both "bachelor" and "degree" inside the SAME education pattern is
    one signal, not two, so it can't out-score an unrelated
    single-pattern match elsewhere. Total raw hits is kept only as a
    tie-breaker between labels with an equal primary score (e.g. a
    sentence dense with soft-skill words beats a label with one
    incidental keyword match)."""
    scores: Dict[str, Tuple[int, int]] = {}
    for label, patterns in _CUES.items():
        group_hits = 0
        total_hits = 0
        for p in patterns:
            hits = len(p.findall(text))
            if hits:
                group_hits += 1
                total_hits += hits
        if group_hits:
            scores[label] = (group_hits, total_hits)
    return scores


def classify_chunk(chunk: Chunk) -> Chunk:
    """Assign a primary label (drives which section bucket the chunk's
    text lands in) and zero or more secondary labels (extra buckets the
    chunk's text is *also* surfaced under, for recall - e.g. a stray tech
    mention inside a "Requirements" paragraph still reaches the "skills"
    NER pass). Mutates and returns the chunk."""
    scores = _score_chunk(chunk.text)
    base_label = chunk.header_hint
    base_score = scores.get(base_label, (0, 0))[0]

    if scores:
        # Deterministic tie-break chain: most distinct cue patterns
        # matched, then most raw hits, then canonical position - so ties
        # never depend on dict/hash ordering.
        best_label = max(
            scores,
            key=lambda label: (scores[label][0], scores[label][1], -CANONICAL_LABELS.index(label)),
        )
        best_score = scores[best_label][0]
    else:
        best_label, best_score = base_label, 0

    # "header" is the generic default a chunk gets when no real section
    # header governs it - it isn't a genuine claim on the content, so any
    # confident content signal is enough to reclassify it. A REAL header
    # (requirements, skills, ...) is a stronger claim and needs the
    # content signal to clearly beat it before we override it.
    no_real_header = base_label == "header"
    min_score = 1 if no_real_header else _CONTENT_MIN_SCORE
    margin = 0 if no_real_header else _OVERRIDE_MARGIN

    if (
        best_label != base_label
        and best_score >= min_score
        and best_score >= base_score + margin
    ):
        primary = best_label
    else:
        primary = base_label

    secondary = sorted(
        label
        for label, (group_hits, _total) in scores.items()
        if label != primary and group_hits >= _SECONDARY_MIN_SCORE
    )

    chunk.primary_label = primary
    chunk.secondary_labels = secondary
    return chunk


# ---------------------------------------------------------------------------
# Stage 3: merge
# ---------------------------------------------------------------------------
def merge_chunks(chunks: List[Chunk]) -> Dict[str, str]:
    """Group classified chunks into the final {canonical_label: text}
    dict. Additive and order-preserving: every chunk's text is appended
    (never overwrites), and identical text is never appended twice under
    the same label, so information is neither lost nor duplicated."""
    buckets: Dict[str, List[str]] = {}
    seen: Dict[str, set] = {}

    def add(label: str, text: str):
        bucket = buckets.setdefault(label, [])
        dedup = seen.setdefault(label, set())
        if text not in dedup:
            bucket.append(text)
            dedup.add(text)

    for chunk in sorted(chunks, key=lambda c: c.order):
        add(chunk.primary_label, chunk.text)
        for label in chunk.secondary_labels:
            add(label, chunk.text)

    return {label: "\n".join(texts).strip() for label, texts in buckets.items()}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def split_jd_sections(text: str) -> Dict[str, str]:
    """chunk -> classify -> merge. Drop-in replacement for the old
    single-pass line scanner with the same return shape
    ({canonical_label: text}), but robust to missing headers, repeated
    headers, and mixed-content paragraphs (see module docstring)."""
    chunks = chunk_jd(text)
    chunks = [classify_chunk(c) for c in chunks]
    return merge_chunks(chunks)
