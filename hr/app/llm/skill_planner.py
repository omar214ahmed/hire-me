import re


class SkillCoveragePlanner:
    """
    Turns a job description's skills blob into a deterministic queue of
    (skill, angle) pairs, so every generated question is deliberately
    pointed at a different skill/angle combination instead of hoping the
    LLM organically avoids repeating itself.

    This is the structural half of duplicate prevention: it guarantees
    topical diversity by construction. The semantic half — catching two
    questions about the *same* skill/angle slot that still came out
    near-identical — is handled separately by QuestionUniquenessGuard.
    """

    # Rotating "lenses" applied to each skill. Cycling through these before
    # ever reusing a skill means a short interview touches breadth (many
    # skills) before depth (many angles on one skill).
    ANGLES = [
        "core concepts and definitions",
        "trade-offs and design decisions",
        "debugging or troubleshooting a real failure",
        "performance and scalability considerations",
        "best practices and common pitfalls",
        "comparing it with an alternative tool or approach",
        "applying it in a real-world past experience",
        "how it fits into a larger system architecture",
    ]

    _SPLIT_PATTERN = re.compile(r"[,;/\n]+|\band\b", flags=re.IGNORECASE)

    def __init__(self, skills: str):
        self._skills = self._parse_skills(skills)
        if not self._skills:
            self._skills = ["general software engineering"]
        self._queue = self._build_queue()
        self._cursor = 0

    @classmethod
    def _parse_skills(cls, skills: str) -> list[str]:
        raw_parts = cls._SPLIT_PATTERN.split(skills or "")
        seen: set[str] = set()
        parsed: list[str] = []
        for part in raw_parts:
            cleaned = part.strip(" .\t")
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            parsed.append(cleaned)
        return parsed

    def _build_queue(self) -> list[tuple[str, str]]:
        queue: list[tuple[str, str]] = []
        for angle in self.ANGLES:
            for skill in self._skills:
                queue.append((skill, angle))
        return queue

    def next(self) -> tuple[str, str]:
        """Returns the next (skill, angle) pair, cycling forever once the
        full skill x angle grid has been exhausted (for interviews longer
        than skills x len(ANGLES) questions)."""
        pair = self._queue[self._cursor % len(self._queue)]
        self._cursor += 1
        return pair
