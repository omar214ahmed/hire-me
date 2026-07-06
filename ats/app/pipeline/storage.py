"""
Persistence layer: stores JD extractions and CV extractions in Postgres
with pgvector for embedding columns. Same function signatures as the old
JSON-file version — routers are untouched.
"""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from helpers.database import close_pool as _close_pool, get_pool


async def close_pool():
    """Compatibility wrapper used by the Celery task runner."""
    await _close_pool()


def _candidate_text_key(parsed: Optional[Dict]) -> str:
    """Normalize CV text into a stable key for duplicate detection."""
    if not isinstance(parsed, dict):
        return ""

    full_text = parsed.get("full_text")
    if isinstance(full_text, str) and full_text.strip():
        text = full_text
    else:
        sections = parsed.get("sections") or {}
        if isinstance(sections, dict):
            text = " ".join(
                str(sections.get(section, ""))
                for section in ("skills", "experience", "education", "summary")
                if sections.get(section)
            )
        else:
            text = ""

    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _parse_embedding(value) -> list:
    """Convert stored vector string back to a list of floats."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    # asyncpg returns it as a string like '[-0.03, 0.12, ...]'
    import json
    return json.loads(str(value).replace("(", "[").replace(")", "]"))

# -----------------------------
# Jobs (JD)
# -----------------------------

async def save_job(
    jd_text: str,
    jd_extracted: Dict,
    jd_query: str,
    jd_embedding: Optional[list] = None,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO jobs (job_id, jd_text, extracted, query, jd_embedding, created_at)
            VALUES ($1, $2, $3::jsonb, $4, $5::vector, $6)
            """,
            job_id,
            jd_text,
            json.dumps(jd_extracted),
            jd_query,
            str(jd_embedding) if jd_embedding is not None else None,
            datetime.now(timezone.utc),
        )
    return job_id


async def get_job(job_id: str) -> Optional[Dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM jobs WHERE job_id = $1", job_id
        )
    if row is None:
        return None
    return _job_row_to_dict(row)


async def list_jobs() -> List[Dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM jobs ORDER BY created_at")
    return [_job_row_to_dict(r) for r in rows]


def _job_row_to_dict(row) -> Dict:
    d = dict(row)
    if isinstance(d.get("extracted"), str):
        d["extracted"] = json.loads(d["extracted"])
    if d.get("created_at"):
        d["created_at"] = d["created_at"].isoformat()
    d["jd_embedding"] = _parse_embedding(d.get("jd_embedding"))
    return d


def job_to_hr_view(job: Dict) -> Dict:
    """
    Reshape a full job record (as returned by get_job/list_jobs) into the
    minimal {id, job_title, hard_skills} contract consumed by the HR
    service. Pure/stateless so it can be unit-tested without a database.

    `extracted` is keyed by the GLiNER label-map's canonical keys (see
    pipeline/jd_processor.py: LABEL_KEY_MAP) — "required job title" is a
    list (almost always 0 or 1 items in practice), "required hard skill"
    a list of technical/programming skills.
    """
    extracted = job.get("extracted") or {}
    titles = extracted.get("required job title") or []
    hard_skills = extracted.get("required hard skill") or []

    job_title = titles[0] if titles else None

    return {
        "id": job.get("job_id"),
        "job_title": job_title,
        "hard_skills": list(hard_skills),
    }


# -----------------------------
# Candidates (CV)
# -----------------------------

async def save_candidate(
    file_id: str,
    filename: str,
    cv_parsed: Dict,
    cv_embedding: Optional[list] = None,
) -> str:
    pool = await get_pool()
    normalized_text = _candidate_text_key(cv_parsed)
    async with pool.acquire() as conn:
        if normalized_text:
            existing = await conn.fetchrow(
                """
                SELECT candidate_id
                FROM candidates
                WHERE parsed->>'full_text' IS NOT NULL
                  AND lower(regexp_replace(parsed->>'full_text', '\\s+', ' ', 'g')) = $1
                LIMIT 1
                """,
                normalized_text,
            )
            if existing is not None:
                return existing["candidate_id"]

        await conn.execute(
            """
            INSERT INTO candidates
                (candidate_id, file_id, filename, parsed, cv_embedding, created_at)
            VALUES ($1, $2, $3, $4::jsonb, $5::vector, $6)
            ON CONFLICT (candidate_id) DO NOTHING
            """,
            cv_parsed["id"],
            file_id,
            filename,
            json.dumps(cv_parsed),
            str(cv_embedding) if cv_embedding is not None else None,
            datetime.now(timezone.utc),
        )
    return cv_parsed["id"]


async def get_candidate(candidate_id: str) -> Optional[Dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM candidates WHERE candidate_id = $1", candidate_id
        )
    if row is None:
        return None
    return _candidate_row_to_dict(row)


async def list_candidates() -> List[Dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM candidates ORDER BY created_at")
    return [_candidate_row_to_dict(r) for r in rows]


async def get_candidates_with_embeddings() -> List[Dict]:
    """
    Returns only candidates that already have a stored embedding.
    Used by the ranker's fallback path (explicit candidate_ids) instead of
    re-embedding on every match request.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM candidates WHERE cv_embedding IS NOT NULL ORDER BY created_at"
        )
    return [_candidate_row_to_dict(r) for r in rows]


async def get_top_candidates_by_similarity(jd_embedding: list, limit: int) -> List[Dict]:
    """
    The "reduce the number of CVs" step: ask Postgres/pgvector directly
    for the `limit` candidates with the highest cosine similarity to the
    JD embedding, using the HNSW index (idx_candidates_cv_embedding) —
    this scales to a large candidate pool without pulling every row (and
    every embedding) into Python first.

    `cv_embedding <=> $1` is pgvector's cosine *distance* (0 = identical,
    2 = opposite), so `1 - distance` gives the cosine *similarity* score.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *, 1 - (cv_embedding <=> $1::vector) AS semantic_score
            FROM candidates
            WHERE cv_embedding IS NOT NULL
            ORDER BY cv_embedding <=> $1::vector
            LIMIT $2
            """,
            str(jd_embedding),
            limit,
        )
    results = []
    seen_keys = set()
    for row in rows:
        d = _candidate_row_to_dict(row)
        key = _candidate_text_key(d.get("parsed"))
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        d["semantic_score"] = float(row["semantic_score"])
        results.append(d)
    return results


def _candidate_row_to_dict(row) -> Dict:
    d = dict(row)
    if isinstance(d.get("parsed"), str):
        d["parsed"] = json.loads(d["parsed"])
    if d.get("created_at"):
        d["created_at"] = d["created_at"].isoformat()
    d["cv_embedding"] = _parse_embedding(d.get("cv_embedding"))
    return d