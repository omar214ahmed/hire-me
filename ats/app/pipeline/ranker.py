"""
Steps 4-6 of the pipeline (matches the whiteboard flow):
  - Embed both sides with BGE-M3 (done once, at ingestion time — see
    routers/candidates.py and routers/jobs.py)
  - COS similarity reduces the candidate pool (e.g. 10,000 -> 150-100) —
    this happens in Postgres/pgvector, see storage.get_top_candidates_by_similarity
  - BGE-reranker-v2-m3 cross-encoder reranks that shortlist -> final score

`matcher.hard_match` (skills/experience/education/language rule-based
score) is no longer part of the filtering/ranking decision — it's
attached to each result purely as explainability metadata for the UI, so
a recruiter can see *why* a candidate ranked where it did.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.models import ModelRegistry
from pipeline.matcher import hard_match


def combine_scores(rerank_score: float, hard_match_score: float, has_hard_requirements: bool = False) -> float:
    """Blend the reranker score with hard-match evidence.

    The reranker is still the main signal, but a weak hard-match should not
    be allowed to look almost perfect when the JD/CV requirement overlap is
    poor. If hard requirements are present and the hard-match score is low,
    pull the final score down noticeably; if the hard-match is strong, keep
    the reranker score almost intact.
    """
    if not has_hard_requirements:
        return float(rerank_score)

    hard_score = max(0.0, min(1.0, float(hard_match_score)))
    rerank = max(0.0, min(1.0, float(rerank_score)))

    if hard_score < 0.5:
        return round(0.6 * rerank + 0.4 * hard_score, 4)
    return round(0.85 * rerank + 0.15 * hard_score, 4)


def semantic_scores(jd_embedding: List[float], cv_embeddings: List[List[float]]) -> List[float]:
    """
    Cosine similarity between the JD embedding and a list of CV embeddings,
    computed in Python. Only used for the explicit-candidate_ids path
    (routers/matching.py), where the DB-side pgvector query
    (storage.get_top_candidates_by_similarity) isn't applicable since the
    candidate set is already fixed by the caller.
    """
    if not cv_embeddings:
        return []
    jd_vec = np.array(jd_embedding).reshape(1, -1)
    cv_vecs = np.array(cv_embeddings)
    scores = cosine_similarity(jd_vec, cv_vecs)[0]
    return [float(s) for s in scores]


def _cv_text(cv_parsed: Dict) -> str:
    sections = cv_parsed.get("sections", {})
    parts = [
        sections.get("skills", ""),
        sections.get("experience", ""),
        sections.get("education", ""),
    ]
    return " ".join(p for p in parts if p)


def attach_explainability(jd_extracted: Dict, candidate_records: List[Dict]) -> List[Dict]:
    """
    Turns raw candidate DB rows (each already carrying a `semantic_score`
    from the cosine-similarity shortlist step) into the candidate dicts
    the reranker expects, with the rule-based hard-match breakdown
    attached for explainability only — it does not affect ordering.
    """
    candidates = []
    for record in candidate_records:
        cv = record["parsed"]
        breakdown = hard_match(jd_extracted, cv)
        candidates.append({
            "cv_id": cv["id"],
            "semantic_score": round(record.get("semantic_score", 0.0), 4),
            "hard_match": breakdown,
            "cv_text": _cv_text(cv),
        })
    return candidates


def rerank_candidates(jd_text: str, candidates: List[Dict], top_n: int) -> List[Dict]:
    """
    Cross-encoder reranks the (already cosine-shortlisted) candidates.
    This produces the *final* score/order — see whiteboard: "apply bge
    m3 reranker -> Final score".
    """
    if not candidates:
        return candidates

    reranker = ModelRegistry.get_reranker()
    pairs = [[jd_text, c["cv_text"]] for c in candidates]
    rerank_scores = reranker.compute_score(pairs, normalize=True, max_length=1024)

    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]

    for c, rs in zip(candidates, rerank_scores):
        c["rerank_score"] = round(float(rs), 4)

    for candidate in candidates:
        hard = candidate.get("hard_match", {})
        hard_total = hard.get("total", 1.0) if isinstance(hard, dict) else 1.0
        has_hard_requirements = bool(
            hard.get("breakdown") if isinstance(hard, dict) else None
        )
        candidate["final_score"] = combine_scores(
            candidate["rerank_score"],
            hard_total,
            has_hard_requirements=has_hard_requirements,
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:top_n]


def rank_candidates(
    jd_text: str,
    jd_extracted: Dict,
    candidate_records: List[Dict],
    top_n: int = 20,
) -> List[Dict]:
    """
    Full ranking pipeline over an already cosine-shortlisted set of
    candidates: attach hard-match breakdown (explainability only) ->
    cross-encoder rerank -> top_n final results.
    """
    candidates = attach_explainability(jd_extracted, candidate_records)
    return rerank_candidates(jd_text, candidates, top_n=top_n)