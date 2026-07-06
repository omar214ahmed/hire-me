import asyncio
import threading

from pipeline import ranker, storage
from pipeline.celery_app import celery


def _run_coroutine(coro):
    """Run an async coroutine from sync code, even if another loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = {}
    error = {}

    def runner():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive
            error["exc"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "exc" in error:
        raise error["exc"]
    return result["value"]


@celery.task(bind=True)
def run_match(
    self,
    job_id: str,
    candidate_ids: list = None,
    shortlist_limit: int = 150,
    top_n: int = 20,
):
    """
    Celery task that runs the full matching pipeline:
      1. shortlist: pgvector cosine similarity in Postgres narrows the
         candidate pool down to `shortlist_limit` (default 150) — unless
         `candidate_ids` was explicitly passed, in which case that fixed
         set is used instead (semantic score computed in Python for that
         small set, since it's not a DB-wide query).
      2. rerank: BGE-reranker-v2-m3 cross-encoder scores the shortlist,
         final order/score comes from this step, truncated to `top_n`.

    Runs in a worker process, not in the FastAPI event loop.
    """

    async def _run_match_async():
        try:
            job = await storage.get_job(job_id)
            if not job:
                return {"error": "JOB_NOT_FOUND"}

            jd_embedding = job.get("jd_embedding")
            if not jd_embedding:
                return {"error": "JD_EMBEDDING_MISSING"}

            if candidate_ids:
                records = await asyncio.gather(
                    *[storage.get_candidate(cid) for cid in candidate_ids]
                )
                records = [c for c in records if c is not None and c.get("cv_embedding")]
                seen_keys = set()
                deduped_records = []
                for record in records:
                    key = storage._candidate_text_key(record.get("parsed"))
                    if key and key in seen_keys:
                        continue
                    if key:
                        seen_keys.add(key)
                    deduped_records.append(record)
                records = deduped_records
                sem_scores = ranker.semantic_scores(
                    jd_embedding, [c["cv_embedding"] for c in records]
                )
                for record, score in zip(records, sem_scores):
                    record["semantic_score"] = score
                records.sort(key=lambda r: r["semantic_score"], reverse=True)
                candidate_records = records
            else:
                candidate_records = await storage.get_top_candidates_by_similarity(
                    jd_embedding, shortlist_limit
                )

            if not candidate_records:
                return {"error": "NO_CANDIDATES_FOUND"}

            ranked = ranker.rank_candidates(
                jd_text=job["jd_text"],
                jd_extracted=job["extracted"],
                candidate_records=candidate_records,
                top_n=top_n,
            )

            for r in ranked:
                r.pop("cv_text", None)

            return {
                "signal": "MATCH_SUCCESS",
                "job_id": job_id,
                "shortlisted": len(candidate_records),
                "results": ranked,
            }
        finally:
            await storage.close_pool()

    return _run_coroutine(_run_match_async())