from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from pipeline import storage
from pipeline.celery_app import celery
from pipeline.tasks import run_match
from pipeline.schemas import MatchRequest

matching_router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["api_v1", "matching"],
)


@matching_router.get("/{job_id}/shortlist")
async def get_shortlist(
    job_id: str,
    limit: int = Query(default=150, ge=1, le=10000,
                        description="How many top candidates to pull from the DB, by cosine similarity."),
):
    """
    The "COS similarity to reduce the number of CVs" step, exposed as its
    own endpoint: pulls the `limit` candidates with the highest cosine
    similarity to the JD straight from Postgres/pgvector (no reranking,
    no model inference — just an indexed nearest-neighbor query), so you
    can pick how many candidates you want back before paying for the
    (more expensive) reranker step.
    """
    job = await storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")

    if not job.get("jd_embedding"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JD_EMBEDDING_MISSING — re-submit this job description",
        )

    candidates = await storage.get_top_candidates_by_similarity(job["jd_embedding"], limit)

    return {
        "job_id": job_id,
        "count": len(candidates),
        "shortlist": [
            {
                "candidate_id": c["parsed"]["id"],
                "filename": c["filename"],
                "semantic_score": round(c["semantic_score"], 4),
            }
            for c in candidates
        ],
    }


@matching_router.post("/{job_id}/match")
async def match_candidates(job_id: str, payload: MatchRequest = MatchRequest()):
    """
    Dispatches matching to a Celery worker and returns a task_id immediately.
    Client polls /api/v1/jobs/tasks/{task_id} to get results.

    - `shortlist_limit` candidates are pulled from the DB by cosine
      similarity (skipped if `candidate_ids` is given explicitly).
    - The shortlist is then reranked by the cross-encoder; `top_n` of
      those reranked results are returned as the final score.
    """
    job = await storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")

    if not job.get("jd_embedding"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JD_EMBEDDING_MISSING — re-submit this job description",
        )

    task = run_match.delay(
        job_id=job_id,
        candidate_ids=payload.candidate_ids or [],
        shortlist_limit=payload.shortlist_limit,
        top_n=payload.top_n,
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "signal": "MATCH_QUEUED",
            "task_id": task.id,
        },
    )


@matching_router.get("/tasks/{task_id}")
async def get_task_result(task_id: str):
    """Poll this endpoint to check if matching is done and get results."""
    task = celery.AsyncResult(task_id)

    if task.state == "PENDING":
        return {"status": "pending"}
    elif task.state == "STARTED":
        return {"status": "running"}
    elif task.state == "SUCCESS":
        return {"status": "done", "result": task.result}
    elif task.state == "FAILURE":
        return {"status": "failed", "error": str(task.info)}
    else:
        return {"status": task.state.lower()}