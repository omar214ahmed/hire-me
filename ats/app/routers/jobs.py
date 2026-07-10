from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from pipeline import jd_processor, storage
from pipeline.models import ModelRegistry
from pipeline.schemas import JobHRView

jobs_router = APIRouter(
    prefix="/api/v1/jobs",
    tags=["api_v1", "jobs"],
)


@jobs_router.post("/")
async def create_job(payload: dict):
    """
    Accepts a raw job description string, runs GLiNER extraction,
    stores the structured labels + embedding query, returns the job_id.

    Body: { "description": "<job description text>" }
    """
    description = (payload or {}).get("description", "").strip()

    if not description or len(description) < 20:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "JD_TEXT_TOO_SHORT_OR_MISSING"},
        )

    try:
        jd_result = jd_processor.extract_from_jd_with_sections_v2(description)
        jd_extracted = jd_result["extracted"]
        jd_extraction = jd_result["jd_extraction"]  # confidence-scored JDExtraction
        jd_query = jd_processor.build_jd_query(jd_extracted, sections=jd_result["sections"])
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": "JD_EXTRACTION_FAILED", "error": str(e)},
        )

    # -----------------------------
    # Embed JD query once and persist
    # -----------------------------
    try:
        embedder = ModelRegistry.get_embedder()
        jd_embedding = embedder.encode(
            [jd_query],
            batch_size=1,
            max_length=512,
        )["dense_vecs"][0].tolist()
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": "JD_EMBEDDING_FAILED", "error": str(e)},
        )

    # Full structured dump (incl. per-field confidence) - yes, this
    # duplicates the flat values that are also in `extracted`, but a
    # review dashboard needs the confidence sitting right next to each
    # value, and jsonb storage of one extra small object per job is a
    # trivial cost next to the value of not having to re-derive it.
    extraction_meta = jd_extraction.model_dump()

    job_id = await storage.save_job(
        description, jd_extracted, jd_query, jd_embedding,
        extraction_meta=extraction_meta,
    )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "signal": "JD_PROCESSED_SUCCESS",
            "job_id": job_id,
            "extracted": jd_extracted,
            "query": jd_query,
            "extraction_method": jd_extraction.extraction_method,
            "needs_review": jd_extraction.needs_review,
        },
    )


@jobs_router.get("/{job_id}", response_model=None)
async def get_job(job_id: str):
    job = await storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")
    return job


@jobs_router.get(
    "/{job_id}/summary",
    response_model=JobHRView,
    summary="Minimal job view for other services (e.g. the HR system)",
    description=(
        "Returns only the fields another service needs to build an "
        "interview prompt: job id, job title, and required hard skills. "
        "This is the endpoint the HR service's ATS client calls — it is "
        "intentionally separate from GET /{job_id} (which returns the "
        "full internal job record used by the ATS console/matcher) so "
        "the two can evolve independently."
    ),
)
async def get_job_summary(job_id: str):
    job = await storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")
    return storage.job_to_hr_view(job)


@jobs_router.get("/")
async def list_jobs():
    return {"jobs": await storage.list_jobs()}