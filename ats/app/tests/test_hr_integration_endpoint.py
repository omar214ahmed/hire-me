"""
Tests for the HR-facing job endpoint added for the ATS<->HR microservice
integration: GET /api/v1/jobs/{job_id}/summary, and the pure
job_to_hr_view() transform it's built on.

conftest.py already sets the dummy env vars needed to import
helpers.config without a real .env/database.
"""
import httpx
import pytest
from fastapi import FastAPI
from unittest.mock import AsyncMock, patch

from pipeline.storage import job_to_hr_view
from routers.jobs import jobs_router

FAKE_JOB = {
    "job_id": "job-15",
    "jd_text": "some jd",
    "extracted": {
        "required job title": ["Machine Learning Engineer"],
        "required hard skill": ["Python", "TensorFlow", "PyTorch", "Docker"],
        "required soft skill": ["Communication"],
    },
    "query": "q",
    "jd_embedding": None,
    "created_at": "2026-01-01T00:00:00+00:00",
}


def test_job_to_hr_view_normal_job():
    assert job_to_hr_view(FAKE_JOB) == {
        "id": "job-15",
        "job_title": "Machine Learning Engineer",
        "hard_skills": ["Python", "TensorFlow", "PyTorch", "Docker"],
    }


def test_job_to_hr_view_missing_fields_defaults_gracefully():
    assert job_to_hr_view({"job_id": "xyz", "extracted": {}}) == {
        "id": "xyz",
        "job_title": None,
        "hard_skills": [],
    }
    assert job_to_hr_view({"job_id": "xyz2", "extracted": None}) == {
        "id": "xyz2",
        "job_title": None,
        "hard_skills": [],
    }


@pytest.mark.asyncio
async def test_get_job_summary_endpoint_returns_minimal_view():
    app = FastAPI()
    app.include_router(jobs_router)

    async def fake_get_job(job_id):
        return FAKE_JOB if job_id == "job-15" else None

    with patch("routers.jobs.storage.get_job", new=AsyncMock(side_effect=fake_get_job)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/jobs/job-15/summary")
            assert resp.status_code == 200
            assert resp.json() == {
                "id": "job-15",
                "job_title": "Machine Learning Engineer",
                "hard_skills": ["Python", "TensorFlow", "PyTorch", "Docker"],
            }


@pytest.mark.asyncio
async def test_get_job_summary_endpoint_404_for_unknown_job():
    app = FastAPI()
    app.include_router(jobs_router)

    with patch("routers.jobs.storage.get_job", new=AsyncMock(return_value=None)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/jobs/nope/summary")
            assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_job_full_endpoint_unchanged():
    """Backward-compatibility guard: the pre-existing full job endpoint
    must keep returning the full internal record, unaffected by the new
    /summary endpoint."""
    app = FastAPI()
    app.include_router(jobs_router)

    with patch("routers.jobs.storage.get_job", new=AsyncMock(return_value=FAKE_JOB)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/jobs/job-15")
            assert resp.status_code == 200
            body = resp.json()
            assert body["extracted"]["required job title"] == ["Machine Learning Engineer"]
            assert "jd_text" in body
