"""
Tests for integrations/ats_client.py — the HR service's only way of
learning about a job (it never touches the ATS's database directly).

The "ATS" here is a tiny in-process FastAPI app hit via httpx's
ASGITransport, so these tests run with no real network and no Docker.
"""
import httpx
import pytest
from fastapi import FastAPI, HTTPException

from integrations.ats_client import ATSClient, ATSNotFoundError, ATSUnavailableError


def make_fake_ats(fail_times: int = 0):
    """A minimal fake ATS exposing GET /api/v1/jobs/{job_id}/summary.

    `fail_times` lets a test simulate a service that returns 503 for the
    first N requests to a given job_id before recovering (verifies retry
    behavior), keyed per job_id so tests don't interfere with each other.
    """
    app = FastAPI()
    call_counts: dict[str, int] = {}

    @app.get("/api/v1/jobs/{job_id}/summary")
    async def summary(job_id: str):
        if job_id == "missing":
            raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")
        if job_id == "always-down":
            raise HTTPException(status_code=503, detail="down")
        if job_id == "flaky":
            call_counts[job_id] = call_counts.get(job_id, 0) + 1
            if call_counts[job_id] <= fail_times:
                raise HTTPException(status_code=503, detail="temporarily down")
        return {
            "id": job_id,
            "job_title": "Machine Learning Engineer",
            "hard_skills": ["Python", "PyTorch"],
        }

    return app, call_counts


def make_client(app, max_retries=3) -> ATSClient:
    client = ATSClient(
        base_url="http://ats-test", timeout=2.0, max_retries=max_retries, backoff_base=0.01
    )
    # Point the internal httpx client at the in-process fake app instead of
    # the network.
    client._client = httpx.AsyncClient(
        base_url="http://ats-test",
        transport=httpx.ASGITransport(app=app),
        timeout=2.0,
    )
    return client


@pytest.mark.asyncio
async def test_get_job_summary_happy_path():
    app, _ = make_fake_ats()
    client = make_client(app)
    try:
        result = await client.get_job_summary("job-15")
        assert result == {
            "id": "job-15",
            "job_title": "Machine Learning Engineer",
            "hard_skills": ["Python", "PyTorch"],
        }
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_job_summary_404_raises_not_found_without_retrying():
    app, calls = make_fake_ats()
    client = make_client(app, max_retries=5)
    try:
        with pytest.raises(ATSNotFoundError):
            await client.get_job_summary("missing")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_job_summary_recovers_from_transient_failures():
    app, calls = make_fake_ats(fail_times=2)
    client = make_client(app, max_retries=5)
    try:
        result = await client.get_job_summary("flaky")
        assert result["job_title"] == "Machine Learning Engineer"
        assert calls["flaky"] == 3  # failed twice, succeeded on the 3rd attempt
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_job_summary_raises_unavailable_after_exhausting_retries():
    app, _ = make_fake_ats()
    client = make_client(app, max_retries=3)
    try:
        with pytest.raises(ATSUnavailableError):
            await client.get_job_summary("always-down")
    finally:
        await client.aclose()
