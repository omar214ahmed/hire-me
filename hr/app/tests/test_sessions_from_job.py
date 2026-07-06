"""
Tests for POST /sessions/from-job/{job_id} — the HR endpoint that fetches
a job from the ATS service and uses it to seed an interview session,
instead of the caller typing role/skills by hand.

The LLM/transcription chains aren't exercised here (no Ollama/Whisper
needed) — only the ATS-integration wiring: does the router call the ATS
client correctly, and does it translate ATS errors into the right HTTP
status codes.
"""
import httpx
import pytest
from fastapi import FastAPI, HTTPException

from integrations.ats_client import ATSClient
from routers.sessions import router as sessions_router


class _FakeChains:
    question_chain = None
    classification_chain = None
    evaluation_chain = None


def make_app(ats_app):
    """Build a minimal HR app: the real sessions router, wired to fake
    chains/transcript (never invoked in these tests) and an ATSClient
    pointed at an in-process fake ATS app."""
    app = FastAPI()
    app.state.sessions = {}
    app.state.chains = _FakeChains()
    app.state.transcript = object()

    ats_client = ATSClient(base_url="http://ats-test", timeout=2.0, max_retries=2, backoff_base=0.01)
    ats_client._client = httpx.AsyncClient(
        base_url="http://ats-test",
        transport=httpx.ASGITransport(app=ats_app),
        timeout=2.0,
    )
    app.state.ats_client = ats_client

    app.include_router(sessions_router)
    return app, ats_client


def make_fake_ats():
    ats_app = FastAPI()

    @ats_app.get("/api/v1/jobs/{job_id}/summary")
    async def summary(job_id: str):
        if job_id == "missing":
            raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")
        return {
            "id": job_id,
            "job_title": "Machine Learning Engineer",
            "hard_skills": ["Python", "PyTorch", "Docker"],
        }

    return ats_app


@pytest.mark.asyncio
async def test_create_session_from_job_autofills_role_and_skills():
    app, ats_client = make_app(make_fake_ats())
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://hr-test") as client:
            resp = await client.post("/sessions/from-job/job-15")
            assert resp.status_code == 201, resp.text
            data = resp.json()
            assert data["role"] == "Machine Learning Engineer"
            assert data["skills"] == "Python, PyTorch, Docker"
            assert data["job_id"] == "job-15"

            # get_session should reflect the same job_id
            resp2 = await client.get(f"/sessions/{data['id']}")
            assert resp2.status_code == 200
            assert resp2.json()["job_id"] == "job-15"
    finally:
        await ats_client.aclose()


@pytest.mark.asyncio
async def test_create_session_from_unknown_job_returns_404():
    app, ats_client = make_app(make_fake_ats())
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://hr-test") as client:
            resp = await client.post("/sessions/from-job/missing")
            assert resp.status_code == 404
    finally:
        await ats_client.aclose()


@pytest.mark.asyncio
async def test_manual_session_creation_still_works_unaffected():
    app, ats_client = make_app(make_fake_ats())
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://hr-test") as client:
            resp = await client.post(
                "/sessions", json={"role": "Backend Engineer", "skills": "Go, Postgres"}
            )
            assert resp.status_code == 201
            data = resp.json()
            assert data["role"] == "Backend Engineer"
            assert data["job_id"] is None
    finally:
        await ats_client.aclose()
