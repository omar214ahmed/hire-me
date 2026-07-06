"""
HTTP client the HR service uses to fetch job data from the ATS service.

This is the *only* way the HR service learns about jobs (title, required
hard skills, etc.) — per the platform's architecture, HR never connects to
the ATS's Postgres database directly:

    HR service --HTTP--> ATS service --SQL--> PostgreSQL

Inside Docker Compose, base_url must be the ATS's *service name*
(e.g. "http://ats:8000"), never "localhost" — see helpers/config.py.
"""

import logging

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("hr_interview.ats_client")


class ATSNotFoundError(Exception):
    """The ATS service is reachable but has no job with that id."""


class ATSUnavailableError(Exception):
    """The ATS service could not be reached, or kept failing, after retries."""


class _RetryableATSError(Exception):
    """Internal signal: a transient error (network issue or 5xx) worth retrying."""


class ATSClient:
    """Thin async wrapper around the ATS's HR-facing job endpoint."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ):
        self._base_url = base_url.rstrip("/")
        self._max_retries = max(1, max_retries)
        self._backoff_base = backoff_base
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_job_summary(self, job_id: str) -> dict:
        """
        GET {ATS_API_URL}/api/v1/jobs/{job_id}/summary

        Returns:
            {"id": "...", "job_title": "...", "hard_skills": ["...", ...]}

        Raises:
            ATSNotFoundError: the ATS responded 404 — no such job. Not
                retried; retrying a 404 wastes time and won't change
                the answer.
            ATSUnavailableError: a network error, timeout, or 5xx that
                persisted across every retry attempt, or an unexpected
                4xx that a retry can't fix.
        """
        path = f"/api/v1/jobs/{job_id}/summary"

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(
                    multiplier=self._backoff_base, min=self._backoff_base, max=10
                ),
                retry=retry_if_exception_type(_RetryableATSError),
                reraise=True,
            ):
                with attempt:
                    return await self._do_request(path, job_id)
        except _RetryableATSError as exc:
            raise ATSUnavailableError(
                f"ATS service unavailable after {self._max_retries} attempt(s): {exc}"
            ) from exc

        # Unreachable in practice (AsyncRetrying always either returns or
        # raises), but keeps type-checkers/linters happy.
        raise ATSUnavailableError("ATS service unavailable")

    async def _do_request(self, path: str, job_id: str) -> dict:
        try:
            response = await self._client.get(path)
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.PoolTimeout,
        ) as exc:
            logger.warning("ATS request to %s failed (network error): %s", path, exc)
            raise _RetryableATSError(str(exc)) from exc

        if response.status_code == 404:
            raise ATSNotFoundError(f"Job {job_id!r} not found in ATS")

        if response.status_code >= 500:
            logger.warning("ATS returned %s for %s", response.status_code, path)
            raise _RetryableATSError(f"ATS returned {response.status_code}")

        if response.status_code >= 400:
            # Any other 4xx (bad request, etc.) — a retry won't help.
            raise ATSUnavailableError(
                f"ATS returned unexpected status {response.status_code}: {response.text}"
            )

        return response.json()
