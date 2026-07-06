import time
import logging
from pathlib import Path
from uuid import uuid4
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from schemas import EvaluationSchema, CreateSessionRequest, SessionResponse, QuestionResponse, AnswerResponse, SummaryResponse
from interview import InterviewSession
from llm import QuestionsGenerator, Evaluator, ClassificationQuestion, Transcript
from integrations import ATSNotFoundError, ATSUnavailableError

logger = logging.getLogger("hr_interview.api")
router = APIRouter(prefix="/sessions", tags=["sessions"])




def _get_store(request: Request) -> dict:
    return request.app.state.sessions


def _new_session_entry(request: Request, role: str, skills: str, job_id: Optional[str] = None) -> tuple[str, dict]:
    """Build and register a new InterviewSession. Shared by the manual
    (role/skills typed in by hand) and job-based (fetched from the ATS)
    creation paths so both stay in sync."""
    store = _get_store(request)
    session_id = str(uuid4())
    session = InterviewSession(
        transcript=request.app.state.transcript,
        generator=QuestionsGenerator(request.app.state.chains),
        classifier=ClassificationQuestion(request.app.state.chains),
        evaluator=Evaluator(request.app.state.chains),
    )
    entry = {
        "session": session,
        "role": role,
        "skills": skills,
        "job_id": job_id,
        "results": [],
    }
    store[session_id] = entry
    return session_id, entry


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(body: CreateSessionRequest, request: Request):
    session_id, entry = _new_session_entry(request, role=body.role, skills=body.skills)
    return SessionResponse(
        id=session_id,
        role=entry["role"],
        skills=entry["skills"],
        job_id=entry["job_id"],
    )


@router.post("/from-job/{job_id}", response_model=SessionResponse, status_code=201)
async def create_session_from_job(job_id: str, request: Request):
    """
    Create an interview session directly from an ATS job id, instead of
    the caller typing role/skills by hand.

    Workflow (HR -> ATS -> PostgreSQL):
        1. Receive job_id from the caller.
        2. GET {ATS_API_URL}/api/v1/jobs/{job_id}/summary via the ATS client.
        3. Use the returned job_title / hard_skills to seed the session,
           exactly like POST /sessions but auto-filled.
        4. Question generation then proceeds as usual via
           POST /sessions/{id}/questions.
    """
    ats_client = request.app.state.ats_client

    try:
        job = await ats_client.get_job_summary(job_id)
    except ATSNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found in ATS: {exc}")
    except ATSUnavailableError as exc:
        logger.error("ATS service unavailable while fetching job %s: %s", job_id, exc)
        raise HTTPException(
            status_code=503,
            detail="ATS service is currently unavailable. Please try again shortly.",
        )

    role = job.get("job_title") or "Unspecified Role"
    hard_skills = job.get("hard_skills") or []
    skills = ", ".join(hard_skills)

    session_id, entry = _new_session_entry(request, role=role, skills=skills, job_id=job.get("id", job_id))
    return SessionResponse(
        id=session_id,
        role=entry["role"],
        skills=entry["skills"],
        job_id=entry["job_id"],
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    return SessionResponse(
        id=session_id,
        role=entry["role"],
        skills=entry["skills"],
        job_id=entry.get("job_id"),
        current_question=sess.current_question,
        current_category=sess.current_category,
        question_history=sess.question_history,
        results_count=len(entry["results"]),
    )


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request):
    store = _get_store(request)
    if session_id not in store:
        raise HTTPException(404, "Session not found")
    del store[session_id]


@router.post("/{session_id}/questions", response_model=QuestionResponse)
async def generate_question(session_id: str, request: Request):
    start = time.perf_counter()
    logger.info("POST /sessions/%s/questions started", session_id)
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    sess.generate_question(entry["role"], entry["skills"])
    sess.classify_current_question()

    elapsed = time.perf_counter() - start
    threshold_warn = 8000
    if elapsed >= threshold_warn:
        logger.warning(
            "SLOW | POST /sessions/%s/questions took %.2f seconds (>= %.0f s) | role='%s' | skills='%s'",
            session_id, elapsed, threshold_warn, entry["role"], entry["skills"],
        )
    else:
        logger.info(
            "POST /sessions/%s/questions completed in %.2f seconds | role='%s'",
            session_id, elapsed, entry["role"],
        )

    return QuestionResponse(
        question=sess.current_question,
        category=sess.current_category,
    )


@router.get("/{session_id}/current-question", response_model=QuestionResponse)
async def get_current_question(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    if not sess.current_question:
        raise HTTPException(404, "No question generated yet")
    return QuestionResponse(
        question=sess.current_question,
        category=sess.current_category,
    )


@router.post("/{session_id}/answers", response_model=AnswerResponse)
async def submit_answer(session_id: str, request: Request, file: UploadFile = File(...)):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]

    filename = file.filename or "audio.ogg"
    suffix = Path(filename).suffix.lower() or ".ogg"
    audio_bytes = await file.read()

    result = sess.evaluate_answer(audio_bytes, suffix)
    ev_score = result.get("evaluation", {}).get("score")
    print(ev_score)
    if ev_score is None:
        logger.warning("Storing result with score=None, coercing to 0")
        result["evaluation"]["score"] = 0
    entry["results"].append(result)
    ev = result.get("evaluation", {})
    return AnswerResponse(
        question=result["question"],
        category=result.get("category"),
        transcript=result.get("transcript", ""),
        evaluation=EvaluationSchema(
            score=ev.get("score"),
            feedback=ev.get("feedback"),
            status=ev.get("status"),
            message=ev.get("message"),
        ),
    )


@router.get("/{session_id}/answers", response_model=list[AnswerResponse])
async def get_answers(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    results = []
    for r in entry["results"]:
        ev = r.get("evaluation", {})
        results.append(AnswerResponse(
            question=r["question"],
            category=r.get("category"),
            transcript=r.get("transcript", ""),
            evaluation=EvaluationSchema(
                score=ev.get("score"),
                feedback=ev.get("feedback"),
                status=ev.get("status"),
                message=ev.get("message"),
            ),
        ))
    return results


@router.get("/{session_id}/summary", response_model=SummaryResponse)
async def get_summary(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    summary = sess.finish(entry["results"])
    results = []
    for r in summary.get("results", entry["results"]):
        ev = r.get("evaluation", {})
        results.append(AnswerResponse(
            question=r["question"],
            category=r.get("category"),
            transcript=r.get("transcript", ""),
            evaluation=EvaluationSchema(
                score=ev.get("score"),
                feedback=ev.get("feedback"),
                status=ev.get("status"),
                message=ev.get("message"),
            ),
        ))
    return SummaryResponse(
        total_questions=summary["total_questions"],
        evaluated=summary["evaluated"],
        final_score=summary["final_score"],
        results=results,
    )
