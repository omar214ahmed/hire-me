# HireMe Platform

HireMe is a two-service hiring platform made of two independent FastAPI microservices that only ever talk to each other over HTTP:

```
┌───────────────────────┐        HTTP        ┌───────────────────────────┐
│   ATS Service         │ <───────────────── │   HR Service              │
│   (port 8000)         │  GET /jobs/{id}    │   (port 8001)             │
│                       │      /summary      │                           │
│  - Resume parsing     │ ─────────────────> │  - AI interview questions │
│  - JD extraction      │                    │  - Speech-to-text answers │
│  - Candidate ranking  │                    │  - LLM answer scoring     │
│  - PostgreSQL+pgvector│                    │  - No direct DB access    │
└──────────┬────────────┘                    └──────────┬────────────────┘
           │                                            │
      PostgreSQL                                       Ollama
      (pgvector, HNSW)                             (chat + embedding models)
           │
         Redis
     (Celery queue)
```

- **ATS** (`ats/`) — the source of truth for jobs and candidates. Parses resumes and job descriptions, embeds them, and ranks candidates against a job using a semantic-search + cross-encoder-rerank pipeline.
- **HR** (`hr/`) — an AI interview assistant. It generates interview questions (optionally seeded from an ATS job), transcribes spoken answers, and scores them with a local LLM. It never touches the ATS database directly — it only calls the ATS's `/summary` endpoint over HTTP.

Both services are wired together (plus Postgres, Redis, and Ollama) via the root `docker-compose.yml`.

---

## Table of Contents

- [Architecture](#architecture)
- [Full File Structure](#full-file-structure)
- [Quick Start (Docker Compose)](#quick-start-docker-compose)
- [Configuration](#configuration)
- [ATS Service](#ats-service)
  - [Matching Pipeline](#matching-pipeline)
  - [ATS API Reference](#ats-api-reference)
  - [Database Schema](#database-schema)
- [HR Service](#hr-service)
  - [Interview Pipeline](#interview-pipeline)
  - [Question Diversity: Skill Planner + Uniqueness Guard](#question-diversity-skill-planner--uniqueness-guard)
  - [HR API Reference](#hr-api-reference)
- [Web Consoles](#web-consoles)
- [Running Tests](#running-tests)
- [Models Used](#models-used)

---

## Architecture

**ATS pipeline (per candidate/job pair):**

1. **Preprocessing** — PDF/DOCX resumes are extracted, cleaned, and validated (file type, size, char/page counts).
2. **JD extraction** (`jd_processor.py`) — the job description is cleaned, split into sections (skills / experience / education / languages / requirements), and run through **GLiNER** (zero-shot NER) to pull out structured requirements (title, years of experience, hard/soft skills, degrees, languages, location, job type, "nice to have" extras).
3. **CV extraction** (`cv_processor.py`) — resumes are split into the same section types with regex, and structured facts (email, phone, years of experience, degrees, skills, languages) are extracted.
4. **Embedding** — both the JD query string and the CV text are embedded once, at ingestion time, using **BGE-M3** (1024-dim dense vectors), and stored in Postgres via **pgvector**.
5. **Shortlist** — cosine similarity search directly in Postgres (`pgvector` HNSW index) narrows the full candidate pool down to a configurable shortlist (default 150), without any model inference.
6. **Rerank** — the shortlist is scored by the **BGE-Reranker-v2-m3** cross-encoder, which produces the primary ranking signal.
7. **Hard-match explainability** — a separate rule-based comparison (skills 40% / experience 25% / education 20% / languages 15%) is computed for the same JD/CV pair. It does **not** decide the shortlist or the base ranking order; it's blended into the final score only as a corrective — pulling the score down when hard requirements are present and poorly met — and is always returned to the client so a recruiter can see *why* a candidate ranked where it did (see [Matching Pipeline](#matching-pipeline)).
8. **Async execution** — the shortlist/rerank steps run inside a **Celery** worker (backed by Redis), so the API returns a `task_id` immediately and the caller polls for results.

**HR pipeline (per interview session):**

1. A session is created either manually (role + skills typed in) or **from an ATS job** (`POST /sessions/from-job/{job_id}`), in which case the HR service calls the ATS's `/api/v1/jobs/{job_id}/summary` endpoint to auto-fill the role and required skills.
2. A **local chat LLM (via Ollama)** generates an interview question, deliberately targeted at a specific (skill, angle) pair so consecutive questions don't overlap, and checked against every question already asked in the session for semantic duplicates.
3. The question is classified into one of `technical` / `problem_solving` / `behavioral`.
4. The candidate's spoken answer (audio upload) is transcribed locally with **faster-whisper**.
5. The transcript is scored by the LLM (0-10 + written feedback).
6. Sessions accumulate a full history of question/answer/score pairs and can be summarized into a final average score.

---

## Full File Structure

```
hireme-platform/
├── docker-compose.yml            # Orchestrates postgres, redis, ats, ats-worker, ollama, hr
├── .env                          # Root-level compose variables (POSTGRES_PASSWORD, ONNX_MODELS_DIR)
│
├── ats/                                     # ── ATS microservice ──
│   ├── Dockerfile
│   ├── .dockerignore
│   └── app/
│       ├── main.py                          # FastAPI app, lifespan model warm-up, router mounting
│       ├── requirements.txt
│       ├── requirements-dev.txt
│       ├── .env                             # ATS service configuration
│       ├── helpers/
│       │   ├── config.py                    # Pydantic Settings (models, DB, Celery, file limits)
│       │   └── database.py                  # asyncpg connection pool (get_pool / close_pool)
│       ├── migrations/
│       │   └── schema.sql                   # Postgres schema (jobs, candidates, pgvector, HNSW index)
│       ├── preprocessing/
│       │   ├── validator.py                 # FileService: upload validation, unique filepaths
│       │   ├── dispatcher.py                # Routes PDF/DOCX to the right extractor, then cleans text
│       │   ├── extractor.py                 # PDF (pdfplumber/PyMuPDF) + DOCX (python-docx) text extraction
│       │   └── cleaner.py                   # Resume text normalization
│       ├── pipeline/
│       │   ├── models.py                    # ModelRegistry: singleton GLiNER / BGE-M3 / reranker loader (ONNX or torch)
│       │   ├── onnx_embedder.py             # ONNX INT8 runtime wrapper for BGE-M3
│       │   ├── onnx_reranker.py             # ONNX INT8 runtime wrapper for BGE-Reranker-v2-m3
│       │   ├── jd_processor.py              # JD cleaning, section splitting, GLiNER extraction, query building
│       │   ├── cv_processor.py              # CV section splitting + regex field extraction
│       │   ├── matcher.py                   # Rule-based hard-match scoring (skills/experience/education/lang)
│       │   ├── ranker.py                    # Semantic shortlisting + cross-encoder reranking + score blending
│       │   ├── storage.py                   # All Postgres reads/writes (jobs, candidates, similarity search)
│       │   ├── schemas.py                   # Pydantic request/response models (incl. JobHRView for the HR service)
│       │   ├── celery_app.py                # Celery app configuration (Redis broker/backend)
│       │   └── tasks.py                     # `run_match` Celery task (the async matching pipeline)
│       ├── routers/
│       │   ├── base.py                      # GET /api/v1/  (app name/version)
│       │   ├── candidates.py                # POST /api/v1/candidates/upload
│       │   ├── jobs.py                      # /api/v1/jobs (create/get/list/summary)
│       │   └── matching.py                  # /api/v1/jobs/{id}/shortlist, /match, /tasks/{id}
│       ├── static/
│       │   ├── index.html
│       │   └── ats_console.html             # 3-step screening console (post JD -> screen CVs -> shortlist)
│       ├── scripts/
│       │   └── smoke_test.sh
│       └── tests/                           # pytest suite (integration + unit)
│           ├── conftest.py
│           ├── test_api_integration.py
│           ├── test_hr_integration_endpoint.py
│           ├── test_jd_ner_fix.py
│           ├── test_jd_optional_marker_and_job_type_fix.py
│           ├── test_jd_section_splitter_fix.py
│           ├── test_jd_text_cleaning.py
│           ├── test_onnx_model_config.py
│           └── test_pipeline_logic.py
│
└── hr/                                      # ── HR (interview) microservice ──
    ├── Dockerfile
    ├── .dockerignore
    └── app/
        ├── main.py                         # FastAPI app, mounts /console, wires LLM + embeddings + ATS client
        ├── requirements.txt
        ├── .env / .env.example             # HR service configuration
        ├── helpers/
        │   ├── config.py                   # Pydantic Settings (Whisper, LLM, embeddings, ATS integration)
        │   └── logger.py                   # Logging setup
        ├── integrations/
        │   └── ats_client.py               # HTTP client to ATS's /jobs/{id}/summary, with retries/backoff
        ├── interview/
        │   └── interview_session.py        # InterviewSession: generate -> classify -> evaluate -> finish
        ├── llm/
        │   ├── base.py                     # LLM provider interface
        │   ├── llm_interface.py
        │   ├── chains.py                   # Chains: wires prompts + LLM + structured-output parsing together
        │   ├── prompts.py                  # Prompt templates (question generation, classification, evaluation)
        │   ├── questions_generator.py      # QuestionsGenerator: orchestrates planner + uniqueness guard
        │   ├── skill_planner.py            # SkillCoveragePlanner: deterministic (skill, angle) queue
        │   ├── question_similarity.py      # QuestionUniquenessGuard: embedding-based near-duplicate rejection
        │   ├── classification.py           # ClassificationQuestion (technical/problem_solving/behavioral)
        │   ├── evaluator.py                # Evaluator (LLM-scored answer -> EvaluationSchema)
        │   ├── transcript.py               # Transcript: wraps Whisper transcription
        │   └── providers/
        │       ├── ollama_provider.py      # Chat LLM + embeddings providers (both via Ollama)
        │       └── faster_whisper_provider.py  # Speech-to-text provider (faster-whisper)
        ├── routers/
        │   └── sessions.py                 # All /sessions endpoints
        ├── schemas/
        │   ├── api_schema.py               # CreateSessionRequest / SessionResponse / QuestionResponse / etc.
        │   ├── question_schema.py          # QuestionSchema
        │   ├── classification_schema.py    # ClassificationSchema
        │   └── evaluation_schema.py        # EvaluationSchema
        ├── static/
        │   ├── index.html
        │   ├── hr_console.html             # Interview console UI
        │   └── hr_console.js
        └── tests/
            ├── conftest.py
            ├── test_ats_client.py
            └── test_sessions_from_job.py
```

---

## Quick Start (Docker Compose)

**Prerequisites:** Docker + Docker Compose, and a folder of pre-exported ONNX INT8 models for GLiNER, BGE-M3, and BGE-Reranker-v2-m3 (see [Models Used](#models-used)).

1. Set the repo-root `.env`:
   ```env
   POSTGRES_PASSWORD=changeme
   ONNX_MODELS_DIR=/path/to/your/onnx_models   # must contain gliner/, bge-m3-int8/, bge-reranker-int8/
   ```
2. Start everything:
   ```bash
   docker compose up --build
   ```
   This brings up, in dependency order: `postgres` (with pgvector + schema auto-applied on first run), `redis`, `ats` (API, port 8000), `ats-worker` (Celery), `ollama` (port 11434), and `hr` (API, port 8001).
3. Pull both Ollama models the first time — the chat model (matches `LLM_OLLAMA_MODEL`, default `qwen2.5:3b`) and the dedicated embedding model used by HR's question-uniqueness guard:
   ```bash
   docker exec -it <ollama_container_name> ollama pull qwen2.5:3b
   docker exec -it <ollama_container_name> ollama pull nomic-embed-text
   ```
4. Open the consoles:
   - ATS screening console: `http://localhost:8000/console/`
   - HR interview console: `http://localhost:8001/console/`

---

## Configuration

### ATS (`ats/app/.env`)

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` / `APP_VERSION` | `ats` / `0.1` | App metadata |
| `DATABASE_URL` | — | Postgres connection string (overridden in-network by compose) |
| `FILE_ALLOWED_TYPES` | pdf, docx, doc | Accepted resume MIME types |
| `FILE_MAX_SIZE` | `5242880` (5 MB) | Max upload size in bytes |
| `FILE_MIN_CHARS` / `FILE_MAX_CHARS` | `50` / `50000` | Extracted-text sanity bounds |
| `FILE_MAX_PAGES` | `10` | Max resume page count |
| `DEFAULT_SHORTLIST_LIMIT` | `150` | Candidates pulled by cosine similarity before reranking |
| `DEFAULT_RERANK_TOP_N` | `20` | Final reranked results returned |
| `USE_ONNX` | `true` | Use local ONNX INT8 models instead of downloading torch weights |
| `GLINER_ONNX_DIR`, `BGE_M3_ONNX_DIR`, `RERANKER_ONNX_DIR` (+ `*_ONNX_FILE`) | — | Paths to exported ONNX model folders + filenames |
| `GLINER_MODEL`, `EMBEDDER_MODEL`, `RERANKER_MODEL` | `urchade/gliner_medium-v2.1`, `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3` | HF model ids used when `USE_ONNX=false` |
| `MODELS_CACHE_DIR` / `MODELS_LOCAL_ONLY` | `~/.cache/huggingface/hub` / `false` | HF cache dir + offline flag (torch path only) |
| `STORAGE_DIR` | `storage` | Where uploaded resume files land |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker/backend |

### HR (`hr/app/.env`)

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` / `APP_VERSION` | `HR system` / `1.0.0` | App metadata |
| `ALLOWED_MIME_TYPES` | mp3/wav/ogg/aac/flac/mp4 audio | Accepted answer-audio MIME types |
| `MAX_FILE_SIZE_MB` | `10` | Max audio upload size |
| `WHISPER_MODEL_SIZE` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `small` / `cpu` / `int8` | faster-whisper transcription settings |
| `LLM_OLLAMA_MODEL` | `qwen2.5:3b` | Ollama chat model used for question generation, classification, and scoring |
| `LLM_MAX_NEW_TOKENS` / `LLM_TEMPERATURE` / `LLM_TIMEOUT` | `300` / `0.3` / `3000` | Chat LLM generation settings |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server (must be the Docker **service name**, not `localhost`) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Dedicated Ollama embedding model, used only by the question-uniqueness guard — pull separately: `ollama pull nomic-embed-text` |
| `QUESTION_SIMILARITY_THRESHOLD` | `0.86` | Cosine similarity above which a new question counts as a near-duplicate of one already asked |
| `QUESTION_MAX_GENERATION_ATTEMPTS` | `3` | Retries per skill/angle slot before accepting the least-similar candidate seen |
| `ATS_API_URL` | `http://ats:8000` | ATS service base URL (must be the Docker **service name**, not `localhost`) |
| `ATS_REQUEST_TIMEOUT` | `5.0` | Per-attempt HTTP timeout to ATS |
| `ATS_MAX_RETRIES` | `3` | Retries on network errors / 5xx (not on 404) |
| `ATS_RETRY_BACKOFF_SECONDS` | `0.5` | Exponential backoff base |

The HR chat model is loaded with an explicit `num_ctx=8192`. Left unset, it silently falls back to Ollama's server default (4096), which — combined with the per-session question history — caused the context window to overflow and get truncated a few questions into an interview.

---

## ATS Service

Base URL: `http://localhost:8000`

### Matching Pipeline

- **Semantic score** — cosine similarity between JD and CV BGE-M3 embeddings; used to shortlist (in Postgres via pgvector), or, for an explicit candidate list, computed directly in Python.
- **Rerank score** — BGE-Reranker-v2-m3 cross-encoder score over `(jd_text, cv_text)` pairs; this is the primary ranking signal.
- **Hard-match score** — rule-based score (skills 40%, experience 25%, education 20%, languages 15%) comparing GLiNER-extracted JD requirements against regex-extracted CV facts. Always attached to results as a `breakdown` for explainability.
- **Final score** — if the JD has no explicit hard requirements, `final_score = rerank_score`. If hard requirements exist and the hard-match score is below 0.5, the final score is pulled down: `0.6 × rerank + 0.4 × hard_match`. Otherwise: `0.85 × rerank + 0.15 × hard_match`.

Models are loaded once at startup (`main.py`'s `lifespan` calls `ModelRegistry.warm_up()`), and again on first task execution inside the Celery worker process, since it's a separate process with its own `ModelRegistry` singleton.

### ATS API Reference

All routes are prefixed with `/api/v1`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root health check — `{"status": "running"}` |
| `GET` | `/api/v1/` | App name/version |
| `POST` | `/api/v1/candidates/upload` | Upload a resume (PDF/DOCX). Validates, extracts, cleans, parses (regex), embeds (BGE-M3), and stores the candidate. Returns parsed fields (email, phone, years of experience, degrees, skills, languages). |
| `POST` | `/api/v1/jobs/` | Create a job from a raw JD string. Body: `{"description": "<text>"}` (min 20 chars). Runs GLiNER extraction, builds the embedding query, embeds it, and stores the job. Returns `job_id`, `extracted` labels, and the `query` string. |
| `GET` | `/api/v1/jobs/{job_id}` | Full internal job record (jd_text, extracted labels, embedding, etc.) — used by the ATS console/matcher. |
| `GET` | `/api/v1/jobs/{job_id}/summary` | Minimal job view (`id`, `job_title`, `hard_skills`) — this is the endpoint the **HR service** calls; kept separate so the internal job schema can evolve independently. |
| `GET` | `/api/v1/jobs/` | List all jobs. |
| `GET` | `/api/v1/jobs/{job_id}/shortlist?limit=150` | Pgvector cosine-similarity shortlist only (no reranking) — lets you preview/tune how many candidates would go into the expensive rerank step. |
| `POST` | `/api/v1/jobs/{job_id}/match` | Dispatches the full matching pipeline (shortlist + rerank) to a Celery worker. Body (`MatchRequest`, all optional): `candidate_ids`, `shortlist_limit` (default 150), `top_n` (default 20). Returns `202` with a `task_id` immediately. |
| `GET` | `/api/v1/jobs/tasks/{task_id}` | Poll a match task. Returns `{"status": "pending" \| "running" \| "done" \| "failed"}`, with `result` once done. |

**Upload response signals:** `CV_PREPROCESSED_SUCCESS`, `FILE_TYPE_NOT_SUPPORTED`, `FILE_SIZE_EXCEEDED`, `FILE_CONTENT_TOO_SHORT`, `FILE_CONTENT_TOO_LONG`, `FILE_TOO_MANY_PAGES`, `FILE_UPLOAD_FAILED`, `PREPROCESSING_FAILED`, `CV_LABELING_FAILED`, `CV_EMBEDDING_FAILED`.

**Job creation response signals:** `JD_PROCESSED_SUCCESS`, `JD_TEXT_TOO_SHORT_OR_MISSING`, `JD_EXTRACTION_FAILED`, `JD_EMBEDDING_FAILED`.

**GLiNER JD labels extracted:** job title, years of experience, programming language/technical skill, soft skill/personality trait, education degree, field of study, spoken/written language, preferred (optional) technology tool, city/country/region, job type/work arrangement.

### Database Schema

Postgres + `pgvector`, applied automatically on first container start from `ats/app/migrations/schema.sql`:

```sql
jobs (
  job_id        VARCHAR(32) PRIMARY KEY,
  jd_text       TEXT NOT NULL,
  extracted     JSONB NOT NULL,      -- GLiNER label buckets
  query         TEXT NOT NULL,       -- embedding query string
  jd_embedding  vector(1024),        -- BGE-M3 dense embedding
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
)

candidates (
  candidate_id  VARCHAR(32) PRIMARY KEY,  -- = file_id from upload
  file_id       VARCHAR(64) NOT NULL,
  filename      VARCHAR(255) NOT NULL,
  parsed        JSONB NOT NULL,       -- full cv_processor.parse_cv() output
  cv_embedding  vector(1024),         -- BGE-M3 dense embedding
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
)
```

An HNSW index (`vector_cosine_ops`) on `candidates.cv_embedding` powers the shortlist query.

---

## HR Service

Base URL: `http://localhost:8001`

### Interview Pipeline

`InterviewSession` drives each session through four steps:

1. `generate_question(role, skills)` — the chat LLM writes one question, deliberately aimed at a specific (skill, angle) pair via `QuestionsGenerator`, and rejected/retried if it's a near-duplicate of anything already asked.
2. `classify_current_question()` — the LLM classifies the question as `technical`, `problem_solving`, or `behavioral`. Defaults to `technical` if the call fails.
3. `evaluate_answer(audio_bytes, suffix)` — transcribes the audio locally (faster-whisper), rejects answers under 3 words as `invalid_answer` before wasting an LLM call, otherwise scores the transcript with the LLM (0-10 + feedback).
4. `finish(results)` — averages all valid scores into a final `"x/10"` summary.

All three LLM chains (`question_chain`, `classification_chain`, `evaluation_chain`) are built with `llm.with_structured_output(..., method="json_mode")`. `qwen2.5:3b` writes plain JSON straight into the message content rather than making a real tool call, so relying on the default tool-calling structured-output mode left `tool_calls` empty and silently discarded a valid JSON response sitting in `content` — `json_mode` fixes that by asking the model for raw JSON directly. The question and evaluation chains also pass `include_raw=True`, so a parsing failure surfaces the raw output and error for logging instead of raising, letting the caller fall back cleanly (`score=0`, a default category, or a fallback question) instead of a 500 error.

### Question Diversity: Skill Planner + Uniqueness Guard

Two mechanisms work together, each covering a failure mode the other doesn't:

**`SkillCoveragePlanner` (`llm/skill_planner.py`)** — the *structural* half. It parses the skills string into a list, then builds a deterministic queue of every `(skill, angle)` combination, cycling through a fixed set of angles (core concepts, trade-offs, debugging, performance, best practices, comparisons, real-world experience, system architecture) before ever repeating a skill — guaranteeing topical breadth before depth.

**`QuestionUniquenessGuard` (`llm/question_similarity.py`)** — the *semantic* half. An LLM told "don't repeat yourself" will happily reword the same question ("What is a race condition?" vs. "What causes two threads to conflict over shared state?") — duplicates a plain string/set comparison would never catch. The guard embeds every accepted question with the dedicated `EMBEDDING_MODEL` (`nomic-embed-text` by default, served by Ollama — separate from the chat model), compares each new candidate's cosine similarity against **every** previously accepted vector in the session, and rejects it if any similarity meets or exceeds `QUESTION_SIMILARITY_THRESHOLD` (default `0.86`). It fails open if the embedding call errors — better an occasional duplicate slip through than a broken interview.

One `QuestionsGenerator` (and therefore one planner + one guard) is created per interview session, so this state naturally resets for the next interview.

### HR API Reference

All routes are prefixed with `/sessions`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "app": ..., "version": ...}` |
| `GET` | `/` | Redirects to `/console/` |
| `POST` | `/sessions` | Create a session manually. Body: `{"role": "...", "skills": "..."}`. |
| `POST` | `/sessions/from-job/{job_id}` | Create a session seeded from an ATS job: fetches `{ATS_API_URL}/api/v1/jobs/{job_id}/summary` and uses its `job_title`/`hard_skills` as role/skills. Returns `404` if the ATS has no such job, `503` if the ATS is unreachable. |
| `GET` | `/sessions/{session_id}` | Session state: role, skills, job_id, current question/category, question history, results count. |
| `DELETE` | `/sessions/{session_id}` | Delete a session (`204`). |
| `POST` | `/sessions/{session_id}/questions` | Generate the next interview question + classify its category. |
| `GET` | `/sessions/{session_id}/current-question` | Get the currently active question (404 if none generated yet). |
| `POST` | `/sessions/{session_id}/answers` | Submit a spoken answer as an audio file upload (`multipart/form-data`, field `file`). Transcribes and scores it; returns question, transcript, and evaluation. |
| `GET` | `/sessions/{session_id}/answers` | List all answers/evaluations recorded so far in the session. |
| `GET` | `/sessions/{session_id}/summary` | Final summary: total questions, evaluated count, average `final_score` (`"x/10"`), and the full results list. |

**`EvaluationSchema`:** `score` (0-10, int), `feedback` (str), `status` (optional — e.g. `error`, `invalid_answer`), `message` (optional).

---

## Web Consoles

- **ATS console** (`http://localhost:8000/console/`) — a vanilla JS, 3-tab UI: (1) post a job description, (2) upload/screen resumes, (3) view the ranked shortlist with score breakdowns. Defaults its API base URL to the ATS server it's served from.
- **HR console** (`http://localhost:8001/console/`) — pick or preview an ATS job (or type a role/skills manually) → create a session → generate a question → record/upload an answer → see the transcript + score → repeat → view the final summary. The ATS and HR base URLs are both editable in the console header.

---

## Running Tests

**ATS:**
```bash
cd ats/app
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
Covers JD text cleaning, section splitting, NER extraction fixes, ONNX model config, the core pipeline logic, full API integration, and the HR-facing `/summary` integration endpoint.

**HR:**
```bash
cd hr/app
pip install -r requirements.txt
pytest
```
Covers the ATS HTTP client (retries/backoff/error handling) and the job-based session creation flow.

---

## Models Used

| Model | Service | Role | Notes |
|---|---|---|---|
| `urchade/gliner_medium-v2.1` (GLiNER) | ATS | Zero-shot NER on job descriptions | Extracts job title, years of experience, hard/soft skills, degrees, languages, location, job type. Runs via local ONNX INT8 export by default (`USE_ONNX=true`). |
| `BAAI/bge-m3` | ATS | Dense embedding (1024-dim) | Embeds JD queries and CV text once, at ingestion time, for the cosine-similarity shortlist step. |
| `BAAI/bge-reranker-v2-m3` | ATS | Cross-encoder reranking | Scores `(jd_text, cv_text)` pairs directly; produces the primary ranking signal for the final shortlist. |
| `qwen2.5:3b` (configurable via `LLM_OLLAMA_MODEL`) | HR | Chat LLM | Generates questions, classifies them, scores answers. Served locally via Ollama, loaded with `num_ctx=8192` to avoid mid-session context truncation. |
| `nomic-embed-text` (configurable via `EMBEDDING_MODEL`) | HR | Embedding model | ~137M parameters, 768-dim output. Used exclusively by the question-uniqueness guard. Served locally via Ollama; pull separately with `ollama pull nomic-embed-text`. |
| `faster-whisper` (`WHISPER_MODEL_SIZE`, default `small`) | HR | Speech-to-text | Transcribes candidate audio answers locally; no cloud speech API involved. |

Both services run fully offline once their models are pulled/downloaded — no external API keys are required anywhere in the platform. ATS defaults to local **INT8-quantized ONNX** exports of GLiNER, BGE-M3, and BGE-Reranker-v2-m3 (set `USE_ONNX=false` in `ats/app/.env` to fall back to full torch weights downloaded from Hugging Face instead). The expected local folder layout (mounted read-only into the `ats`/`ats-worker` containers via `ONNX_MODELS_DIR`) is:

```
onnx_models/
├── gliner/                 # model_int8.onnx
├── bge-m3-int8/             # model_quantized.onnx
└── bge-reranker-int8/       # model_quantized.onnx
```