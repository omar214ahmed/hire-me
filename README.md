# HireMe Platform — ATS + HR Microservices

Two independent FastAPI services, orchestrated (not merged) via a single
root `docker-compose.yml`:

```
                 Browser
                    │
                    ▼
          HR FastAPI Service (:8001)
                    │  HTTP REST (httpx, with retries)
                    ▼
          ATS FastAPI Service (:8000)
                    │  asyncpg
                    ▼
                PostgreSQL
```

- **ATS** (`ats/`) is the single source of truth for job data. It owns
  PostgreSQL and is otherwise unchanged from the original project — JD
  upload/extraction, candidate matching, the screening console, Celery
  worker, everything still works exactly as before.
- **HR** (`hr/`) generates and evaluates interview questions via a local
  Ollama LLM + faster-whisper. It has **no database of its own for job
  data** and **never opens a connection to the ATS's Postgres** — the only
  way it learns about a job is by calling the ATS over HTTP.

```
hireme-platform/
├── docker-compose.yml        # orchestrates everything
├── .env                      # compose-level substitution (POSTGRES_PASSWORD, ONNX_MODELS_DIR)
├── ats/
│   ├── Dockerfile
│   ├── .dockerignore
│   └── app/                  # original ATS project, unmerged
│       ├── main.py, routers/, pipeline/, helpers/, migrations/, ...
│       └── .env              # ATS app-level settings
└── hr/
    ├── Dockerfile             # was missing — added
    ├── .dockerignore
    └── app/                   # original HR project (src/), unmerged
        ├── main.py, routers/, llm/, interview/, helpers/, ...
        ├── integrations/      # NEW: ATSClient (httpx + retries)
        └── .env
```

---

## What was added for the integration (everything else is untouched)

### ATS side — one new endpoint

`GET /api/v1/jobs/{job_id}/summary` — a **new, additional** endpoint
(`ats/app/routers/jobs.py`, backed by `pipeline/storage.job_to_hr_view` and
`pipeline/schemas.JobHRView`). It returns only what HR needs:

```json
{
  "id": "a1b2c3d4e5f6",
  "job_title": "Machine Learning Engineer",
  "hard_skills": ["Python", "TensorFlow", "PyTorch", "Docker"]
}
```

This is deliberately a **new path**, not a change to the existing
`GET /api/v1/jobs/{job_id}` (which returns the full internal record —
`jd_text`, `extracted`, `query`, `jd_embedding`, `created_at` — and is
already used by the screening console and matching flow). Reshaping that
endpoint in place would have been a breaking change for existing
consumers; a purpose-built endpoint keeps the HR contract stable even if
the ATS's internal `extracted` schema changes later.

`job_to_hr_view()` reads `extracted["required job title"]` (first item, if
any) and `extracted["required hard skill"]` — the canonical keys the JD
extraction pipeline already produces (see `pipeline/jd_processor.py`'s
`LABEL_KEY_MAP`) — and is a small pure function, unit-tested independent
of the database.

### HR side

- **`hr/app/integrations/ats_client.py`** — `ATSClient`, an async httpx
  wrapper around `GET {ATS_API_URL}/api/v1/jobs/{job_id}/summary` with:
  - a configurable timeout per attempt,
  - retry with exponential backoff on network errors and 5xx responses
    only (a 404 is not retried — it means "no such job", not "try again"),
  - `ATSNotFoundError` (job doesn't exist) vs. `ATSUnavailableError`
    (couldn't reach ATS / it kept failing) as distinct exception types, so
    the router can map each to the right HTTP status.
- **`POST /sessions/from-job/{job_id}`** (`hr/app/routers/sessions.py`) —
  fetches the job from ATS, then creates a session exactly like the
  existing `POST /sessions`, with `role`/`skills` auto-filled from
  `job_title`/`hard_skills` instead of typed by hand. Maps:
  - ATS 404 → HR 404 ("job not found in ATS")
  - ATS unreachable/5xx after retries → HR 503 ("ATS unavailable, try again")

  The existing `POST /sessions` (manual role/skills) and the rest of the
  session lifecycle (`/questions`, `/answers`, `/summary`) are untouched —
  `from-job` only changes *how* a session gets its role/skills.
- **`helpers/config.py`** — added `ATS_API_URL`, `ATS_REQUEST_TIMEOUT`,
  `ATS_MAX_RETRIES`, `ATS_RETRY_BACKOFF_SECONDS`.
- **`main.py`** — builds one shared `ATSClient` at startup, attaches it to
  `app.state.ats_client`, and closes its connection pool on shutdown via a
  `lifespan` handler (previously the app had none).
- **`Dockerfile`** — the HR project had none; added one (Python 3.11-slim,
  installs `requirements.txt`, runs `uvicorn` on `:8001`). Also added
  `ffmpeg` (needed by faster-whisper/ctranslate2 for audio decoding) and a
  `.dockerignore`.
- **`requirements.txt`** — added `httpx` and `tenacity` (retry helper) for
  the ATS client.

### Root

- **`docker-compose.yml`** — builds and runs `postgres`, `redis`, `ats`,
  `ats-worker` (Celery), `ollama` (the HR service's LLM runtime), and `hr`,
  all on one bridge network (`hireme-net`), with healthchecks and
  `depends_on: condition: service_healthy` so `hr` doesn't start racing
  ahead of `ats`.
- **`.env`** (repo root) — `POSTGRES_PASSWORD` and `ONNX_MODELS_DIR`. These
  are read by *docker-compose itself* for `${...}` substitution, which is
  why they live here rather than inside `ats/app/.env` (env files nested
  inside a service folder are never seen by compose's own variable
  substitution — only `env_file:` entries pass them *into* a container).

---

## Environment variables

**ATS** (`ats/app/.env`, plus compose-injected overrides):

| Variable | Meaning | Compose override |
|---|---|---|
| `DATABASE_URL` | Postgres DSN | `postgresql://hireme:${POSTGRES_PASSWORD}@postgres:5432/hireme` |
| `REDIS_URL` | Celery broker | `redis://redis:6379/0` |
| `GLINER_ONNX_DIR` / `BGE_M3_ONNX_DIR` / `RERANKER_ONNX_DIR` | local ONNX model paths | `/app/onnx_models/...` (host folder mounted via `ONNX_MODELS_DIR`) |
| `APP_NAME`, `FILE_ALLOWED_TYPES`, ... | app settings | not overridden |

**HR** (`hr/app/.env`, plus compose-injected overrides):

| Variable | Meaning | Compose override |
|---|---|---|
| `ATS_API_URL` | Base URL of the ATS service | `http://ats:8000` (service name, **never** `localhost`) |
| `ATS_REQUEST_TIMEOUT` | Per-attempt HTTP timeout (seconds) | `5.0` |
| `ATS_MAX_RETRIES` | Retry attempts on network/5xx errors | `3` |
| `ATS_RETRY_BACKOFF_SECONDS` | Base for exponential backoff | `0.5` |
| `LLM_OLLAMA_MODEL` | Ollama model tag | `qwen2.5:3b` |
| `WHISPER_DEVICE` | `cpu` (default here) or `gpu` | left as `.env` value |

**Root** (`.env`, compose-only):

| Variable | Meaning |
|---|---|
| `POSTGRES_PASSWORD` | Shared between the `postgres` service and `ats`/`ats-worker`'s `DATABASE_URL` |
| `ONNX_MODELS_DIR` | Host path to your exported GLiNER/BGE-M3/reranker ONNX models |

---

## Docker networking

All services share the compose file's default network (`hireme-net`). Inside
containers, service **names** are the hostnames:

- HR → ATS: `http://ats:8000`
- HR → Ollama: `http://ollama:11434` (configured inside `hr/app/.env` /
  `llm/providers/ollama_provider.py` via `langchain_ollama.ChatOllama`,
  which defaults to `http://localhost:11434` — override this if you point
  HR at the `ollama` container instead of a host-installed Ollama)
- ATS/worker → Postgres: `postgres:5432`
- ATS/worker → Redis: `redis:6379`

`localhost` inside any one container refers to that container itself, never
another service — this is why `ATS_API_URL` must stay as the service name in
Docker, and why `helpers/config.py`'s default is already `http://ats:8000`
rather than `http://localhost:8000`.

---

## Local testing

### Prerequisites

- Docker + Docker Compose.
- A host folder with the ATS's exported ONNX models (GLiNER / BGE-M3 /
  reranker — see `ats/app/README.md` for how these are produced), pointed
  to by `ONNX_MODELS_DIR` in the root `.env`. This is unchanged from the
  original ATS project.
- `docker pull ollama/ollama` will happen automatically; after the stack is
  up, pull the model once:
  ```bash
  docker compose exec ollama ollama pull qwen2.5:3b
  ```

### Bring the stack up

```bash
cd hireme-platform
docker compose up --build
```

- ATS Swagger: http://localhost:8000/docs
- ATS screening console (unchanged): http://localhost:8000/console/
- HR Swagger: http://localhost:8001/docs

### End-to-end flow

1. **ATS already has a job.** Either use the console at
   `http://localhost:8000/console/`, or:
   ```bash
   curl -X POST http://localhost:8000/api/v1/jobs/ \
     -H "Content-Type: application/json" \
     -d '{"description": "<a job description of 20+ characters>"}'
   ```
   Note the returned `job_id`.

2. **Confirm the HR-facing view directly:**
   ```bash
   curl http://localhost:8000/api/v1/jobs/<job_id>/summary
   # {"id": "<job_id>", "job_title": "...", "hard_skills": [...]}
   ```

3. **Create an HR session from that job (this is the integration):**
   ```bash
   curl -X POST http://localhost:8001/sessions/from-job/<job_id>
   # {"id": "<session_id>", "role": "...", "skills": "...", "job_id": "<job_id>", ...}
   ```
   Internally this is: `HR → GET http://ats:8000/api/v1/jobs/<job_id>/summary → build session`.

4. **Generate a question** (uses the auto-filled role/skills):
   ```bash
   curl -X POST http://localhost:8001/sessions/<session_id>/questions
   ```

5. Submit an answer, get a summary, etc. — same as the original HR API
   (`POST /sessions/{id}/answers`, `GET /sessions/{id}/summary`), unaffected
   by this integration.

### Error handling, on purpose

- `POST /sessions/from-job/does-not-exist` → **404** (ATS said no such job).
- If the `ats` container is stopped/unhealthy, `POST /sessions/from-job/...`
  retries a few times (exponential backoff) and then returns **503** rather
  than hanging or crashing the HR process.

---

## Design notes / production best practices applied

- **Clean separation, one direction of dependency.** HR depends on ATS's
  HTTP API; ATS has zero knowledge of HR. Either service can be redeployed,
  rescaled, or replaced independently.
- **Purpose-built integration contract.** The `/summary` endpoint and
  `JobHRView` schema are a small, stable surface — changes to the ATS's
  internal extraction pipeline (e.g. new GLiNER labels) don't ripple into
  HR unless `job_to_hr_view()` is deliberately updated.
- **Resilience at the network boundary.** Retries only apply to errors that
  retrying can plausibly fix (connection failures, timeouts, 5xx); a 404 or
  other 4xx fails fast. Errors are typed (`ATSNotFoundError` vs.
  `ATSUnavailableError`) so the caller maps them to the right response
  instead of guessing from a status code.
- **No shared database, no shared code.** Each service keeps its own
  Dockerfile, `requirements.txt`, and `.env`; the only shared artifact is
  the root `docker-compose.yml` and the HTTP contract itself.
- **Config via environment, not hardcoding.** `ATS_API_URL` is a setting,
  not a literal in the client code — swap it for a different environment
  (staging, a different port, a different host) without a code change.
- **Healthchecks + explicit startup ordering.** `hr` waits for `ats` to
  report healthy (not just "container started") before starting, avoiding
  a class of "worked on my machine, failed in CI" flakiness.
- **Backward compatible.** The ATS's existing `GET /{job_id}`, matching,
  candidate upload, and console endpoints are byte-for-byte unchanged. The
  HR's existing `POST /sessions`, `/questions`, `/answers`, `/summary`
  endpoints are unchanged; `from-job` is additive.

## Things to know before running this for real

- The ATS side is unchanged and still needs its ONNX models
  (GLiNER/BGE-M3/reranker) present on the host — this integration doesn't
  touch that requirement.
- `WHISPER_DEVICE` defaults to `cpu` here so the stack runs without a GPU.
  Set it to `gpu`/`cuda` only if the host has one and the container image
  has the matching CUDA/ctranslate2 runtime — the upstream `python:3.11-slim`
  base image used here does not.
- The HR service's `OllamaProvider`/`WhisperLoader` load at **import time**
  (`main.py`), so `hr` will fail to start if Ollama isn't reachable or the
  Whisper model can't be loaded — this is pre-existing behavior, not
  something this integration changed.
# hire-me
