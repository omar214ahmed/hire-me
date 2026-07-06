# HireMe ATS — Pipeline & API Documentation

This document explains how the matching pipeline works end-to-end and how
the codebase is organized. Storage is PostgreSQL + pgvector (`pipeline/storage.py`,
via `asyncpg`); the ML models can run either from Hugging Face/PyTorch or
from local ONNX INT8 exports (`pipeline/models.py`).

---

## 1. High-Level Flow

```
CV upload  ──▶  validate type+size ──▶ save file ──▶ extract text (pdf/docx)
                                              │
                                              ▼
                              check content (char count, page count)
                                              │
                                              ▼
                                      clean text (ftfy/regex)
                                              │
                                              ▼
                          split into sections + regex field extraction
                          (skills, experience, education, languages, email, phone)
                                              │
                                              ▼
                       embed CV text (BGE-M3) ──▶ store candidate record ─┐
                                                                          │
JD submit (raw string) ──▶ split into sections (requirements/skills/...) │
                                              │                          │
                                              ▼                          │
                                  GLiNER NER extraction                  │
                                              │                          │
                                              ▼                          │
                       embed JD query (BGE-M3) ──▶ store job record ─────┤
                                                                          │
                                  POST /jobs/{id}/match  ◀────────────────
                                              │
                  1. COS similarity (pgvector, in Postgres) reduces the
                     candidate pool — e.g. 10,000 → 150 (shortlist_limit)
                  2. BGE-reranker-v2-m3 cross-encoder reranks the shortlist
                  3. top_n reranked results = final score
                     (rule-based skills/exp/edu/lang match is attached to
                      each result as explainability only — it does not
                      affect the ranking)
                                              │
                                              ▼
                                ranked shortlist response
```

There are two independent ingestion paths (CVs and JDs) that both write
structured records to Postgres, and a third endpoint that reads both and
produces a ranking. A fourth, `GET /jobs/{id}/shortlist`, exposes the
cosine-similarity step on its own — a fast, DB-only nearest-neighbor query
that lets you preview or pick the shortlist size before paying for the
reranker.

---

## 2. Folder Structure

```
src/
├── main.py                     # FastAPI app, router registration, model warm-up
├── helpers/
│   ├── config.py               # Settings (env vars) — pydantic-settings
│   └── database.py             # asyncpg connection pool (get_pool/close_pool)
├── preprocessing/               # File-level handling (existing, unchanged logic)
│   ├── validator.py             # FileService: validates type/size, builds file paths
│   ├── extractor.py             # extract_pdf / extract_docx → raw text
│   ├── cleaner.py                # clean_resume_text → normalized text
│   └── dispatcher.py             # orchestrates extractor + cleaner per file type
├── pipeline/
│   ├── models.py                 # singleton loader: GLiNER, BGE-M3, reranker
│   │                              #   USE_ONNX=true  → local INT8 ONNX models
│   │                              #   USE_ONNX=false → HF/torch models
│   ├── onnx_embedder.py           # ONNX Runtime BGE-M3 wrapper (CLS pooling + L2 norm)
│   ├── onnx_reranker.py            # ONNX Runtime BGE-reranker wrapper (sigmoid score)
│   ├── jd_processor.py             # JD string → sections → GLiNER labels
│   ├── cv_processor.py              # CV text → sections + regex facts
│   ├── matcher.py                    # hard label match (explainability only, doesn't rank)
│   ├── ranker.py                      # cosine shortlist -> cross-encoder rerank -> final score
│   ├── storage.py                      # Postgres + pgvector persistence (asyncpg)
│   ├── celery_app.py                    # Celery app, Redis broker/backend
│   ├── tasks.py                          # run_match — the async matching task
│   └── schemas.py                         # pydantic request/response models
├── routers/
│   ├── base.py                    # GET /api/v1 — health/info
│   ├── candidates.py               # POST /api/v1/candidates/upload
│   ├── jobs.py                      # POST/GET /api/v1/jobs
│   └── matching.py                  # POST /api/v1/jobs/{id}/match, GET /tasks/{id}
├── .env.example                      # copy to .env and fill in real values
├── Dockerfile                         # shared image for app + worker services
├── docker-compose.yml                  # postgres+pgvector, redis, app, worker
├── .dockerignore
├── requirements.txt                     # prod deps
├── requirements-dev.txt                  # + pytest/httpx for testing
├── migrations/
│   └── schema.sql                         # applied automatically by docker-compose
├── tests/
│   ├── conftest.py                         # dummy env vars for import-time Settings
│   └── test_pipeline_logic.py               # unit tests, no external deps needed
├── scripts/
│   └── smoke_test.sh                         # end-to-end HTTP test against the running stack
└── files/candidates/                          # uploaded resumes land here (gitignored)
```

Postgres tables (`jobs`, `candidates`) with `vector` columns for embeddings
must exist before running this — created automatically by Docker Compose,
or manually via `migrations/schema.sql` — see §5.

---

## 3. The Pipeline Stages in Detail

### Stage 1 — CV ingestion (`preprocessing/` → `pipeline/cv_processor.py`)
1. `validator.FileService.validate_uploaded_file` checks MIME type + size
   (before any parsing — cheap, rejects obviously-bad uploads fast).
2. `extractor.extract_pdf` / `extract_docx` pulls raw text (pdfplumber →
   PyMuPDF fallback for PDFs; python-docx for Word, paragraphs + tables),
   plus a page count (native for PDFs, estimated from paragraph volume
   for DOCX).
3. `validator.FileService.validate_extracted_content` — the "check type"
   gate: rejects the file if the extracted text is too short (empty/
   scanned doc), too long (probably not a resume), or has too many pages
   — `FILE_MIN_CHARS` / `FILE_MAX_CHARS` / `FILE_MAX_PAGES` in `.env`.
4. `cleaner.clean_resume_text` fixes encoding issues, strips zero-width
   chars, page footers, collapses whitespace.
5. `cv_processor.parse_cv()`:
   - `split_cv_sections()` — regex-detects section headers (experience,
     education, skills, languages, summary) and buckets the lines under
     each.
   - Field extractors pull `email`, `phone`, `years_of_experience` (sums
     date ranges like `2019 to Present`), `degrees`, `skills`, `languages`.
6. CV text is embedded once (BGE-M3) and the result is handed to
   `storage.save_candidate()` along with the embedding.

### Stage 2 — JD ingestion (`pipeline/jd_processor.py`)
1. Raw JD string comes in via `POST /api/v1/jobs`.
2. `split_jd_sections()` — the same header-detection approach as
   `cv_processor.split_cv_sections`, using the *same category keys*
   (skills, experience, education, languages, summary) plus a JD-only
   `requirements` bucket, so the two sides of the pipeline stay
   comparable.
3. `GLiNER` (zero-shot NER, `urchade/gliner_medium-v2.1`) runs over the
   requirements/skills/experience/education/languages sections (falling
   back to the full JD text if section-splitting found nothing useful),
   with a fixed label set (job title, years of experience, hard skill,
   soft skill, education degree, field of study, language, nice-to-have
   tool, location, job type).
4. Entities are bucketed into canonical keys (see `LABEL_KEY_MAP` in
   `jd_processor.py`).
5. `build_jd_query()` builds a short text string from the extracted
   fields plus the raw `requirements`/`skills` section text — this is
   what gets embedded.
6. JD query is embedded once (BGE-M3) and the result is handed to
   `storage.save_job()` along with the embedding.

### Stage 3 — Matching (`pipeline/ranker.py` + `pipeline/matcher.py`)
Triggered by `POST /api/v1/jobs/{job_id}/match`. This dispatches to a
**Celery task** (`pipeline/tasks.py: run_match`), queued via **Redis**
(`pipeline/celery_app.py`) — the endpoint returns a `task_id` immediately
(202 Accepted); poll `GET /api/v1/jobs/tasks/{task_id}` for the result.
Requires a Celery worker process running alongside the FastAPI app
(`celery -A pipeline.celery_app worker`) and a reachable Redis instance.

1. **Shortlist (cosine similarity, in Postgres)** —
   `storage.get_top_candidates_by_similarity()` runs
   `ORDER BY cv_embedding <=> $1 LIMIT $2` directly against Postgres,
   using the `idx_candidates_cv_embedding` HNSW index — this is the
   "reduce 10,000 CVs down to ~150" step, and it never re-embeds or pulls
   every row into Python. `shortlist_limit` in the request body controls
   how many candidates come back (default 150). If `candidate_ids` is
   passed explicitly instead, that fixed set is used and scored with
   `ranker.semantic_scores()` (plain cosine similarity in Python, since
   it's a small fixed set rather than a DB-wide query) — this is also
   what `GET /api/v1/jobs/{job_id}/shortlist` uses to expose this step as
   its own read-only endpoint.
2. **Rerank (`ranker.rerank_candidates`)** — the shortlist is scored with
   the cross-encoder `BAAI/bge-reranker-v2-m3` (JD text vs CV text
   pairs). This is the **final score** — `top_n` of the reranked results
   are returned, sorted by `rerank_score`.
3. **Explainability (`matcher.hard_match`)** — a rule-based
   skills/experience/education/language score (weighted 40/25/20/15) is
   still computed and attached to each result, so a recruiter can see
   *why* a candidate ranked where it did — but it no longer affects
   filtering or ordering; cosine similarity decides the shortlist, and
   the reranker decides the final order.

Models are loaded once at startup (`main.py`'s `lifespan` calls
`ModelRegistry.warm_up()`), not per-request — and, in the Celery worker
process, again on first task execution, since the worker is a separate
process with its own `ModelRegistry` singleton. First request/task after
deploy will still be the slowest if the models weren't pre-downloaded
(HF path) or if the ONNX files are on slow storage (ONNX path).

---

## 4. API Endpoints

| Method | Path                              | Purpose                                      |
|--------|-----------------------------------|-----------------------------------------------|
| GET    | `/api/v1/`                        | health/app info                              |
| POST   | `/api/v1/candidates/upload`       | upload CV (pdf/docx) → parsed + stored       |
| POST   | `/api/v1/jobs`                    | submit JD string → labeled + stored          |
| GET    | `/api/v1/jobs/{job_id}`           | fetch a stored job                           |
| GET    | `/api/v1/jobs`                    | list stored jobs                             |
| GET    | `/api/v1/jobs/{job_id}/shortlist` | cosine-similarity shortlist only (no rerank), `?limit=N` |
| POST   | `/api/v1/jobs/{job_id}/match`     | queue shortlist+rerank (Celery) → returns `task_id` |
| GET    | `/api/v1/jobs/tasks/{task_id}`    | poll match task status/result                |

`POST /api/v1/jobs` body:
```json
{ "description": "We are looking for a Senior ML Engineer with 5+ years..." }
```

`GET /api/v1/jobs/{job_id}/shortlist?limit=150` — read-only, no Celery
task needed (single indexed Postgres query):
```json
{
  "job_id": "abc123",
  "count": 150,
  "shortlist": [
    { "candidate_id": "cand1", "filename": "jane_doe.pdf", "semantic_score": 0.812 }
  ]
}
```

`POST /api/v1/jobs/{job_id}/match` body (all fields optional):
```json
{ "candidate_ids": null, "shortlist_limit": 150, "top_n": 20 }
```
Omit `candidate_ids` to shortlist against every stored candidate via the
cosine-similarity step above; pass it to match against a fixed set
instead (e.g. the output of a previous `/shortlist` call), skipping the
DB-wide query. `shortlist_limit` is ignored when `candidate_ids` is set.

---

## 5. PostgreSQL + pgvector

Storage is Postgres, accessed via `asyncpg` (`helpers/database.py`,
`pipeline/storage.py`). Embeddings are stored as native `vector` columns
(pgvector extension) so `jd_embedding`/`cv_embedding` can be cast with
`::vector` on insert and read back directly — this is what lets
`ranker.py`'s semantic step skip re-embedding on every match request.

The schema lives in `migrations/schema.sql`. **Docker Compose applies it
automatically** on first start (mounted into Postgres's
`docker-entrypoint-initdb.d`). Running manually (no Docker)? Apply it
yourself before starting the app:

```bash
psql "$DATABASE_URL" -f migrations/schema.sql
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE jobs (
    job_id        VARCHAR(32) PRIMARY KEY,
    jd_text       TEXT NOT NULL,
    extracted     JSONB NOT NULL,     -- GLiNER label buckets
    query         TEXT NOT NULL,      -- embedding query string
    jd_embedding  vector(1024),       -- BGE-M3 dense embedding
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE candidates (
    candidate_id  VARCHAR(32) PRIMARY KEY,  -- = file_id from upload
    file_id       VARCHAR(64) NOT NULL,
    filename      VARCHAR(255) NOT NULL,
    parsed        JSONB NOT NULL,       -- full cv_processor.parse_cv() output
    cv_embedding  vector(1024),         -- BGE-M3 dense embedding
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- recommended once volumes grow:
CREATE INDEX idx_jobs_created_at ON jobs (created_at);
CREATE INDEX idx_candidates_created_at ON candidates (created_at);
CREATE INDEX idx_candidates_cv_embedding ON candidates
    USING hnsw (cv_embedding vector_cosine_ops);
```

`vector(1024)` matches BGE-M3's output dimension — keep this consistent
regardless of whether the model is running via HF/torch or ONNX, since
both produce the same 1024-dim dense vector.

`pipeline/jd_processor.py`, `pipeline/cv_processor.py`, `pipeline/matcher.py`,
`pipeline/ranker.py`, `pipeline/models.py` are all storage-agnostic — they
take/return plain dicts and have no knowledge of the storage backend.
Routers only talk to `storage.py`'s functions, never to the DB directly.

---

## 6. Model Backend: ONNX (local INT8) vs HF/Torch

`pipeline/models.py`'s `ModelRegistry` supports two backends, switched with
one setting:

- **`USE_ONNX=true`** *(default)* — loads local INT8 ONNX exports via
  `pipeline/onnx_embedder.py` / `pipeline/onnx_reranker.py` (using
  `optimum[onnxruntime]`) and GLiNER's built-in ONNX loader. Faster CPU
  inference, smaller memory footprint, no network access needed. Requires
  the six `*_ONNX_DIR` / `*_ONNX_FILE` settings below to point at valid
  export folders (each must contain the `.onnx` file **and** its
  tokenizer/config files — see `optimum`'s `save_pretrained()` output).
- **`USE_ONNX=false`** — original HF/torch path (`gliner`, `FlagEmbedding`),
  downloads or loads full-precision weights from `MODELS_CACHE_DIR`.

Both paths expose the same interface (`.ner()`, `.embedder()`, `.reranker()`
with `.encode()`/`.compute_score()` matching the original torch libraries'
signatures), so nothing else in the pipeline needs to know which backend is
active.

⚠️ Before relying on the ONNX path in production, spot-check a few
embeddings/rerank scores against the torch path on identical input — INT8
quantization is a lossy approximation, and this hasn't been benchmarked
against the reference implementation in this repo.

---

## 7. Environment Variables

| Variable             | Purpose                                              |
|-----------------------|-------------------------------------------------------|
| `APP_NAME` / `APP_VERSION` | App metadata                                    |
| `FILE_ALLOWED_TYPES`  | Allowed MIME types for CV upload                      |
| `FILE_MAX_SIZE`        | Max upload size in bytes                             |
| `FILE_MIN_CHARS` / `FILE_MAX_CHARS` | Extracted-text length bounds (rejects empty/scanned or too-long docs) |
| `FILE_MAX_PAGES`       | Max page count (native for PDF, estimated for DOCX)  |
| `DEFAULT_SHORTLIST_LIMIT` | Default `shortlist_limit` for `/match` (cosine step) |
| `DEFAULT_RERANK_TOP_N`  | Default `top_n` for `/match` (final reranked results) |
| `DATABASE_URL`          | Postgres connection string (pgvector extension required) |
| `MODELS_CACHE_DIR`     | HuggingFace cache dir, used when `USE_ONNX=false`     |
| `MODELS_LOCAL_ONLY`    | `true` once HF models are pre-downloaded (torch path only) |
| `GLINER_MODEL`          | HF model id for JD NER (torch path only)             |
| `EMBEDDER_MODEL`         | HF model id for semantic embedding (torch path only) |
| `RERANKER_MODEL`         | HF model id for final reranking (torch path only)    |
| `STORAGE_DIR`             | Where uploaded resume files land                    |
| `USE_ONNX`                 | `true`/`false` — select ONNX vs torch model backend |
| `GLINER_ONNX_DIR` / `GLINER_ONNX_FILE` | Local GLiNER ONNX export dir + filename |
| `BGE_M3_ONNX_DIR` / `BGE_M3_ONNX_FILE` | Local BGE-M3 ONNX export dir + filename |
| `RERANKER_ONNX_DIR` / `RERANKER_ONNX_FILE` | Local reranker ONNX export dir + filename |

See `.env.example` for a filled-out template — copy it to `.env` and set
real values. **Never commit `.env`** (it's already in `.gitignore`).

---

## 8. Running the Project

### Option A — Docker Compose (recommended, single command)

This starts Postgres+pgvector, Redis, the FastAPI app, and the Celery
worker together. The Postgres container automatically applies
`migrations/schema.sql` on its first start (via `docker-entrypoint-initdb.d`)
— no manual migration step needed.

```bash
cp .env.example .env
# edit .env: at minimum set POSTGRES_PASSWORD and ONNX_MODELS_DIR
# (or set USE_ONNX=false to use the HF/torch model path instead)

docker compose up --build
```

Watch the logs until the app has finished loading models:
```bash
docker compose logs -f app
# wait for: "All pipeline models loaded (ONNX)."
```

The API is then available at `http://localhost:8000` (interactive docs at
`http://localhost:8000/docs`).

To tear down (keeping data):
```bash
docker compose down
```
To tear down and wipe the database too:
```bash
docker compose down -v
```

**Notes:**
- `ONNX_MODELS_DIR` in `.env` must point at the **host** folder containing
  your `gliner/`, `bge-m3-int8/`, `bge-reranker-int8/` exports — it gets
  mounted read-only into both the `app` and `worker` containers.
- `DATABASE_URL` / `REDIS_URL` in `.env` are only used for the non-Docker
  path (Option B below) — `docker-compose.yml` overrides both with the
  in-network service names (`postgres`, `redis`) automatically.
- First `docker compose up --build` will be slow (installing torch,
  onnxruntime, gliner, etc. into the image). Subsequent runs reuse the
  cached image layer unless `requirements.txt` changes.

### Option B — Manual (no Docker)

```bash
pip install -r requirements.txt
cp .env.example .env   # edit DATABASE_URL, ONNX paths, etc.

# apply the schema once, against your own Postgres instance:
psql "$DATABASE_URL" -f migrations/schema.sql

uvicorn main:app --reload --app-dir src

# separate terminal — required for POST /jobs/{id}/match to work:
celery -A pipeline.celery_app worker --loglevel=info
```

Redis must be running and reachable at `REDIS_URL`. If `USE_ONNX=false`,
the first run will also download GLiNER / bge-m3 / bge-reranker-v2-m3 (a
few GB) — set `MODELS_LOCAL_ONLY=true` afterward so subsequent starts
don't hit the network.

---

## 9. Testing

**Unit tests** (`tests/test_pipeline_logic.py`) cover the storage/model-agnostic
logic — hard matching (`matcher.py`), CV section/field extraction
(`cv_processor.py`), and JD query building (`jd_processor.py`). These run
with zero external dependencies (no Postgres, Redis, or ML models needed):

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

**Smoke test** (`scripts/smoke_test.sh`) exercises the real running stack
end-to-end over HTTP — health check, job creation, listing, and (if you
pass a resume file) candidate upload + the full async matching flow via
Celery. Run this after `docker compose up`:

```bash
chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh                       # health + job creation
./scripts/smoke_test.sh path/to/resume.pdf     # + upload + matching
```

**What's not covered by either:** correctness of the ONNX INT8 models
themselves (embedding/rerank quality vs the reference torch models) — see
§6's note on spot-checking those before relying on them in production.
