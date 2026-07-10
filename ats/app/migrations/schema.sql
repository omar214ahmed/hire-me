-- HireMe ATS — initial schema
-- Run this against the database DATABASE_URL points to before starting the app.
-- Requires a Postgres instance with the pgvector extension available
-- (e.g. the `pgvector/pgvector` Docker image, or Neon/Supabase which ship it).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs (
    job_id          VARCHAR(32) PRIMARY KEY,
    jd_text         TEXT NOT NULL,
    extracted       JSONB NOT NULL,     -- flat {canonical_key: [values]} buckets (legacy shape; extract_from_jd_v2 also uses this)
    query           TEXT NOT NULL,      -- embedding query string
    jd_embedding    vector(1024),       -- BGE-M3 dense embedding (NULL until embedded)
    -- v2 extraction metadata (pipeline.schemas.JDExtraction, minus the
    -- flat values which are already in `extracted`): per-field
    -- confidence, extraction_method ("legacy_regex" | "llm_structured"),
    -- needs_review, routing_reason. Nullable so existing rows created
    -- before this column existed remain valid.
    extraction_meta JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Lets a reviewer dashboard cheaply list "needs_review" JDs without
-- scanning/parsing the whole extraction_meta blob per row.
CREATE INDEX IF NOT EXISTS idx_jobs_needs_review
    ON jobs (((extraction_meta->>'needs_review')::boolean))
    WHERE extraction_meta IS NOT NULL;

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id  VARCHAR(32) PRIMARY KEY,  -- = file_id from upload
    file_id       VARCHAR(64) NOT NULL,
    filename      VARCHAR(255) NOT NULL,
    parsed        JSONB NOT NULL,       -- full cv_processor.parse_cv() output
    cv_embedding  vector(1024),         -- BGE-M3 dense embedding (NULL until embedded)
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at);
CREATE INDEX IF NOT EXISTS idx_candidates_created_at ON candidates (created_at);

-- Nearest-neighbor index for cosine similarity search — actively used by
-- storage.get_top_candidates_by_similarity() (`ORDER BY cv_embedding <=> $1`)
-- to shortlist candidates directly in Postgres instead of pulling every
-- embedding into Python. Required once candidate volume grows past a few
-- thousand rows.
CREATE INDEX IF NOT EXISTS idx_candidates_cv_embedding ON candidates
    USING hnsw (cv_embedding vector_cosine_ops);
