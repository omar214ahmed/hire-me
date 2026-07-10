from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL : str
    APP_NAME: str
    APP_VERSION: str


    FILE_ALLOWED_TYPES: list[str]
    FILE_MAX_SIZE: int  # bytes

    # -----------------------------
    # Extracted-content checks (post-parse, pre-section-split)
    # -----------------------------
    FILE_MIN_CHARS: int = 50        # below this: empty/scanned/corrupt doc
    FILE_MAX_CHARS: int = 50000     # above this: probably not a resume
    FILE_MAX_PAGES: int = 10        # PDF page cap; DOCX page count is estimated

    # -----------------------------
    # Matching pipeline defaults
    # -----------------------------
    DEFAULT_SHORTLIST_LIMIT: int = 150   # cosine-similarity narrows the DB down to this many
    DEFAULT_RERANK_TOP_N: int = 20       # final cross-encoder reranked results returned

    # -----------------------------
    # Pipeline / ML models
    # -----------------------------
    MODELS_CACHE_DIR: str = "~/.cache/huggingface/hub"
    MODELS_LOCAL_ONLY: bool = False  # set True once models are pre-downloaded

    GLINER_MODEL: str = "urchade/gliner_medium-v2.1"
    EMBEDDER_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # -----------------------------
    # ONNX (local INT8 models)
    # -----------------------------
    # USE_ONNX defaults to False: the previous defaults for the three
    # *_ONNX_DIR settings below were hardcoded to one developer's home
    # directory (/home/omar-ahmed/onnx_models/...). On any other
    # machine - another dev's laptop, CI, a fresh deployment - that path
    # simply doesn't exist, so GLiNER.from_pretrained(..., local_files_
    # only=True) fails, the legacy extraction path produces zero
    # entities, and every JD field silently comes back empty (this is
    # the bug that motivated pipeline/jd_keyword_extractor.py as a
    # dependency-free safety net - see that module's docstring).
    #
    # Set USE_ONNX=True and the three dirs below explicitly (env var or
    # .env) once the ONNX models have actually been downloaded/quantized
    # on *this* machine and their real path is known. Until then,
    # USE_ONNX=False uses the plain torch/HF Hub loading path instead
    # (MODELS_CACHE_DIR above), which downloads/caches normally and
    # isn't tied to any one person's filesystem layout.
    USE_ONNX: bool = False

    GLINER_ONNX_DIR: str = "./onnx_models/gliner"
    GLINER_ONNX_FILE: str = "model_int8.onnx"

    BGE_M3_ONNX_DIR: str = "./onnx_models/bge-m3-int8"
    BGE_M3_ONNX_FILE: str = "model_quantized.onnx"

    RERANKER_ONNX_DIR: str = "./onnx_models/bge-reranker-int8"
    RERANKER_ONNX_FILE: str = "model_quantized.onnx"

    # -----------------------------
    # JD extraction (v2 architecture)
    # -----------------------------
    # "auto"   -> jd_router decides legacy vs. llm per-JD (recommended)
    # "llm"    -> always use the structured-output LLM path
    # "legacy" -> always use the regex/GLiNER path (old behavior, cheapest)
    JD_EXTRACTION_MODE: str = "auto"

    # Anthropic API key for the structured-output extractor. If unset/empty,
    # the router transparently falls back to the legacy path for every JD
    # (see jd_llm_extractor.is_available()) instead of failing requests.
    ANTHROPIC_API_KEY: str = ""
    JD_LLM_MODEL: str = "claude-sonnet-4-6"
    JD_LLM_MAX_TOKENS: int = 2000
    JD_LLM_TIMEOUT_SECONDS: float = 30.0

    # Router thresholds ("auto" mode): a JD only qualifies for the cheap
    # legacy path if it is short, has enough recognized headers, and looks
    # single-language. Anything else routes to the LLM path.
    JD_ROUTER_MAX_LEGACY_CHARS: int = 1500
    JD_ROUTER_MIN_RECOGNIZED_HEADERS: int = 2

    # Confidence gate: a JD extraction with more than this many low-confidence
    # / missing core fields is flagged needs_review instead of shipped silently.
    JD_MIN_CONFIDENCE: float = 0.5
    JD_MAX_LOW_CONFIDENCE_FIELDS_BEFORE_REVIEW: int = 2

    # -----------------------------
    # Storage (JD / CV labels)
    # -----------------------------
    STORAGE_DIR: str = "storage"

    # -----------------------------
    # Celery / Redis
    # -----------------------------
    REDIS_URL: str = "redis://localhost:6379/0"

    model_config = {
        "env_file": ".env",
        # POSTGRES_PASSWORD / ONNX_MODELS_DIR in .env are consumed by
        # docker-compose for variable substitution only (see the comment
        # block at the bottom of .env) — they aren't Settings fields.
        # Without extra="ignore", pydantic-settings hard-fails on them
        # with "Extra inputs are not permitted", which was breaking
        # Settings() (and therefore the whole app + test suite) on any
        # machine that loads this .env as-is.
        "extra": "ignore",
    }


@lru_cache()
def get_settings():
    return Settings()
