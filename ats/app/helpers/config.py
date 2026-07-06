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
    USE_ONNX: bool = True

    GLINER_ONNX_DIR: str = "/home/omar-ahmed/onnx_models/gliner"
    GLINER_ONNX_FILE: str = "model_int8.onnx"

    BGE_M3_ONNX_DIR: str = "/home/omar-ahmed/onnx_models/bge-m3-int8"
    BGE_M3_ONNX_FILE: str = "model_quantized.onnx"

    RERANKER_ONNX_DIR: str = "/home/omar-ahmed/onnx_models/bge-reranker-int8"
    RERANKER_ONNX_FILE: str = "model_quantized.onnx"

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
