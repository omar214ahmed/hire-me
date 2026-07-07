from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    ALLOWED_MIME_TYPES: list[str]
    MAX_FILE_SIZE_MB: int
    WHISPER_MODEL_SIZE: str
    WHISPER_DEVICE: str
    WHISPER_COMPUTE_TYPE: str
    LLM_MAX_NEW_TOKENS: int = 250
    LLM_TEMPERATURE: float = 0.6
    LLM_TIMEOUT: int = 300
    LLM_OLLAMA_MODEL: str
    # Base URL of the Ollama server. Inside Docker Compose this must be
    # the service name ("http://ollama:11434"), never localhost — the hr
    # container and the ollama container are different network
    # namespaces, so "localhost" inside hr refers to hr itself, not the
    # ollama container (same reasoning as ATS_API_URL below).
    OLLAMA_BASE_URL: str = "http://ollama:11434"

    # -----------------------------
    # Question generation (uniqueness/diversity guard)
    # -----------------------------
    # Dedicated embedding model used only to measure semantic similarity
    # between generated questions (see llm/question_similarity.py). This is
    # NOT the chat model above — pull it separately in the Ollama
    # container: `ollama pull nomic-embed-text`.
    EMBEDDING_MODEL: str = "nomic-embed-text"
    # Cosine similarity (0-1) above which a newly generated question is
    # considered a near-duplicate of one already asked in the session.
    QUESTION_SIMILARITY_THRESHOLD: float = 0.86
    # How many times to ask the LLM again for a given skill/angle slot
    # before giving up and accepting the least-similar candidate seen.
    QUESTION_MAX_GENERATION_ATTEMPTS: int = 3

    # -----------------------------
    # ATS integration (microservice)
    # -----------------------------
    # Base URL of the ATS service. Inside Docker Compose this must be the
    # service name ("http://ats:8000"), never localhost/127.0.0.1 — the
    # HR container and the ATS container are different network
    # namespaces, and "localhost" inside the HR container refers to the
    # HR container itself.
    ATS_API_URL: str = "http://ats:8000"
    ATS_REQUEST_TIMEOUT: float = 5.0   # seconds, per HTTP attempt
    ATS_MAX_RETRIES: int = 3           # retry only on network errors / 5xx
    ATS_RETRY_BACKOFF_SECONDS: float = 0.5  # base for exponential backoff

    class Config:
        env_file = ".env" 

def get_settings():
    return Settings()