"""
Sets dummy values for required Settings fields so test collection doesn't
need a real .env, Ollama, or Whisper model — the tests here only exercise
the ATS integration layer (integrations/ats_client.py, the
/sessions/from-job/{job_id} route), not the LLM/transcription chains.
"""
import os

os.environ.setdefault("APP_NAME", "hr-test")
os.environ.setdefault("APP_VERSION", "0.0.0-test")
os.environ.setdefault("ALLOWED_MIME_TYPES", '["audio/wav"]')
os.environ.setdefault("MAX_FILE_SIZE_MB", "10")
os.environ.setdefault("WHISPER_MODEL_SIZE", "small")
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("WHISPER_COMPUTE_TYPE", "int8")
os.environ.setdefault("LLM_OLLAMA_MODEL", "qwen2.5:3b")
os.environ.setdefault("ATS_API_URL", "http://ats-test")
