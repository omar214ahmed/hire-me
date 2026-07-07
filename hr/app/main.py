import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from helpers import get_settings
from helpers.logger import setup_logger
from integrations import ATSClient
from llm.chains import Chains
from llm.transcript import Transcript
from llm.providers.faster_whisper_provider import WhisperLoader
from llm.providers.ollama_provider import OllamaProvider, OllamaEmbeddingsProvider


setup_logger()
logger = logging.getLogger("hr_interview")

settings = get_settings()

llm_provider = OllamaProvider(settings)
embeddings_provider = OllamaEmbeddingsProvider(settings)
chains = Chains(llm_provider.get_llm(), embeddings_provider.get_embeddings())
whisper = WhisperLoader(settings)
transcript = Transcript(whisper)

ats_client = ATSClient(
    base_url=settings.ATS_API_URL,
    timeout=settings.ATS_REQUEST_TIMEOUT,
    max_retries=settings.ATS_MAX_RETRIES,
    backoff_base=settings.ATS_RETRY_BACKOFF_SECONDS,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Release the ATS HTTP connection pool on shutdown.
    await ats_client.aclose()


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.sessions: dict = {}
app.state.chains = chains
app.state.transcript = transcript
app.state.ats_client = ats_client
app.state.settings = settings

from routers.sessions import router
app.include_router(router)

# Interview console UI: pick/preview an ATS job (or type role+skills by
# hand) -> create a session -> generate a question -> record/upload the
# candidate's answer -> see transcript + score -> repeat -> final summary.
# Defaults to talking to this same HR service and to http://localhost:8000
# for the ATS; both are editable in the console header for other setups.
app.mount("/console", StaticFiles(directory="static", html=True), name="console")


@app.get("/", include_in_schema=False)
def root():
    # The console is the primary UI for this service; bare host:port
    # should land there instead of 404ing.
    return RedirectResponse(url="/console/")


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}