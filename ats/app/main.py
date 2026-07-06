from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers.base import base_router
from routers.candidates import candidate_router
from routers.jobs import jobs_router
from routers.matching import matching_router
from helpers.database import get_pool, close_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load GLiNER / BGE-M3 / reranker once at startup so the first
    # request isn't slow. Comment out during local dev if you don't
    # have the models cached yet.
    from pipeline.models import ModelRegistry
    ModelRegistry.warm_up()
    yield
    # Clean up DB pool on shutdown
    await close_pool()

app = FastAPI(
    title="HireMe ATS",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend origin(s) in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base_router)
app.include_router(candidate_router)
app.include_router(jobs_router)
app.include_router(matching_router)

# Screening console UI (the 3-page frontend: post JD -> screen CVs ->
# ranked shortlist). Once this server is running, open
# http://localhost:8000/console/ in a browser — the console already
# defaults its API base URL to this same server.
app.mount("/console", StaticFiles(directory="static", html=True), name="console")


@app.get("/")
async def root():
    return {
        "status": "running"
    }
