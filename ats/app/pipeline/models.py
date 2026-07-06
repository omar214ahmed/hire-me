"""
Lazy singleton loader for the ML models used across the pipeline:
  - GLiNER          -> JD requirement extraction (NER)
  - BGE-M3           -> dense embeddings for semantic matching
  - BGE-Reranker-v2-m3 -> cross-encoder final reranking

Models are loaded once per process and cached, since they're expensive
to load (multi-second cold start). FastAPI's lifespan/startup event
calls `warm_up()` so the first request isn't slow.
"""

import os
import shutil
import warnings
from pathlib import Path
from functools import lru_cache
import tempfile

from helpers.config import get_settings

warnings.filterwarnings("ignore")

_settings = get_settings()

# MODELS_CACHE_DIR is expected to already point at the "hub" cache dir
# itself (default "~/.cache/huggingface/hub"), matching what
# HF_HUB_CACHE/TRANSFORMERS_CACHE expect. HF_HOME is different: HF's own
# libraries treat HF_HOME as the *parent* dir and append "/hub" underneath
# it themselves, so pointing HF_HOME at the same "hub" path as the others
# made real downloads land one level deeper, at ".../huggingface/hub/hub/...",
# while anything checking the expected ".../huggingface/hub" path (e.g. `du`)
# saw almost nothing there and looked stuck/broken.
MODELS_DIR = Path(_settings.MODELS_CACHE_DIR).expanduser()
os.environ.setdefault("HF_HOME", str(MODELS_DIR.parent))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODELS_DIR.parent))


def ensure_model_config_file(model_dir: str | os.PathLike, source_name: str = "gliner_config.json") -> bool:
    """Create a config.json file for ONNX model directories that only ship gliner_config.json."""
    model_dir = Path(model_dir)
    target = model_dir / "config.json"
    if target.exists():
        return False

    source = model_dir / source_name
    if not source.exists():
        return False

    try:
        shutil.copy2(source, target)
        return True
    except OSError:
        # The model directory may be mounted read-only inside Docker.
        # In that case, we still allow startup to proceed and rely on the
        # model loader to use the source file directly when possible.
        return False


class ModelRegistry:
    """Holds lazily-initialized model instances."""

    _ner_model = None
    _embedder = None
    _reranker = None

    @classmethod
    def ner(cls):
        if cls._ner_model is None:
            from gliner import GLiNER

            if _settings.USE_ONNX:
                print(f"Loading GLiNER (ONNX INT8) from {_settings.GLINER_ONNX_DIR} ...")
                ensure_model_config_file(_settings.GLINER_ONNX_DIR)
                cls._ner_model = GLiNER.from_pretrained(
                    _settings.GLINER_ONNX_DIR,
                    load_onnx_model=True,
                    onnx_model_file=_settings.GLINER_ONNX_FILE,
                    local_files_only=True,
                )
            else:
                print("Loading GLiNER (torch)...")
                cls._ner_model = GLiNER.from_pretrained(
                    _settings.GLINER_MODEL,
                    local_files_only=_settings.MODELS_LOCAL_ONLY,
                )
        return cls._ner_model

    @classmethod
    def embedder(cls):
        if cls._embedder is None:
            if _settings.USE_ONNX:
                from pipeline.onnx_embedder import ONNXBGEM3Embedder
                print(f"Loading BGE-M3 embedder (ONNX INT8) from {_settings.BGE_M3_ONNX_DIR} ...")
                cls._embedder = ONNXBGEM3Embedder(
                    model_dir=_settings.BGE_M3_ONNX_DIR,
                    onnx_file=_settings.BGE_M3_ONNX_FILE,
                )
            else:
                from FlagEmbedding import BGEM3FlagModel
                print("Loading BGE-M3 embedder (torch)...")
                cls._embedder = BGEM3FlagModel(
                    _settings.EMBEDDER_MODEL,
                    use_fp16=True,
                    local_files_only=_settings.MODELS_LOCAL_ONLY,
                )
        return cls._embedder

    @classmethod
    def reranker(cls):
        if cls._reranker is None:
            if _settings.USE_ONNX:
                from pipeline.onnx_reranker import ONNXReranker
                print(f"Loading BGE reranker (ONNX INT8) from {_settings.RERANKER_ONNX_DIR} ...")
                cls._reranker = ONNXReranker(
                    model_dir=_settings.RERANKER_ONNX_DIR,
                    onnx_file=_settings.RERANKER_ONNX_FILE,
                )
            else:
                from FlagEmbedding import FlagReranker
                print("Loading BGE reranker (torch)...")
                cls._reranker = FlagReranker(
                    _settings.RERANKER_MODEL,
                    use_fp16=True,
                    local_files_only=_settings.MODELS_LOCAL_ONLY,
                )
        return cls._reranker

    # aliases used by routers
    @classmethod
    def get_embedder(cls):
        return cls.embedder()

    @classmethod
    def get_reranker(cls):
        return cls.reranker()

    @classmethod
    def warm_up(cls):
        """Force-load all models. Call this on app startup."""
        cls.ner()
        cls.embedder()
        cls.reranker()
        print("All pipeline models loaded.")


@lru_cache()
def get_model_registry() -> "type[ModelRegistry]":
    return ModelRegistry