from langchain_ollama import ChatOllama, OllamaEmbeddings
from llm.llm_interface import LLMInterface
from helpers import Settings

class OllamaProvider(LLMInterface):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.LLM_OLLAMA_MODEL
        self.max_tokens = settings.LLM_MAX_NEW_TOKENS
        self.temperature = settings.LLM_TEMPERATURE
        self.llm = self._load()

    def _load(self):
        return ChatOllama(
            model=self.model_name,
            base_url=self.settings.OLLAMA_BASE_URL,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            num_ctx=8192,  # was unset -> fell back to server default (4096, per logs),
                           # which combined with unbounded history caused context
                           # overflow/eviction after a few questions
            timeout=self.settings.LLM_TIMEOUT,
        )

    def get_llm(self):
        return self.llm


class OllamaEmbeddingsProvider:
    """
    Loads a dedicated Ollama *embedding* model (not the chat model above).
    Used exclusively by the question-uniqueness guard
    (llm/question_similarity.py) to measure semantic similarity between
    generated interview questions — a job text-based instructions and
    exact-string dedup can't reliably do.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.EMBEDDING_MODEL
        self.embeddings = self._load()

    def _load(self):
        return OllamaEmbeddings(
            model=self.model_name,
            base_url=self.settings.OLLAMA_BASE_URL,
        )

    def get_embeddings(self):
        return self.embeddings