from langchain_ollama import ChatOllama
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
            timeout=self.settings.LLM_TIMEOUT,
        )

    def get_llm(self):
        return self.llm