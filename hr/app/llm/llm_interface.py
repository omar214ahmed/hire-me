from abc import ABC , abstractmethod


class LLMInterface(ABC):

    @abstractmethod
    def _load(self):
        pass


    @abstractmethod
    def get_llm(self):
        pass