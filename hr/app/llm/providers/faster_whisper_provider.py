from faster_whisper import WhisperModel
from llm.llm_interface import LLMInterface
from helpers import Settings


class WhisperLoader(LLMInterface):
    def __init__(self, settings: Settings):

        self.model_type = settings.WHISPER_MODEL_SIZE
        self.device_type = settings.WHISPER_DEVICE
        self.compute_type = settings.WHISPER_COMPUTE_TYPE
        self.llm = self._load()


    def _load(self):

        return WhisperModel(
            model_size_or_path=self.model_type,
            device=self.device_type,
            compute_type=self.compute_type
        ) 
    
    def get_llm(self):
        return self.llm