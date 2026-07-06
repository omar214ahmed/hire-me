import os
import tempfile
from llm.providers.faster_whisper_provider import WhisperLoader


class Transcript:
    def __init__(self, whisper_loader: WhisperLoader):
        self.whisper_loader = whisper_loader

    def transcribe(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        model = self.whisper_loader.get_llm()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            segments, _ = model.transcribe(tmp_path)
            return " ".join(segment.text for segment in segments)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
