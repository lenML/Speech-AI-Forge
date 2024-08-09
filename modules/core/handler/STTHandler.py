from typing import Generator

from modules.core.handler.datacls.stt_model import STTConfig
from modules.core.models.stt.STTModel import STTModel, TranscribeResult
from modules.core.models.zoo.ModelZoo import model_zoo
from modules.core.pipeline.processor import NP_AUDIO


class STTHandler:

    def __init__(self, input_audio: NP_AUDIO, stt_config: STTConfig) -> None:
        assert isinstance(stt_config, STTConfig), "stt_config must be STTConfig"

        self.input_audio = input_audio
        self.stt_config = stt_config
        self.model: STTModel = self.get_model()

        if self.model is None:
            raise Exception(f"Model {self.stt_config.mid} is not supported")

    def get_model(self):
        model_id = self.stt_config.mid.lower()
        if model_id.startswith("whisper"):
            return model_zoo.get_model(model_id="whisper")

        raise Exception(f"Model {model_id} is not supported")

    def enqueue(self) -> TranscribeResult:
        result = self.model.transcribe(audio=self.input_audio, config=self.stt_config)
        return result

    def enqueue_stream(self) -> Generator[TranscribeResult, None, None]:
        raise NotImplementedError(
            "Method 'enqueue_stream' must be implemented by subclass"
        )
