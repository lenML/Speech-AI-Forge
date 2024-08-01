from typing import Generator

from modules.core.handler.datacls.stt_model import STTConfig
from modules.core.models.stt.Whisper import WhisperModel
from modules.core.pipeline.processor import NP_AUDIO


class STTHandler:

    def __init__(self, input_audio: NP_AUDIO, stt_config: STTConfig) -> None:
        assert isinstance(stt_config, STTConfig), "stt_config must be STTConfig"

        self.input_audio = input_audio
        self.stt_config = stt_config
        self.model = self.get_model()

    def get_model(self):
        model_id = self.stt_config.mid.lower()
        if model_id.startswith("whisper"):
            return WhisperModel(model_id=model_id)

        raise Exception(f"Model {model_id} is not supported")

    def enqueue(self) -> str:
        result = self.model.transcribe(audio=self.input_audio, config=self.stt_config)
        return result.text

    def enqueue_stream(self) -> Generator[str, None, None]:
        raise NotImplementedError(
            "Method 'enqueue_stream' must be implemented by subclass"
        )
