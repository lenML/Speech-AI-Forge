from typing import Generator
from modules.core.handler.datacls.stt_model import STTConfig
from modules.core.pipeline.processor import NP_AUDIO


class STTHandler:

    def __init__(self, input_audio: NP_AUDIO, stt_config: STTConfig) -> None:
        assert isinstance(stt_config, STTConfig), "stt_config must be STTConfig"

        self.input_audio = input_audio
        self.stt_config = stt_config

    def enqueue(self) -> str:
        raise NotImplementedError("Method 'enqueue' must be implemented by subclass")

    def enqueue_stream(self) -> Generator[str, None, None]:
        raise NotImplementedError(
            "Method 'enqueue_stream' must be implemented by subclass"
        )
