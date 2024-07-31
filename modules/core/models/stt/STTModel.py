from dataclasses import dataclass
from typing import Any, Dict, Optional
from modules.core.handler.datacls.stt_model import STTConfig, STTOutputFormat
from modules.core.pipeline.processor import NP_AUDIO


@dataclass(frozen=True, repr=False, eq=False)
class TranscribeResult:
    text: str
    segments: list

    # decoding output
    audio_features: Any = None
    language: str = ""
    language_probs: Optional[Dict[str, float]] = None


class STTModel:

    def __init__(self) -> None:
        pass

    def download(self):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    def transcribe(self, audio: NP_AUDIO, config: STTConfig) -> TranscribeResult:
        pass
