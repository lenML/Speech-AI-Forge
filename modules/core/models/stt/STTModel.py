from dataclasses import dataclass
from typing import Any, Dict, Optional

from modules.core.handler.datacls.stt_model import STTConfig, STTOutputFormat
from modules.core.models.BaseZooModel import BaseZooModel
from modules.core.pipeline.processor import NP_AUDIO


@dataclass(frozen=True, repr=False, eq=False)
class TranscribeResult:
    text: str
    segments: list

    info: dict


class STTModel(BaseZooModel):

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id=model_id)

    def transcribe(self, audio: NP_AUDIO, config: STTConfig) -> TranscribeResult:
        raise NotImplementedError()
