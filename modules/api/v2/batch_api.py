"""
batch api ***施工中***

输入 segments => list[audio]
segment指的是语音片段，并且应该支持多说话人同时推理
"""

from typing import Optional

from pydantic import BaseModel, Field

from modules.api.v2.dcls import SpeakerConfig
from modules.core.handler.datacls.audio_model import AdjustConfig, EncoderConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tn_model import TNConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig


class Segment(BaseModel):
    text: str

    prompt: Optional[str] = None
    spk: Optional[SpeakerConfig] = None

    adjsct: Optional[AdjustConfig] = None
    enhance: Optional[EnhancerConfig] = None


class BatchParams(BaseModel):
    segments: list[Segment]

    encoder: Optional[EncoderConfig] = None
    infer: Optional[InferConfig] = None
    tn: Optional[TNConfig] = None
    tts: TTSConfig = Field(default_factory=TTSConfig)
