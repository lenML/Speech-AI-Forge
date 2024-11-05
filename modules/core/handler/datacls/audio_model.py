from enum import Enum
from typing import Optional

from pydantic import BaseModel


class AudioFormat(str, Enum):
    mp3 = "mp3"
    wav = "wav"
    ogg = "ogg"
    acc = "acc"
    flac = "flac"
    raw = "raw"


class AdjustConfig(BaseModel):
    pitch: float = 0
    speed_rate: float = 1
    volume_gain_db: float = 0

    # 响度均衡
    normalize: bool = True
    headroom: float = 1


class EncoderConfig(BaseModel):
    # 32k / 64k / 96k / 128k / 192k / 256k / 320k
    bitrate: str = "64k"
    format: AudioFormat = AudioFormat.mp3
    acodec: Optional[str] = None
