from enum import Enum

from pydantic import BaseModel


class AudioFormat(str, Enum):
    mp3 = "mp3"
    wav = "wav"
    ogg = "ogg"
    acc = "acc"
    flac = "flac"


class AdjustConfig(BaseModel):
    pitch: float = 0
    speed_rate: float = 1
    volume_gain_db: float = 0

    # 响度均衡
    normalize: bool = True
    headroom: float = 1
