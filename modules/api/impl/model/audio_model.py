from enum import Enum

from pydantic import BaseModel


class AudioFormat(str, Enum):
    mp3 = "mp3"
    wav = "wav"


class AdjustConfig(BaseModel):
    pitch: float = 0
    speed_rate: float = 1
    volume_gain_db: float = 0
