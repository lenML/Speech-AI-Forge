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

    # 移除静音 （只支持非流式）
    remove_silence: bool = False
    remove_silence_threshold: float = -42

class EncoderConfig(BaseModel):
    # NOTE: 默认格式设置为 raw ，即不需要编码的格式
    # NOTE: raw 格式为 pcm 数据流，应该以 wav 格式解码
    format: AudioFormat = AudioFormat.raw
    # 32k / 64k / 96k / 128k / 192k / 256k / 320k
    bitrate: str = "64k"
    acodec: Optional[str] = None
