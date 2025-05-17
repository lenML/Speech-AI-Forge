from dataclasses import dataclass
from typing import Iterable

from faster_whisper.transcribe import Segment, TranscriptionInfo


@dataclass(repr=False, eq=False)
class WhisperSegment:
    seek: int
    start: float
    end: float
    text: str
    tokens: list
    temperature: float
    avg_logprob: float
    compression_ratio: float
    noise_level: float


@dataclass(repr=False, eq=False)
class WhisperTranscribeResult:
    segments: Iterable[Segment]
    language: str
    # 这个duration的主要作用是用来做 pregress
    duration: float
