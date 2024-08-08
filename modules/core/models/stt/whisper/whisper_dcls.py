from dataclasses import dataclass
from typing import Iterable

from faster_whisper.transcribe import Segment, TranscriptionInfo


@dataclass(repr=False, eq=False, frozen=True)
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


@dataclass(repr=False, eq=False, frozen=True)
class WhisperTranscribeResult:
    segments: Iterable[Segment]
    info: TranscriptionInfo
