from dataclasses import dataclass
from typing import Iterable, List, Optional


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
class SttWord:
    start: float
    end: float
    word: str


@dataclass(repr=False, eq=False)
class SttSegment:
    text: str
    # 单位秒
    start: float
    end: float
    words: Optional[List[SttWord]] = None


@dataclass(repr=False, eq=False)
class SttResult:
    segments: Iterable[SttSegment]
    language: str
    # 这个duration的主要作用是用来做 pregress
    duration: float
