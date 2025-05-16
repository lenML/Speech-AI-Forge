from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel


class STTOutputFormat(str, Enum):
    txt = "txt"
    vtt = "vtt"
    srt = "srt"
    tsv = "tsv"
    lrc = "lrc"
    json = "json"


class STTConfig(BaseModel):
    # model id
    mid: str = "whisper.large"

    # 参考文本、文稿匹配
    refrence_transcript: Optional[str] = None

    prompt: Optional[Union[str, List[int]]] = None
    prefix: Optional[Union[str, List[int]]] = None

    language: Optional[str] = None
    temperature: Optional[float] = None
    sample_len: Optional[int] = None
    best_of: Optional[int] = None
    beam_size: Optional[int] = None
    patience: Optional[int] = None
    length_penalty: Optional[float] = None

    format: Optional[STTOutputFormat] = STTOutputFormat.txt

    highlight_words: Optional[bool] = False
    max_line_count: Optional[int] = None
    max_line_width: Optional[int] = None
    max_words_per_line: Optional[int] = None
