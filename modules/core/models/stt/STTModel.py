from dataclasses import dataclass
from typing import Any, Dict, Optional

from modules import config as global_config
from modules.core.handler.datacls.stt_model import STTConfig, STTOutputFormat
from modules.core.models.BaseZooModel import BaseZooModel
from modules.core.models.stt.whisper.SegmentNormalizer import SubtitleSegment
from modules.core.models.stt.whisper.whisper_dcls import SttResult
from modules.core.models.stt.whisper.writer import get_writer
from modules.core.pipeline.processor import NP_AUDIO
from modules.devices import devices
from modules.utils.monkey_tqdm import disable_tqdm


@dataclass(frozen=True, repr=False, eq=False)
class TranscribeResult:
    text: str
    segments: list[SubtitleSegment]
    language: str


class STTModel(BaseZooModel):

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id=model_id)

    def convert_result_with_format(self, config: STTConfig, result: SttResult) -> str:
        writer_options = {
            "highlight_words": config.highlight_words,
            "max_line_count": config.max_line_count,
            "max_line_width": config.max_line_width,
            "max_words_per_line": config.max_words_per_line,
        }

        format = config.format

        writer = get_writer(format.value)
        with disable_tqdm(enabled=global_config.runtime_env_vars.off_tqdm):
            output = writer.write(result=result, options=writer_options)

        return TranscribeResult(
            text=output,
            segments=writer.subtitles,
            language=result.language,
        )

    def transcribe(self, audio: NP_AUDIO, config: STTConfig) -> str:
        result = self.transcribe_to_result(audio=audio, config=config)
        result_formated = self.convert_result_with_format(config=config, result=result)
        return result_formated

    def transcribe_to_result(self, audio: NP_AUDIO, config: STTConfig) -> SttResult:
        raise NotImplementedError()
