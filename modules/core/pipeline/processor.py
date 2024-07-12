from typing import Union

import numpy as np
from pydub import AudioSegment

from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.utils import audio_utils as audio_utils

NP_AUDIO = tuple[int, np.ndarray]
AUDIO = Union[NP_AUDIO, AudioSegment]


class TextProcessor:
    def process(self, segment: TTSSegment, context: TTSPipelineContext) -> TTSSegment:
        raise NotImplementedError


class AudioProcessor:
    def process(self, audio: AUDIO, context: TTSPipelineContext) -> AUDIO:
        if isinstance(audio, tuple):
            return self._process_array(audio, context)
        elif isinstance(audio, AudioSegment):
            return self._process_segment(audio, context)
        else:
            raise ValueError("Unsupported audio type")

    def _process_array(self, audio: NP_AUDIO, context: TTSPipelineContext) -> NP_AUDIO:
        sr, data = audio
        segment = audio_utils.ndarray_to_segment(ndarray=data, frame_rate=sr)
        processed_segment = self._process_segment(segment, context)
        return audio_utils.audiosegment_to_librosawav(processed_segment)

    def _process_segment(
        self, audio: AudioSegment, context: TTSPipelineContext
    ) -> AudioSegment:
        ndarray = audio_utils.audiosegment_to_librosawav(audio)
        processed_ndarray = self._process_array(ndarray, context)
        return audio_utils.ndarray_to_segment(processed_ndarray)
