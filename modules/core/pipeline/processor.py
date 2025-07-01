from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
from pydub import AudioSegment

from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.generate.dcls import SynthAudio
from modules.utils import audio_utils as audio_utils

# NOTE: 这里有个隐藏设定，默认是 f32 的 dtype，但是其实应该还是有一些地方传递的不是 f32
# FIXME: 最好增强这个 NP_AUDIO 数据类型，支持自动转换 dtype 好一点
NP_AUDIO = Tuple[int, npt.NDArray]
AUDIO = Union[NP_AUDIO, AudioSegment]


class SegmentProcessor:
    """
    用于处理单个 segment
    比如 tn 模块
    比如 vc 模块
    """

    def pre_process(
        self, segment: TTSSegment, context: TTSPipelineContext
    ) -> TTSSegment:
        return segment

    def after_process(self, result: SynthAudio, context: TTSPipelineContext) -> None:
        """
        后处理 result
        处理结果挂在 result 中即可
        """
        return


class AudioProcessor:
    """
    后处理，或者叫做音频处理，比如 响度均衡
    """

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
