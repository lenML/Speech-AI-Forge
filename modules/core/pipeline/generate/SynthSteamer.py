import numpy as np

from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.generate.dcls import SynthAudio


class SynthStreamer:

    def __init__(
        self, segments: list[SynthAudio], context: TTSPipelineContext, model: TTSModel
    ) -> None:
        self.segments = segments
        self.context = context
        self.model = model
        self.output_wav = np.empty(0)

    def flush(self):
        """
        刷新合并音频
        """
        output_wav = np.empty(0)

        for seg in self.segments:
            data = seg.data
            if data.size == 0 and not seg.done:
                # 空检查
                break
            output_wav = np.concatenate((output_wav, data), axis=0)
            if not seg.done:
                # 未完成的块，退出，因为只需要合并已经完成的块
                break

        self.output_wav = output_wav
        return output_wav

    def write(self):
        raise NotImplementedError

    def read(self) -> np.ndarray:
        cursor = self.output_wav.size
        self.flush()
        return self.output_wav[cursor:]
