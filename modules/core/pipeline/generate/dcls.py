import numpy as np
import numpy.typing as npt

from modules.core.pipeline.dcls import TTSSegment


class SynthAudio:
    """
    NOTE: 这个对象是在不同线程中传递的
    TODO: 不清楚有没有必要加 lock ，目前好像可以不加
    """

    def __init__(self, segment: TTSSegment) -> None:
        self.seg = segment
        self.data: npt.NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self.sr = 24000
        self.done = False


class TTSBucket:
    def __init__(self, segments: list[SynthAudio]) -> None:
        self.segments = segments


class TTSBatch:
    def __init__(self, segments: list[SynthAudio]) -> None:
        self.segments = segments

    def is_done(self):
        return all([result.done for result in self.segments])
