import numpy as np

from modules.core.pipeline.dcls import TTSSegment


class SynthAudio:
    def __init__(self, segment: TTSSegment) -> None:
        self.seg = segment
        self.data = np.empty(0)
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
