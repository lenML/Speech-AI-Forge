from typing import Generator, Optional

import numpy as np

from modules.core.pipeline.dcls import TTSSegment
from modules.core.pipeline.processor import TTSPipelineContext
from modules.core.tn.TNPipeline import TNPipeline


class TTSModel:

    def __init__(self, name: str) -> None:
        self.name = name
        self.hash = ""
        self.tn: Optional[TNPipeline] = None

    def load(self, context: TTSPipelineContext) -> None:
        pass

    def unload(self, context: TTSPipelineContext) -> None:
        pass

    def generate(
        self, segment: TTSSegment, context: TTSPipelineContext
    ) -> tuple[int, np.ndarray]:
        return self.generate_batch([segment], context=context)[0]

    def generate_batch(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> list[tuple[int, np.ndarray]]:
        raise NotImplementedError("generate_batch method is not implemented")

    def generate_stream(
        self, segment: TTSSegment, context: TTSPipelineContext
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        for batch in self.generate_batch_stream([segment], context=context):
            yield batch[0]

    def generate_batch_stream(
        self, segments: list[TTSSegment], context: TTSPipelineContext
    ) -> Generator[list[tuple[int, np.ndarray]], None, None]:
        raise NotImplementedError("generate_batch_stream method is not implemented")
