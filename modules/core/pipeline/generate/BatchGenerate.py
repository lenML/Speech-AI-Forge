import logging
import threading

import numpy as np

from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.generate.dcls import TTSBatch, TTSBucket
from modules.core.pipeline.processor import SegmentProcessor
from modules.utils import audio_utils

logger = logging.getLogger(__name__)


class BatchGenerate:

    def __init__(
        self,
        buckets: list[TTSBucket],
        context: TTSPipelineContext,
        model: TTSModel,
    ) -> None:
        self.buckets = buckets
        self.model = model
        self.context = context
        self.batches = self.build_batches()

        self.done = threading.Event()

    def is_done(self):
        return all([seg.done for batch in self.batches for seg in batch.segments])

    def build_batches(self) -> list[TTSBatch]:
        batch_size = self.context.infer_config.batch_size

        batches = []
        for bucket in self.buckets:
            for i in range(0, len(bucket.segments), batch_size):
                batch = bucket.segments[i : i + batch_size]
                batches.append(TTSBatch(segments=batch))
        return batches

    def generate(self):
        self.model.reset()
        stream = self.context.infer_config.stream

        try:
            for batch in self.batches:
                is_break = batch.segments[0].seg._type == "break"
                if is_break:
                    self.generate_break(batch)
                    continue

                if stream:
                    self.generate_batch_stream(batch)
                else:
                    self.generate_batch(batch)
        # except Exception as e:
        #     logger.error(f"error: {e}")
        finally:
            self.done.set()

    def generate_break(self, batch: TTSBatch):
        for seg in batch.segments:
            sr, data = audio_utils.silence_np(
                duration_s=seg.seg.duration_ms / 1000,
                sample_rate=self.model.get_sample_rate(),
            )
            seg.data = data
            seg.sr = sr
            seg.done = True

    def generate_batch(self, batch: TTSBatch):
        model = self.model
        segments = [audio.seg for audio in batch.segments]
        results = model.generate_batch(segments=segments, context=self.context)
        for audio, result in zip(batch.segments, results):
            sr, data = result
            audio.data = data
            audio.sr = sr
            audio.done = True

            self.after_process(result=audio)

    def generate_batch_stream(self, batch: TTSBatch):
        model = self.model
        segments = [audio.seg for audio in batch.segments]

        # NOTE: stream 不支持 after_process
        for seg in segments:
            if seg.duration_ms is not None:
                logger.warning("Not support duration_ms in stream mode")
                break
            if seg.speed_rate is not None:
                logger.warning("Not support speed_rate in stream mode")
                break

        for results in model.generate_batch_stream(
            segments=segments, context=self.context
        ):
            for audio, result in zip(batch.segments, results):
                sr, data = result
                if data.size == 0:
                    audio.done = True
                    continue
                audio.data = np.concatenate([audio.data, data], axis=0)
                audio.sr = sr

        # NOTE: 这里在最后设置 done 是因为流式生成的时候，目前不知道单个segment是否结束
        for seg in batch.segments:
            seg.done = True

    def after_process(self, result: TTSBatch):
        # NOTE: 按道理说这个应该给 pipeline 来控制，但是不太好决定 segement 的处理时机，所以放在这里
        # TODO: 最好还是不要把 module 传到 generator 中用
        for module in self.context.modules:
            if isinstance(module, SegmentProcessor):
                module.after_process(result=result, context=self.context)
