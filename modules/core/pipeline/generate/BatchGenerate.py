import logging
import threading

import numpy as np

from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.generate.dcls import TTSBatch, TTSBucket
from modules.utils import audio_utils

logger = logging.getLogger(__name__)


class BatchGenerate:
    def __init__(
        self, buckets: list[TTSBucket], context: TTSPipelineContext, model: TTSModel
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
        for batch in self.batches:
            is_break = batch.segments[0].seg._type == "break"
            if is_break:
                self.generate_break(batch)
                continue

            if stream:
                self.generate_batch_stream(batch)
            else:
                self.generate_batch(batch)

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

            if audio.seg.duration_ms is not None:
                # 表示需要调整时间
                result_duration = data.size / sr * 1000
                audio.data = audio_utils.apply_prosody_to_audio_data(
                    sr=sr,
                    audio_data=audio.data,
                    rate=audio.seg.duration_ms / result_duration,
                )

    def generate_batch_stream(self, batch: TTSBatch):
        model = self.model
        segments = [audio.seg for audio in batch.segments]

        # NOTE: stream 不支持 duration_ms
        for seg in segments:
            if seg.duration_ms is not None:
                logger.warning("Not support duration_ms in stream mode")
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

        for seg in batch.segments:
            seg.done = True
