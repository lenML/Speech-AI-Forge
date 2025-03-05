import asyncio
import threading
from time import time
from typing import Union

from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.generate.BatchGenerate import BatchGenerate
from modules.core.pipeline.generate.Bucketizer import Bucketizer
from modules.core.pipeline.generate.dcls import SynthAudio
from modules.core.pipeline.generate.SynthSteamer import SynthStreamer


class BatchSynth:
    def __init__(
        self,
        input_segments: list[TTSSegment],
        context: TTSPipelineContext,
        model: TTSModel,
    ) -> None:
        self.segments = [SynthAudio(segment=segment) for segment in input_segments]
        self.streamer = SynthStreamer(
            segments=self.segments, context=context, model=model
        )
        self.bucketizer = Bucketizer(segments=self.segments)
        self.buckets = self.bucketizer.build_buckets()

        self.generator = BatchGenerate(
            buckets=self.buckets, context=context, model=model
        )
        self.context = context

        self.thread1 = None

    async def wait_done(
        self, timeout: int, query_stop_fn: Union[None, callable] = None
    ):
        start_time = time()
        while not self.generator.done.is_set():
            if query_stop_fn is not None and query_stop_fn():
                raise ConnectionAbortedError()
            if time() - start_time > timeout:
                raise asyncio.TimeoutError()
            await asyncio.sleep(0.5)

    def is_done(self):
        return self.generator.is_done()

    def sr(self):
        return self.segments[0].sr

    def read(self):
        return self.streamer.read()

    def start_generate(self):
        sync_gen = self.context.infer_config.sync_gen
        if sync_gen:
            self.start_generate_sync()
        else:
            self.start_generate_async()

    def start_generate_async(self):
        if self.thread1 is not None:
            return
        gen_t1 = threading.Thread(target=self.generator.generate, args=(), daemon=True)
        gen_t1.start()
        self.thread1 = gen_t1
        return gen_t1

    def start_generate_sync(self):
        self.generator.generate()
