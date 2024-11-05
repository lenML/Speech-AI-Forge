from time import sleep
from typing import Generator, Literal, Union

from pydub import AudioSegment

from modules.core.models.AudioReshaper import AudioReshaper
from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSSegment
from modules.core.pipeline.generate.BatchSynth import BatchSynth
from modules.core.pipeline.generate.Chunker import TTSChunker
from modules.core.pipeline.processor import (
    AUDIO,
    NP_AUDIO,
    AudioProcessor,
    SegmentProcessor,
    TTSPipelineContext,
)
from modules.utils import audio_utils

"""
TODO: 简化 pipeline，不要在这里做类型转换，并且增加 segment class 实现
"""
class AudioPipeline:

    def __init__(self, context: TTSPipelineContext) -> None:
        self.context = context

        self.audio_sr = 44100

    def add_module(self, module: Union[AudioProcessor, SegmentProcessor]):
        if module not in self.context.modules:
            self.context.modules.append(module)

    def generate(self) -> NP_AUDIO:
        pass

    def process_np_audio(self, audio: NP_AUDIO) -> NP_AUDIO:
        audio = self.process_audio(audio)
        audio = self.ensure_audio_type(audio, "ndarray")
        audio = AudioReshaper.normalize_audio(audio=audio, target_sr=self.audio_sr)
        return audio

    def ensure_audio_type(
        self, audio: AUDIO, output_type: Literal["ndarray", "segment"]
    ):
        if output_type == "segment":
            audio = self._to_audio_segment(audio)
        elif output_type == "ndarray":
            audio = self._to_ndarray(audio)
        else:
            raise ValueError(f"Invalid output_type: {output_type}")
        return audio

    def _to_audio_segment(self, audio: AUDIO) -> AudioSegment:
        if isinstance(audio, tuple):
            sr, data = audio
            audio = audio_utils.ndarray_to_segment(ndarray=data, frame_rate=sr)
        return audio

    def _to_ndarray(self, audio: AUDIO) -> NP_AUDIO:
        if isinstance(audio, AudioSegment):
            sr = audio.frame_rate
            audio = audio_utils.audiosegment_to_librosawav(audio)
            return sr, audio
        return audio

    def process_pre(self, seg: TTSSegment):
        for module in self.context.modules:
            if isinstance(module, SegmentProcessor):
                seg = module.pre_process(segment=seg, context=self.context)
        return seg

    def process_audio(self, audio: AUDIO):
        for module in self.context.modules:
            if isinstance(module, AudioProcessor):
                audio = module.process(audio=audio, context=self.context)
        return audio


class TTSPipeline(AudioPipeline):

    def __init__(self, context: TTSPipelineContext):
        super().__init__(context=context)
        self.model: TTSModel = None

    def set_model(self, model):
        self.model = model

    def create_synth(self):
        chunker = TTSChunker(context=self.context)
        segments = chunker.segments()
        # 其实这个在 chunker 之前调用好点...但是有副作用所以放在后面
        segments = [self.process_pre(seg) for seg in segments]

        synth = BatchSynth(
            input_segments=segments, context=self.context, model=self.model
        )
        return synth

    def generate(self) -> NP_AUDIO:
        synth = self.create_synth()
        synth.start_generate()
        synth.wait_done()
        audio = synth.sr(), synth.read()
        return self.process_np_audio(audio)

    def generate_stream(self) -> Generator[NP_AUDIO, None, None]:
        synth = self.create_synth()
        synth.start_generate()
        while not synth.is_done():
            data = synth.read()
            if data.size > 0:
                audio = synth.sr(), data
                yield self.process_np_audio(audio)
            # TODO: replace with threading.Event
            sleep(0.1)
        data = synth.read()
        if data.size > 0:
            audio = synth.sr(), data
            yield self.process_np_audio(audio)
