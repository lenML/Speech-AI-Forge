from time import sleep
from typing import Generator, Literal, Union

from pydub import AudioSegment

from modules.core.models.TTSModel import TTSModel
from modules.core.pipeline.dcls import TTSSegment
from modules.core.pipeline.generate.BatchSynth import BatchSynth
from modules.core.pipeline.generate.Chunker import TTSChunker
from modules.core.pipeline.processor import (
    AUDIO,
    NP_AUDIO,
    AudioProcessor,
    TextProcessor,
    TTSPipelineContext,
)
from modules.utils import audio_utils


class TTSPipeline:
    def __init__(self, context: TTSPipelineContext):
        self.modules: list[Union[AudioProcessor, TextProcessor]] = []
        self.model: TTSModel = None
        self.context = context

    def add_module(self, module):
        self.modules.append(module)

    def set_model(self, model):
        self.model = model

    def create_synth(self):
        chunker = TTSChunker(context=self.context)
        segments = chunker.segments()
        synth = BatchSynth(
            input_segments=segments, context=self.context, model=self.model
        )
        return synth

    def generate(self) -> NP_AUDIO:
        synth = self.create_synth()
        synth.start_generate()
        synth.wait_done()
        return synth.sr(), synth.read()

    def generate_stream(self) -> Generator[NP_AUDIO, None, None]:
        synth = self.create_synth()
        synth.start_generate()
        while not synth.is_done():
            data = synth.read()
            if data.size > 0:
                yield synth.sr(), data
            # TODO: replace with threading.Event
            sleep(0.1)
        data = synth.read()
        if data.size > 0:
            yield synth.sr(), data

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

    def process_text(self, text: TTSSegment):
        for module in self.modules:
            if isinstance(module, TextProcessor):
                text = module.process(text)
        return text

    def process_audio(self, audio: AUDIO):
        for module in self.modules:
            if isinstance(module, AudioProcessor):
                audio = module.process(audio)
        return audio
