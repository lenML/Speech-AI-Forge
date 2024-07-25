from typing import Generator

from modules.core.handler.AudioHandler import AudioHandler
from modules.core.handler.datacls.audio_model import AdjustConfig, EncoderConfig
from modules.core.handler.datacls.tts_model import InferConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.pipeline.factory import PipelineFactory
from modules.core.pipeline.processor import NP_AUDIO, TTSPipelineContext


class SSMLHandler(AudioHandler):
    def __init__(
        self,
        ssml_content: str,
        infer_config: InferConfig,
        adjust_config: AdjustConfig,
        enhancer_config: EnhancerConfig,
        encoder_config: EncoderConfig,
    ) -> None:
        assert isinstance(ssml_content, str), "ssml_content must be a string."
        assert isinstance(
            infer_config, InferConfig
        ), "infer_config must be an InferConfig object."
        assert isinstance(
            adjust_config, AdjustConfig
        ), "adjest_config should be AdjustConfig"
        assert isinstance(
            enhancer_config, EnhancerConfig
        ), "enhancer_config must be an EnhancerConfig object."

        self.ssml_content = ssml_content
        self.infer_config = infer_config
        self.adjest_config = adjust_config
        self.enhancer_config = enhancer_config

        super().__init__(encoder_config=encoder_config, infer_config=infer_config)

        self.validate()

        self.ctx = self.build_ctx()
        self.pipeline = PipelineFactory.create(self.ctx)

    def validate(self):
        # TODO params checker
        pass

    def build_ctx(self):
        ssml_content = self.ssml_content
        infer_config = self.infer_config
        adjust_config = self.adjest_config
        enhancer_config = self.enhancer_config

        ctx = TTSPipelineContext(
            ssml=ssml_content,
            infer_config=infer_config,
            adjust_config=adjust_config,
            enhancer_config=enhancer_config,
        )
        return ctx

    def interrupt(self):
        self.ctx.stop = True
        self.pipeline.model.interrupt()

    def enqueue(self) -> NP_AUDIO:
        return self.pipeline.generate()

    def enqueue_stream(self) -> Generator[NP_AUDIO, None, None]:
        return self.pipeline.generate_stream()
