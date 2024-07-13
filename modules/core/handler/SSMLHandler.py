from typing import Generator


from modules.core.handler.AudioHandler import AudioHandler
from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.chattts_model import ChatTTSConfig, InferConfig
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

        self.validate()

    def validate(self):
        # TODO params checker
        pass

    def create_pipeline(self):
        ssml_content = self.ssml_content
        infer_config = self.infer_config
        adjust_config = self.adjest_config
        enhancer_config = self.enhancer_config

        ctx = TTSPipelineContext(
            ssml=ssml_content,
            tts_config=ChatTTSConfig(),
            infer_config=infer_config,
            adjust_config=adjust_config,
            enhancer_config=enhancer_config,
        )
        pipeline = PipelineFactory.create(ctx)
        return pipeline

    def enqueue(self) -> NP_AUDIO:
        pipeline = self.create_pipeline()
        return pipeline.generate()

    def enqueue_stream(self) -> Generator[NP_AUDIO, None, None]:
        pipeline = self.create_pipeline()
        return pipeline.generate_stream()
