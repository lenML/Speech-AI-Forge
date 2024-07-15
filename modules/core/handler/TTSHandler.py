import logging
from typing import Generator

from modules.core.handler.AudioHandler import AudioHandler
from modules.core.handler.datacls.audio_model import AdjustConfig, EncoderConfig
from modules.core.handler.datacls.chattts_model import ChatTTSConfig, InferConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.factory import PipelineFactory
from modules.core.pipeline.processor import NP_AUDIO
from modules.core.spk.TTSSpeaker import TTSSpeaker

logger = logging.getLogger(__name__)


class TTSHandler(AudioHandler):
    def __init__(
        self,
        text_content: str,
        spk: TTSSpeaker,
        tts_config: ChatTTSConfig,
        infer_config: InferConfig,
        adjust_config: AdjustConfig,
        enhancer_config: EnhancerConfig,
        encoder_config: EncoderConfig,
    ):
        assert isinstance(text_content, str), "text_content should be str"
        assert isinstance(spk, TTSSpeaker), "spk should be Speaker"
        assert isinstance(
            tts_config, ChatTTSConfig
        ), "tts_config should be ChatTTSConfig"
        assert isinstance(
            infer_config, InferConfig
        ), "infer_config should be InferConfig"
        assert isinstance(
            adjust_config, AdjustConfig
        ), "adjest_config should be AdjustConfig"
        assert isinstance(
            enhancer_config, EnhancerConfig
        ), "enhancer_config should be EnhancerConfig"

        self.text_content = text_content
        self.spk = spk
        self.tts_config = tts_config
        self.infer_config = infer_config
        self.adjest_config = adjust_config
        self.enhancer_config = enhancer_config

        super().__init__(encoder_config=encoder_config, infer_config=infer_config)

        self.validate()

    def validate(self):
        # TODO params checker
        pass

    def create_pipeline(self):
        text_content = self.text_content
        infer_config = self.infer_config
        tts_config = self.tts_config
        adjust_config = self.adjest_config
        enhancer_config = self.enhancer_config
        spk = self.spk

        ctx = TTSPipelineContext(
            text=text_content,
            spk=spk,
            tts_config=tts_config,
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
