import logging
from typing import AsyncGenerator, Generator, Optional

from modules.core.handler.AudioHandler import AudioHandler
from modules.core.handler.datacls.audio_model import AdjustConfig, EncoderConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tn_model import TNConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.factory import PipelineFactory
from modules.core.pipeline.processor import NP_AUDIO
from modules.core.spk.TTSSpeaker import TTSSpeaker

logger = logging.getLogger(__name__)


class TTSHandler(AudioHandler):

    def __init__(
        self,
        *,
        text_content: Optional[str] = None,
        batch_content: Optional[list[str]] = None,
        ssml_content: Optional[str] = None,
        spk: Optional[TTSSpeaker] = None,
        tts_config: TTSConfig = TTSConfig(),
        infer_config: InferConfig = InferConfig(),
        adjust_config: AdjustConfig = AdjustConfig(),
        enhancer_config: EnhancerConfig = EnhancerConfig(),
        encoder_config: EncoderConfig = EncoderConfig(),
        vc_config: VCConfig = VCConfig(),
        tn_config: TNConfig = TNConfig(),
    ):
        assert isinstance(tts_config, TTSConfig), "tts_config should be ChatTTSConfig"
        assert isinstance(
            infer_config, InferConfig
        ), "infer_config should be InferConfig"
        assert isinstance(
            adjust_config, AdjustConfig
        ), "adjest_config should be AdjustConfig"
        assert isinstance(
            enhancer_config, EnhancerConfig
        ), "enhancer_config should be EnhancerConfig"
        assert isinstance(
            encoder_config, EncoderConfig
        ), "encoder_config should be EncoderConfig"
        assert isinstance(vc_config, VCConfig), "vc_config should be VCConfig"
        assert isinstance(tn_config, TNConfig), "tn_config should be TNConfig"

        self.text_content = text_content
        self.batch_content = batch_content
        self.ssml_content = ssml_content
        self.spk = spk
        self.tts_config = tts_config
        self.infer_config = infer_config
        self.adjest_config = adjust_config
        self.enhancer_config = enhancer_config
        self.vc_config = vc_config
        self.tn_config = tn_config

        super().__init__(encoder_config=encoder_config, infer_config=infer_config)

        self.validate()

        self.ctx = self.build_ctx()
        self.pipeline = PipelineFactory.create(self.ctx)

    def validate(self):
        # TODO params checker
        pass

    def build_ctx(self):
        text_content = self.text_content
        batch_content = self.batch_content
        ssml_content = self.ssml_content
        infer_config = self.infer_config
        tts_config = self.tts_config
        adjust_config = self.adjest_config
        enhancer_config = self.enhancer_config
        vc_config = self.vc_config
        tn_config = self.tn_config
        spk = self.spk

        ctx = TTSPipelineContext(
            text=text_content,
            texts=batch_content,
            ssml=ssml_content,
            spk=spk,
            tts_config=tts_config,
            infer_config=infer_config,
            adjust_config=adjust_config,
            enhancer_config=enhancer_config,
            vc_config=vc_config,
            tn_config=tn_config,
        )
        return ctx

    def interrupt(self):
        self.ctx.stop = True
        self.pipeline.model.interrupt()

    def get_sample_rate(self):
        return self.pipeline.model.get_sample_rate()

    async def enqueue(self) -> NP_AUDIO:
        timeout = self.ctx.infer_config.timeout
        return await self.pipeline.generate(timeout=timeout)

    def enqueue_stream(self) -> AsyncGenerator[NP_AUDIO, None]:
        timeout = self.ctx.infer_config.timeout
        return self.pipeline.generate_stream(timeout=timeout)
