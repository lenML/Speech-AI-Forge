from numpy import ndarray
from modules.Enhancer.ResembleEnhance import apply_audio_enhance_full
from modules.core.models.tts.ChatTtsModel import ChatTTSModel
from modules.core.pipeline.dcls import TTSSegment
from modules.core.pipeline.pipeline import TTSPipeline
from modules.core.pipeline.processor import (
    NP_AUDIO,
    AudioProcessor,
    TTSPipelineContext,
    PreProcessor,
)
from modules.core.tn.ChatTtsTN import ChatTtsTN
from modules.utils import audio_utils
from modules.data import styles_mgr
from modules.core.speaker import Speaker, speaker_mgr

import logging

logger = logging.getLogger(__name__)


class EnhancerProcessor(AudioProcessor):
    def _process_array(
        self, audio: tuple[int, ndarray], context: TTSPipelineContext
    ) -> tuple[int, ndarray]:
        enhancer_config = context.enhancer_config

        if not enhancer_config.enabled:
            return audio
        nfe = enhancer_config.nfe
        solver = enhancer_config.solver
        lambd = enhancer_config.lambd
        tau = enhancer_config.tau

        sample_rate, audio_data = audio
        audio_data, sample_rate = apply_audio_enhance_full(
            audio_data=audio_data,
            sr=sample_rate,
            nfe=nfe,
            solver=solver,
            lambd=lambd,
            tau=tau,
        )

        return sample_rate, audio_data


class AdjusterProcessor(AudioProcessor):
    def _process_array(self, audio: NP_AUDIO, context: TTSPipelineContext) -> NP_AUDIO:
        sample_rate, audio_data = audio
        adjust_config = context.adjust_config

        audio_data = audio_utils.apply_prosody_to_audio_data(
            audio_data=audio_data,
            rate=adjust_config.speed_rate,
            pitch=adjust_config.pitch,
            volume=adjust_config.volume_gain_db,
            sr=sample_rate,
        )
        return sample_rate, audio_data


class AudioNormalizer(AudioProcessor):
    def _process_array(self, audio: NP_AUDIO, context: TTSPipelineContext) -> NP_AUDIO:
        adjust_config = context.adjust_config
        if not adjust_config.normalize:
            return audio
        sample_rate, audio_data = audio
        sample_rate, audio_data = audio_utils.apply_normalize(
            audio_data=audio_data, headroom=adjust_config.headroom, sr=sample_rate
        )
        return sample_rate, audio_data


class ChatTtsTNProcessor(PreProcessor):
    def process(self, segment: TTSSegment, context: TTSPipelineContext) -> TTSSegment:
        segment.text = ChatTtsTN.normalize(text=segment.text, config=context.tn_config)
        return segment


class ChatTtsStyleProcessor(PreProcessor):
    """
    计算合并 style/spk
    """

    def get_style_params(self, context: TTSPipelineContext):
        style = context.tts_config.style
        if not style:
            return {}
        params = styles_mgr.find_params_by_name(style)
        return params

    def process(self, segment: TTSSegment, context: TTSPipelineContext) -> TTSSegment:
        params = self.get_style_params(context)
        segment.prompt = (
            segment.prompt or context.tts_config.prompt or params.get("prompt", "")
        )
        segment.prompt1 = (
            segment.prompt1 or context.tts_config.prompt1 or params.get("prompt1", "")
        )
        segment.prompt2 = (
            segment.prompt2 or context.tts_config.prompt2 or params.get("prompt2", "")
        )
        segment.prefix = (
            segment.prefix or context.tts_config.prefix or params.get("prefix", "")
        )

        spk = segment.spk or context.spk

        if isinstance(spk, str):
            if spk == "":
                spk = None
            else:
                spk = speaker_mgr.get_speaker(spk)
        if spk and not isinstance(spk, Speaker):
            spk = None
            logger.warn(f"Invalid spk: {spk}")

        segment.spk = spk

        return segment


class PipelineFactory:
    @classmethod
    def create(cls, ctx: TTSPipelineContext) -> TTSPipeline:
        model_id = ctx.tts_config.mid

        if model_id == "chat-tts":
            return cls.create_chattts_pipeline(ctx)
        else:
            raise Exception(f"Unknown model id: {model_id}")

    @classmethod
    def create_chattts_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = TTSPipeline(ctx)
        pipeline.add_module(ChatTtsTNProcessor())
        pipeline.add_module(EnhancerProcessor())
        pipeline.add_module(AdjusterProcessor())
        pipeline.add_module(AudioNormalizer())
        pipeline.add_module(ChatTtsStyleProcessor())
        pipeline.set_model(ChatTTSModel())
        return pipeline
