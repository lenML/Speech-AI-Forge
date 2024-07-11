from numpy import ndarray
from modules.Enhancer.ResembleEnhance import apply_audio_enhance_full
from modules.core.models.tts.ChatTtsModel import ChatTTSModel
from modules.core.pipeline.dcls import TTSSegment
from modules.core.pipeline.pipeline import TTSPipeline
from modules.core.pipeline.processor import (
    NP_AUDIO,
    AudioProcessor,
    TTSPipelineContext,
    TextProcessor,
)
from modules.core.ssml.SSMLParser import SSMLSegment
from modules.core.tn.ChatTtsTN import ChatTtsTN
from modules.utils import audio_utils


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

        audio_data, sample_rate = apply_audio_enhance_full(
            audio_data=audio_data,
            sr=sample_rate,
            nfe=nfe,
            solver=solver,
            lambd=lambd,
            tau=tau,
        )

        return audio_data, sample_rate


class AdjusterProcessor(AudioProcessor):
    def _process_array(self, audio: NP_AUDIO, context: TTSPipelineContext) -> NP_AUDIO:
        sr, audio_data = audio
        adjust_config = context.adjust_config

        audio_data = audio_utils.apply_prosody_to_audio_data(
            audio_data=audio_data,
            rate=adjust_config.speed_rate,
            pitch=adjust_config.pitch,
            volume=adjust_config.volume_gain_db,
            sr=sr,
        )
        return sr, audio_data


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


class ChatTtsTNProcessor(TextProcessor):
    def process(self, segment: TTSSegment, context: TTSPipelineContext) -> TTSSegment:
        segment.text = ChatTtsTN.normalize(segment.text, context.tts_config)
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
        pipeline.set_model(ChatTTSModel())
        return pipeline
