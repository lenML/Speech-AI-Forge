import logging

from numpy import ndarray

from modules.core.models.zoo.ModelZoo import model_zoo
from modules.core.pipeline.dcls import TTSSegment
from modules.core.pipeline.pipeline import TTSPipeline
from modules.core.pipeline.processor import (
    NP_AUDIO,
    AudioProcessor,
    PreProcessor,
    TTSPipelineContext,
)
from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.core.tn.ChatTtsTN import ChatTtsTN
from modules.core.tn.CosyVoiceTN import CosyVoiceTN
from modules.core.tn.FishSpeechTN import FishSpeechTN
from modules.core.tn.TNPipeline import TNPipeline
from modules.data import styles_mgr
from modules.Enhancer.ResembleEnhance import apply_audio_enhance_full
from modules.utils import audio_utils

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


class TNProcess(PreProcessor):

    def __init__(self, tn_pipeline: TNPipeline) -> None:
        super().__init__()
        self.tn = tn_pipeline

    def process(self, segment: TTSSegment, context: TTSPipelineContext) -> TTSSegment:
        segment.text = self.tn.normalize(text=segment.text, config=context.tn_config)
        return segment


class TTSStyleProcessor(PreProcessor):
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
        segment.emotion = (
            segment.emotion or context.tts_config.emotion or params.get("emotion", "")
        )

        spk = segment.spk or context.spk

        if isinstance(spk, str):
            if spk == "":
                spk = None
            else:
                spk = spk_mgr.get_speaker(spk)
        if spk and not isinstance(spk, TTSSpeaker):
            spk = None
            logger.warn(f"Invalid spk: {spk}")

        segment.spk = spk

        return segment


class PipelineFactory:
    @classmethod
    def create(cls, ctx: TTSPipelineContext) -> TTSPipeline:
        model_id = ctx.tts_config.mid

        if model_id == "chattts" or model_id == "chat-tts":
            return cls.create_chattts_pipeline(ctx)
        elif model_id == "fishspeech" or model_id == "fish-speech":
            return cls.create_fishspeech_pipeline(ctx)
        elif model_id == "cosyvoice" or model_id == "cosy-voice":
            return cls.create_cosyvoice_pipeline(ctx)
        else:
            raise Exception(f"Unknown model id: {model_id}")

    @classmethod
    def create_base_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = TTSPipeline(ctx)
        pipeline.add_module(EnhancerProcessor())
        pipeline.add_module(AdjusterProcessor())
        pipeline.add_module(AudioNormalizer())
        pipeline.add_module(TTSStyleProcessor())
        return pipeline

    @classmethod
    def create_chattts_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = cls.create_base_pipeline(ctx=ctx)
        pipeline.add_module(TNProcess(tn_pipeline=ChatTtsTN))
        model = model_zoo.get_model("chat-tts")
        pipeline.set_model(model)
        return pipeline

    @classmethod
    def create_fishspeech_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = cls.create_base_pipeline(ctx=ctx)
        pipeline.add_module(TNProcess(tn_pipeline=FishSpeechTN))
        model = model_zoo.get_model("fish-speech")
        pipeline.set_model(model)
        return pipeline

    @classmethod
    def create_cosyvoice_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = cls.create_base_pipeline(ctx=ctx)
        pipeline.add_module(TNProcess(tn_pipeline=CosyVoiceTN))
        model = model_zoo.get_model("cosy-voice")
        pipeline.set_model(model)
        return pipeline
