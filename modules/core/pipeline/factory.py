import logging

from modules.core.models.AudioReshaper import AudioReshaper
from modules.core.models.zoo.ModelZoo import model_zoo
from modules.core.pipeline.dcls import TTSSegment
from modules.core.pipeline.pipeline import AudioPipeline, TTSPipeline
from modules.core.pipeline.processor import (
    NP_AUDIO,
    SegmentProcessor,
    TTSPipelineContext,
)
from modules.core.pipeline.processors.Adjuster import (
    AdjusterProcessor,
    AdjustSegmentProcessor,
)
from modules.core.pipeline.processors.Enhancer import EnhancerProcessor
from modules.core.pipeline.processors.Normalizer import AudioNormalizer
from modules.core.pipeline.processors.VoiceClone import VoiceCloneProcessor
from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.core.tn.base_tn import BaseTN
from modules.core.tn.ChatTtsTN import ChatTtsTN
from modules.core.tn.CosyVoiceTN import CosyVoiceTN
from modules.core.tn.F5TtsTN import F5TtsTN
from modules.core.tn.FireRedTtsTN import FireRedTtsTN
from modules.core.tn.FishSpeechTN import FishSpeechTN
from modules.core.tn.IndexTTSTN import IndexTTSTN
from modules.core.tn.SparkTTSTN import SparkTTSTN
from modules.core.tn.TNPipeline import TNPipeline
from modules.data import styles_mgr

logger = logging.getLogger(__name__)


class TNProcess(SegmentProcessor):

    def __init__(self, tn_pipeline: TNPipeline) -> None:
        super().__init__()
        self.tn = tn_pipeline

    def pre_process(
        self, segment: TTSSegment, context: TTSPipelineContext
    ) -> TTSSegment:
        segment.text = self.tn.normalize(text=segment.text, config=context.tn_config)
        return segment


class TTSStyleProcessor(SegmentProcessor):
    """
    计算合并 style/spk
    """

    def get_style_params(self, context: TTSPipelineContext):
        style = context.tts_config.style
        if not style:
            return {}
        params = styles_mgr.find_params_by_name(style)
        return params

    def pre_process(
        self, segment: TTSSegment, context: TTSPipelineContext
    ) -> TTSSegment:
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


class FromAudioPipeline(AudioPipeline):

    def __init__(self, audio: NP_AUDIO, ctx: TTSPipelineContext) -> None:
        super().__init__(context=ctx)
        self.audio = audio

    def generate(self):
        audio_data = self.audio
        audio_data = AudioReshaper.normalize_audio(
            audio=audio_data, target_sr=self.audio_sr
        )
        # audio_data = AudioReshaper.normalize_audio_type(audio=audio_data)
        audio_data = self.process_np_audio(audio=audio_data)
        return audio_data


class PipelineFactory:
    @classmethod
    def create(cls, ctx: TTSPipelineContext) -> TTSPipeline:
        model_id = ctx.tts_config.mid.lower().replace(" ", "-").replace("-", "")

        if model_id == "chattts":
            return cls.create_chattts_pipeline(ctx)
        elif model_id == "fishspeech":
            return cls.create_fishspeech_pipeline(ctx)
        elif model_id == "cosyvoice":
            return cls.create_cosyvoice_pipeline(ctx)
        elif model_id == "firered" or model_id == "fireredtts":
            return cls.create_fire_red_tts_pipeline(ctx)
        elif model_id == "f5" or model_id == "f5tts":
            return cls.create_f5_tts_pipeline(ctx)
        elif model_id == "indextts":
            return cls.create_index_tts_pipeline(ctx)
        elif model_id == "sparktts":
            return cls.create_spark_tts_pipeline(ctx)
        elif model_id == "gptsovitsv4":
            return cls.create_gpt_sovits_v4(ctx)
        else:
            raise Exception(f"Unknown model id: {model_id}")

    @classmethod
    def setup_base_modules(cls, pipeline: AudioPipeline):
        pipeline.add_module(VoiceCloneProcessor())
        pipeline.add_module(EnhancerProcessor())

        # NOTE: 先 normalizer 后 adjuster，不然 volume_gain_db 和 normalize 冲突
        pipeline.add_module(AudioNormalizer())
        pipeline.add_module(AdjusterProcessor())

        pipeline.add_module(TTSStyleProcessor())

        pipeline.add_module(AdjustSegmentProcessor())
        return pipeline

    @classmethod
    def create_chattts_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = TTSPipeline(ctx)
        cls.setup_base_modules(pipeline=pipeline)
        pipeline.add_module(TNProcess(tn_pipeline=ChatTtsTN))
        model = model_zoo.get_chat_tts()
        pipeline.set_model(model)

        pipeline.audio_sr = model.get_sample_rate()
        return pipeline

    @classmethod
    def create_fishspeech_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = TTSPipeline(ctx)
        cls.setup_base_modules(pipeline=pipeline)
        pipeline.add_module(TNProcess(tn_pipeline=FishSpeechTN))
        model = model_zoo.get_fish_speech()
        pipeline.set_model(model)

        pipeline.audio_sr = model.get_sample_rate()
        return pipeline

    @classmethod
    def create_cosyvoice_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = TTSPipeline(ctx)
        cls.setup_base_modules(pipeline=pipeline)
        pipeline.add_module(TNProcess(tn_pipeline=CosyVoiceTN))
        model = model_zoo.get_cosy_voice()
        pipeline.set_model(model)

        pipeline.audio_sr = model.get_sample_rate()
        return pipeline

    @classmethod
    def create_fire_red_tts_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = TTSPipeline(ctx)
        cls.setup_base_modules(pipeline=pipeline)
        pipeline.add_module(TNProcess(tn_pipeline=FireRedTtsTN))
        model = model_zoo.get_fire_red_tts()
        pipeline.set_model(model)

        pipeline.audio_sr = model.get_sample_rate()
        return pipeline

    @classmethod
    def create_f5_tts_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = TTSPipeline(ctx)
        cls.setup_base_modules(pipeline=pipeline)
        pipeline.add_module(TNProcess(tn_pipeline=F5TtsTN))
        model = model_zoo.get_f5_tts()
        pipeline.set_model(model)

        pipeline.audio_sr = model.get_sample_rate()
        return pipeline

    @classmethod
    def create_index_tts_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = TTSPipeline(ctx)
        cls.setup_base_modules(pipeline=pipeline)
        pipeline.add_module(TNProcess(tn_pipeline=IndexTTSTN))
        model = model_zoo.get_index_tts()
        pipeline.set_model(model)

        pipeline.audio_sr = model.get_sample_rate()
        return pipeline

    @classmethod
    def create_spark_tts_pipeline(cls, ctx: TTSPipelineContext):
        pipeline = TTSPipeline(ctx)
        cls.setup_base_modules(pipeline=pipeline)
        pipeline.add_module(TNProcess(tn_pipeline=SparkTTSTN))
        model = model_zoo.get_spark_tts()
        pipeline.set_model(model)

        pipeline.audio_sr = model.get_sample_rate()
        return pipeline

    @classmethod
    def create_gpt_sovits_v4(cls, ctx: TTSPipelineContext):
        pipeline = TTSPipeline(ctx)
        cls.setup_base_modules(pipeline=pipeline)
        # TODO: 可能需要实现自己的TN，不确定需不需要
        pipeline.add_module(TNProcess(tn_pipeline=BaseTN))
        model = model_zoo.get_gpt_sovits_v4()
        pipeline.set_model(model)

        pipeline.audio_sr = model.get_sample_rate()
        return pipeline

    @classmethod
    def create_postprocess_pipeline(cls, audio: NP_AUDIO, ctx: TTSPipelineContext):
        pipeline = FromAudioPipeline(audio=audio, ctx=ctx)
        cls.setup_base_modules(pipeline=pipeline)
        return pipeline
