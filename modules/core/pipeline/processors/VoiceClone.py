from modules.core.handler.datacls.audio_model import AudioFormat, EncoderConfig
from modules.core.handler.VCHandler import VCHandler
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.generate.dcls import SynthAudio
from modules.core.pipeline.processor import NP_AUDIO, AudioProcessor, SegmentProcessor


class VoiceCloneProcessor(AudioProcessor):
    def _process_array(self, audio: NP_AUDIO, context: TTSPipelineContext) -> NP_AUDIO:
        vc_config = context.vc_config
        if not vc_config.enabled:
            return audio
        if vc_config.spk is None and vc_config.spk is not None:
            vc_config.spk = context.spk
        if vc_config.spk is None:
            raise ValueError("spk must be specified")

        handler = VCHandler(
            input_audio=audio,
            vc_config=vc_config,
            # raw encoder
            encoder_config=EncoderConfig(format=AudioFormat.raw),
        )

        return handler.enqueue()


class VoiceCloneSegmentProcessor(SegmentProcessor):
    """
    对单个 segment 结果进行 VoiceClone

        TODO: 得设计一个参数来传递需要使用哪个 spk 做 voice cloning
            还需要决定使用哪个模型，比如 openvoice 和 cosyvoice
    """

    def after_process(self, result: SynthAudio, context: TTSPipelineContext) -> None:
        pass
