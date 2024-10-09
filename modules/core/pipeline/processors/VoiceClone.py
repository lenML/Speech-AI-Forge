from modules.core.handler.datacls.audio_model import AudioFormat, EncoderConfig
from modules.core.handler.VCHandler import VCHandler
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.processor import NP_AUDIO, AudioProcessor


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
