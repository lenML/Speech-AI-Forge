from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.processor import NP_AUDIO, AudioProcessor
from modules.utils import audio_utils


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
