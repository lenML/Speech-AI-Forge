from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.processor import NP_AUDIO, AudioProcessor
from modules.utils import audio_utils


class AdjusterProcessor(AudioProcessor):
    def _process_array(self, audio: NP_AUDIO, context: TTSPipelineContext) -> NP_AUDIO:
        sample_rate, audio_data = audio
        adjust_config = context.adjust_config

        segment_duration = audio_data.shape[0] / sample_rate
        speed_rate = adjust_config.speed_rate
        duration_s = adjust_config.duration_s

        if duration_s is not None:
            speed_rate = duration_s / segment_duration

        audio_data = audio_utils.apply_prosody_to_audio_data(
            audio_data=audio_data,
            rate=speed_rate,
            pitch=adjust_config.pitch,
            volume=adjust_config.volume_gain_db,
            sr=sample_rate,
        )
        return sample_rate, audio_data
