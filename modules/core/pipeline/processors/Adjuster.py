from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.generate.dcls import SynthAudio
from modules.core.pipeline.processor import NP_AUDIO, AudioProcessor, SegmentProcessor
from modules.utils import audio_utils


class AdjusterProcessor(AudioProcessor):
    """
    对整个合成结果进行 adjust
    """

    def _process_array(self, audio: NP_AUDIO, context: TTSPipelineContext) -> NP_AUDIO:
        sample_rate, audio_data = audio
        adjust_config = context.adjust_config
        speed_rate = adjust_config.speed_rate

        audio_data = audio_utils.apply_prosody_to_audio_data(
            audio_data=audio_data,
            rate=speed_rate,
            pitch=adjust_config.pitch,
            volume=adjust_config.volume_gain_db,
            sr=sample_rate,
        )

        if adjust_config.remove_silence:
            audio_data = audio_utils.remove_silence_edges(
                audio=audio_data,
                silence_threshold=adjust_config.remove_silence_threshold,
            )

        return sample_rate, audio_data


class AdjustSegmentProcessor(SegmentProcessor):
    """
    对单个 segment 结果进行 adjust

    NOTE: 我已经忘记单个和整个adjust的区别在哪了...似乎是vc模型才会调用这里处理？
    TODO: 最好整合一下逻辑，不然有点混乱说实话...
    """

    def after_process(self, result: SynthAudio, context: TTSPipelineContext) -> None:
        seg = result.seg
        audio_data = result.data
        sample_rate = result.sr
        speed_rate = seg.speed_rate
        duration_ms = seg.duration_ms
        segment_duration = audio_data.size / sample_rate

        # 因为目前只支持 speed 调整所以只检查 speed
        no_speed_rate = speed_rate == 1 or speed_rate is None
        no_duration_ms = duration_ms is None
        if no_speed_rate and no_duration_ms:
            return

        if duration_ms is not None:
            duration_s = duration_ms / 1000
            speed_rate = segment_duration / duration_s

        if speed_rate == 1:
            return

        audio_data = audio_utils.apply_prosody_to_audio_data(
            audio_data=audio_data,
            rate=speed_rate,
            sr=sample_rate,
        )
        result.data = audio_data
