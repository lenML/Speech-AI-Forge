import pytest

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.factory import PipelineFactory
from tests.pipeline.misc import load_audio, save_audio


@pytest.mark.post_process
def test_process_audio():
    src_audio_path = "./tests/test_inputs/cosyvoice_out1.wav"
    out_audio_path = "./tests/test_outputs/post_process_out1.wav"

    src_audio = load_audio(src_audio_path)
    pipe0 = PipelineFactory.create_postprocess_pipeline(
        audio=src_audio,
        ctx=TTSPipelineContext(
            adjust_config=AdjustConfig(
                pitch=0,
                speed_rate=1.5,
                volume_gain_db=0,
                normalize=True,
                headroom=0.1,
            )
        ),
    )

    audio_sr, audio_data = pipe0.generate()
    assert audio_data.size != 0
    save_audio(
        #
        file_path=out_audio_path,
        audio_data=audio_data,
        sample_rate=audio_sr,
    )
    # 检查文件不为空
    assert load_audio(out_audio_path)[1].size != 0
