import pytest

from modules.core.pipeline.factory import PipelineFactory
from tests.pipeline.voice_clone_pipe import run_voice_clone_pipeline_test


@pytest.mark.pipe_f5_tts
@pytest.mark.asyncio
async def test_f5_voice_clone_pipe():
    voice_target_path = "./tests/test_inputs/chattts_out1.wav"
    out_audio_path = "./tests/test_outputs/pipe_f5_voice_clone_out1.wav"

    await run_voice_clone_pipeline_test(
        pipeline_func=PipelineFactory.create_f5_tts_pipeline,
        voice_target_path=voice_target_path,
        voice_target_text="这是一个测试文本。",
        out_audio_path=out_audio_path,
    )
