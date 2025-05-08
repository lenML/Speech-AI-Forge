import numpy as np
import pytest

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.factory import PipelineFactory
from modules.core.spk.TTSSpeaker import TTSSpeaker
from tests.pipeline.misc import load_audio, load_audio_wav, save_audio
from tests.pipeline.voice_clone_pipe import run_voice_clone_pipeline_test


@pytest.mark.pipe_cosyvoice
@pytest.mark.asyncio
async def test_cosy_voice_clone_pipe():
    voice_target_path = "./tests/test_inputs/chattts_out1.wav"
    out_audio_path = "./tests/test_outputs/pipe_cosyvoice_voice_clone_out1.wav"

    await run_voice_clone_pipeline_test(
        pipeline_func=PipelineFactory.create_cosyvoice_pipeline,
        voice_target_path=voice_target_path,
        voice_target_text="这是一个测试文本。",
        out_audio_path=out_audio_path,
    )
