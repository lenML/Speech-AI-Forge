import numpy as np
import pytest
from lxml import etree

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.factory import PipelineFactory
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.core.ssml.SSMLParser import SSMLBreak, SSMLSegment, create_ssml_v01_parser
from tests.pipeline.misc import load_audio, load_audio_wav, save_audio
from tests.pipeline.voice_clone_pipe import run_voice_clone_pipeline_test


async def run_ssml_pipeline(ssml: str, filename="ssml_test.wav"):
    out_audio_path = f"./tests/test_outputs/{filename}"

    pipe0 = PipelineFactory.create_chattts_pipeline(
        ctx=TTSPipelineContext(
            ssml=ssml,
            tts_config=TTSConfig(
                mid="chat-tts",
            ),
            infer_config=InferConfig(eos=" ", sync_gen=True),
        ),
    )

    audio_sr, audio_data = await pipe0.generate()
    assert audio_data.size != 0
    save_audio(
        #
        file_path=out_audio_path,
        audio_data=audio_data,
        sample_rate=audio_sr,
    )
    # 检查文件不为空
    assert load_audio(out_audio_path)[1].size != 0

    return out_audio_path, audio_sr, audio_data


@pytest.mark.ssml_gen
@pytest.mark.asyncio
async def test_ssml_gen1():
    ssml = """
    <speak version="0.1">
        <voice spk="xiaoyan" style="news">
            <prosody rate="fast">你好</prosody>
            <break time="500ms"/>
            <prosody rate="slow">你好</prosody>
        </voice>
    </speak>
    """

    await run_ssml_pipeline(ssml, filename="ssml_gen1.wav")
