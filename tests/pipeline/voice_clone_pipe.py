from typing import Callable

import numpy as np
import pytest

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.factory import PipelineFactory
from modules.core.pipeline.pipeline import AudioPipeline
from modules.core.spk.TTSSpeaker import TTSSpeaker
from tests.pipeline.misc import load_audio, load_audio_wav, save_audio

# Pipeline function type, which returns a pipeline with a generate() method
PipelineFunc = Callable[
    [TTSPipelineContext], "AudioPipeline"
]  # Type hint for pipeline function


async def run_voice_clone_pipeline_test(
    pipeline_func: PipelineFunc,
    voice_target_path: str,
    voice_target_text: str,
    out_audio_path: str,
    text: str = "你好，这里是音色克隆测试~",
    eos: str = " ",
) -> None:
    """辅助函数：执行音色克隆测试并保存结果音频。

    参数：
    - pipeline_func: 用于创建 TTS 流水线的函数。
    - voice_target_path: 目标语音的文件路径。
    - voice_target_text: 目标语音的文本。
    - out_audio_path: 输出音频的文件路径。
    - text: 生成音频时使用的文本。
    """
    voice_target: bytes = load_audio_wav(voice_target_path)
    voice_spk: TTSSpeaker = TTSSpeaker.from_ref_wav_bytes(
        ref_wav=voice_target, text=voice_target_text
    )

    pipe: AudioPipeline = pipeline_func(
        ctx=TTSPipelineContext(
            text=text,
            tts_config=TTSConfig(mid="cosy-voice"),
            infer_config=InferConfig(eos=eos, sync_gen=True),
            spk=voice_spk,
        ),
    )

    audio_sr, audio_data = await pipe.generate()  # Tuple[int, np.ndarray]
    assert audio_data.dtype == np.float32
    assert audio_data.size != 0

    save_audio(file_path=out_audio_path, audio_data=audio_data, sample_rate=audio_sr)
    # 检查文件不为空
    assert load_audio(out_audio_path)[1].size != 0
