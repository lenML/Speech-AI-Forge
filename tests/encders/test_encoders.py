import pytest

from modules.core.handler.datacls.audio_model import EncoderConfig
from modules.core.handler.datacls.tts_model import TTSConfig
from modules.core.handler.TTSHandler import TTSHandler
from modules.core.spk.SpkMgr import spk_mgr

# 这里测试 encoder 是否可以正常工作
# 正常工作的定义是输出音频正常无噪音无变速

need_test_models = ["chat-tts", "fish-speech", "cosy-voice"]


@pytest.mark.parametrize(
    "model_id, format",
    [
        # raw 格式就是 wav，只是直接输出pcm
        ("chat-tts", "raw"),
        ("fish-speech", "raw"),
        ("cosy-voice", "raw"),
        ("chat-tts", "mp3"),
        ("fish-speech", "mp3"),
        ("cosy-voice", "mp3"),
        ("chat-tts", "wav"),
        ("fish-speech", "wav"),
        ("cosy-voice", "wav"),
    ],
)
@pytest.mark.encoders
@pytest.mark.asyncio
async def test_encoders(model_id, format):
    spk_mona = spk_mgr.get_speaker("mona")
    handler = TTSHandler(
        text_content="云想衣裳花想容，春风拂槛露华浓。 若非群玉山头见，会向瑶台月下逢。",
        spk=spk_mona,
        tts_config=TTSConfig(mid=model_id),
        encoder_config=EncoderConfig(format=format),
    )
    file_bytes = await handler._enqueue_to_bytes()

    ext = format
    if format == "raw":
        ext = "wav"

    # 1. 不为空
    assert len(file_bytes) > 0
    # 2. 保存到 tests/test_outputs 之下，然后人工检查
    with open(f"tests/test_outputs/test_encder_{model_id}_{format}.{ext}", "wb") as f:
        f.write(file_bytes)
