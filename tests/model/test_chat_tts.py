import pytest

from modules.core.models.tts.ChatTtsModel import ChatTTSModel
from modules.core.spk.TTSSpeaker import TTSSpeaker


@pytest.mark.model_chat_tts
def test_create_speaker_from_seed():
    # NOTE: 主要是测试 chat tts 官方库有没有修改，可能导致这个函数错误
    spk = ChatTTSModel.create_speaker_from_seed(42)
    assert isinstance(spk, TTSSpeaker)
