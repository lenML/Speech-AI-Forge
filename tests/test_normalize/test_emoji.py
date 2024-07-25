import pytest

from modules.core.tn import ChatTtsTN

ChatTtsTN.ChatTtsTN.remove_block("replace_unk_tokens")


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("我喜欢吃🍏", "我喜欢吃青苹果"),
        ("I like eating 🍏", "I like eating green_apple"),
    ],
)
@pytest.mark.normalize
def test_text_normalize(input_text, expected_output):
    assert ChatTtsTN.ChatTtsTN.normalize(input_text) == expected_output
