import pytest

from modules import normalization
from modules.normalization import text_normalize

normalization.DISABLE_UNK_TOKEN_CHECK = True


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("我喜欢吃🍏", "我喜欢吃青苹果"),
        ("I like eating 🍏", "I like eating green_apple"),
    ],
)
@pytest.mark.normalize
def test_text_normalize(input_text, expected_output):
    assert text_normalize(input_text) == expected_output
