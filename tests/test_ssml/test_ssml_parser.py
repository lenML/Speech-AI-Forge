import pytest
from lxml import etree

from modules.core.ssml.SSMLParser import SSMLBreak, SSMLSegment, create_ssml_v01_parser


@pytest.fixture
def parser():
    return create_ssml_v01_parser()


@pytest.mark.ssml_parser
def test_speak_tag(parser):
    ssml = """
    <speak version="0.1">
        <voice spk="xiaoyan" style="news">
            <prosody rate="fast">你好</prosody>
            <break time="500ms"/>
            <prosody rate="slow">你好</prosody>
        </voice>
    </speak>
    """
    segments = parser.parse(ssml)
    assert len(segments) == 3
    assert isinstance(segments[0], SSMLSegment)
    assert segments[0].text == "你好"
    assert segments[0].attrs.rate == "fast"
    assert isinstance(segments[1], SSMLBreak)
    assert segments[1].attrs.duration == 500
    assert isinstance(segments[2], SSMLSegment)
    assert segments[2].text == "你好"
    assert segments[2].attrs.rate == "slow"


@pytest.mark.ssml_parser
def test_voice_tag(parser):
    ssml = """
    <speak version="0.1">
        <voice spk="xiaoyan" style="news">你好</voice>
    </speak>
    """
    segments = parser.parse(ssml)
    assert len(segments) == 1
    assert isinstance(segments[0], SSMLSegment)
    assert segments[0].text == "你好"
    assert segments[0].attrs.spk == "xiaoyan"
    assert segments[0].attrs.style == "news"


@pytest.mark.ssml_parser
def test_break_tag(parser):
    ssml = """
    <speak version="0.1">
        <break time="500ms"/>
    </speak>
    """
    segments = parser.parse(ssml)
    assert len(segments) == 1
    assert isinstance(segments[0], SSMLBreak)
    assert segments[0].attrs.duration == 500


@pytest.mark.ssml_parser
def test_prosody_tag(parser):
    ssml = """
    <speak version="0.1">
        <prosody rate="fast">你好</prosody>
    </speak>
    """
    segments = parser.parse(ssml)
    assert len(segments) == 1
    assert isinstance(segments[0], SSMLSegment)
    assert segments[0].text == "你好"
    assert segments[0].attrs.rate == "fast"


@pytest.mark.ssml_parser
def test_unsupported_version(parser):
    ssml = """
    <speak version="0.2">
        <voice spk="xiaoyan" style="news">你好</voice>
    </speak>
    """
    with pytest.raises(ValueError, match=r"Unsupported SSML version 0.2"):
        parser.parse(ssml)


@pytest.mark.ssml_parser
def test_unsupported_tag(parser):
    ssml = """
    <speak version="0.1">
        <unsupported>你好</unsupported>
    </speak>
    """
    with pytest.raises(NotImplementedError, match=r"Tag unsupported not supported."):
        parser.parse(ssml)
