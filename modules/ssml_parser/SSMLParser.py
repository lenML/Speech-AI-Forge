from lxml import etree


from typing import Any, List, Dict, Union
import logging

from modules.data import styles_mgr
from modules.speaker import speaker_mgr
from box import Box
import copy


class SSMLContext(Box):
    def __init__(self, *args, **kwargs):
        self.parent: Union[SSMLContext, None] = None

        self.style = None
        self.spk = None
        self.volume = None
        self.rate = None
        self.pitch = None
        # tempurature
        self.temp = None
        self.top_p = None
        self.top_k = None
        self.seed = None
        self.noramalize = None
        self.prompt1 = None
        self.prompt2 = None
        self.prefix = None

        super().__init__(*args, **kwargs)


class SSMLSegment(Box):
    def __init__(self, text: str, attrs=SSMLContext(), params=None):
        self.attrs = SSMLContext(**attrs)
        self.text = text
        self.params = params


class SSMLBreak:
    def __init__(self, duration_ms: Union[str, int, float]):
        # TODO 支持其他单位
        duration_ms = int(str(duration_ms).replace("ms", ""))
        self.attrs = Box(**{"duration": duration_ms})


class SSMLParser:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("SSMLParser.__init__()")
        self.resolvers = []

    def resolver(self, tag: str):
        def decorator(func):
            self.resolvers.append((tag, func))
            return func

        return decorator

    def parse(self, ssml: str) -> List[Union[SSMLSegment, SSMLBreak]]:
        root = etree.fromstring(ssml)

        root_ctx = SSMLContext()
        segments: List[Union[SSMLSegment, SSMLBreak]] = []
        self.resolve(root, root_ctx, segments)

        return segments

    def resolve(
        self, element: etree.Element, context: SSMLContext, segments: List[SSMLSegment]
    ):
        resolver = [resolver for tag, resolver in self.resolvers if tag == element.tag]
        if len(resolver) == 0:
            raise NotImplementedError(f"Tag {element.tag} not supported.")
        else:
            resolver = resolver[0]

        resolver(element, context, segments, self)


def create_ssml_parser():
    parser = SSMLParser()

    @parser.resolver("speak")
    def tag_speak(
        element: etree.Element,
        context: Box,
        segments: List[Union[SSMLSegment, SSMLBreak]],
        parser: SSMLParser,
    ):
        ctx = context.copy() if context is not None else SSMLContext()

        version = element.get("version")
        if version != "0.1":
            raise ValueError(f"Unsupported SSML version {version}")

        for child in element:
            parser.resolve(child, ctx, segments)

    @parser.resolver("voice")
    def tag_voice(
        element: etree.Element,
        context: Box,
        segments: List[Union[SSMLSegment, SSMLBreak]],
        parser: SSMLParser,
    ):
        ctx = context.copy() if context is not None else SSMLContext()

        ctx.spk = element.get("spk", ctx.spk)
        ctx.style = element.get("style", ctx.style)
        ctx.spk = element.get("spk", ctx.spk)
        ctx.volume = element.get("volume", ctx.volume)
        ctx.rate = element.get("rate", ctx.rate)
        ctx.pitch = element.get("pitch", ctx.pitch)
        # tempurature
        ctx.temp = element.get("temp", ctx.temp)
        ctx.top_p = element.get("top_p", ctx.top_p)
        ctx.top_k = element.get("top_k", ctx.top_k)
        ctx.seed = element.get("seed", ctx.seed)
        ctx.noramalize = element.get("noramalize", ctx.noramalize)
        ctx.prompt1 = element.get("prompt1", ctx.prompt1)
        ctx.prompt2 = element.get("prompt2", ctx.prompt2)
        ctx.prefix = element.get("prefix", ctx.prefix)

        # 处理 voice 开头的文本
        if element.text and element.text.strip():
            segments.append(SSMLSegment(element.text.strip(), ctx))

        for child in element:
            parser.resolve(child, ctx, segments)

            # 处理 voice 结尾的文本
            if child.tail and child.tail.strip():
                segments.append(SSMLSegment(child.tail.strip(), ctx))

    @parser.resolver("break")
    def tag_break(
        element: etree.Element,
        context: Box,
        segments: List[Union[SSMLSegment, SSMLBreak]],
        parser: SSMLParser,
    ):
        time_ms = int(element.get("time", "0").replace("ms", ""))
        segments.append(SSMLBreak(time_ms))

    @parser.resolver("prosody")
    def tag_prosody(
        element: etree.Element,
        context: Box,
        segments: List[Union[SSMLSegment, SSMLBreak]],
        parser: SSMLParser,
    ):
        ctx = context.copy() if context is not None else SSMLContext()

        ctx.spk = element.get("spk", ctx.spk)
        ctx.style = element.get("style", ctx.style)
        ctx.spk = element.get("spk", ctx.spk)
        ctx.volume = element.get("volume", ctx.volume)
        ctx.rate = element.get("rate", ctx.rate)
        ctx.pitch = element.get("pitch", ctx.pitch)
        # tempurature
        ctx.temp = element.get("temp", ctx.temp)
        ctx.top_p = element.get("top_p", ctx.top_p)
        ctx.top_k = element.get("top_k", ctx.top_k)
        ctx.seed = element.get("seed", ctx.seed)
        ctx.noramalize = element.get("noramalize", ctx.noramalize)
        ctx.prompt1 = element.get("prompt1", ctx.prompt1)
        ctx.prompt2 = element.get("prompt2", ctx.prompt2)
        ctx.prefix = element.get("prefix", ctx.prefix)

        if element.text and element.text.strip():
            segments.append(SSMLSegment(element.text.strip(), ctx))

    return parser


if __name__ == "__main__":
    parser = create_ssml_parser()

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
    for segment in segments:
        if isinstance(segment, SSMLBreak):
            print("<break>", segment.attrs)
        elif isinstance(segment, SSMLSegment):
            print(segment.text, segment.attrs)
        else:
            raise ValueError("Unknown segment type")
