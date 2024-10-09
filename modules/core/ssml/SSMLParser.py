import logging
from typing import List, Literal, Union

import python_ms as ms
from box import Box
from lxml import etree


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
        self.emotion = None
        self.duration = None

        super().__init__(*args, **kwargs)


class SSMLSegment(Box):
    def __init__(self, text: str, attrs=SSMLContext(), params=None):
        self.attrs = SSMLContext(**attrs)
        self.text = text
        self.params = params


class SSMLBreak:
    def __init__(self, duration_ms: int):
        self.attrs = Box(**{"duration": duration_ms})


class SSMLParser:
    """
    基础类，在其他模块中不应该手动创建
    用法看 create_ssml_v01_parser
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("SSMLParser.__init__()")
        self.resolvers = []

    def resolver(self, tag: str):
        def decorator(func):
            self.resolvers.append((tag, func))
            return func

        return decorator

    def parse(
        self, ssml: str, root_ctx=SSMLContext()
    ) -> List[Union[SSMLSegment, SSMLBreak]]:
        root = etree.fromstring(ssml)

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


def create_ssml_v01_parser():
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
        ctx.emotion = element.get("emotion", ctx.emotion)
        ctx.duration = element.get("duration", ctx.duration)

        if isinstance(ctx.duration, str):
            ctx.duration = ms(ctx.duration)

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
        time_str = element.get("time", element.get("duration", "0"))
        time_ms = ms(time_str)
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
        ctx.emotion = element.get("emotion", ctx.emotion)
        ctx.duration = element.get("duration", ctx.duration)

        if isinstance(ctx.duration, str):
            ctx.duration = ms(ctx.duration)

        if element.text and element.text.strip():
            segments.append(SSMLSegment(element.text.strip(), ctx))

    return parser


def get_ssml_parser_for(version: Literal["0.1"]):
    if version == "0.1":
        return create_ssml_v01_parser()
    raise ValueError(f"Unsupported SSML version {version}")


if __name__ == "__main__":
    parser = create_ssml_v01_parser()

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
