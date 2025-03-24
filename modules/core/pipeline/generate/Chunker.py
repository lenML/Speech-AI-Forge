import re
from typing import List

from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.generate.SimpleTokenizer import RegexpTokenizer
from modules.core.pipeline.generate.SsmlNormalizer import SsmlNormalizer
from modules.core.ssml.SSMLParser import SSMLContext, get_ssml_parser_for
from modules.core.tools.SentenceSplitter import SentenceSplitter


class TTSChunker:
    # NOTE: 为了性能，所以用这个作为 tokenizer
    tokenizer = RegexpTokenizer()

    def __init__(self, context: TTSPipelineContext) -> None:
        self.context = context

    def segments(self) -> List[TTSSegment]:
        text = self.context.text
        texts = self.context.texts
        ssml = self.context.ssml

        if text is not None and len(text.strip()) > 0:
            return self.text_segments()
        if ssml is not None and len(ssml.strip()) > 0:
            return self.ssml_segments()
        if texts is not None and len(texts) > 0:
            return self.texts_segments()
        raise ValueError("No input text or ssml")

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

        # NOTE: char tokenizer
        # return [ord(char) for char in text]

        # NOTE: 使用 model.tokenizer 延迟有点高，最好还是用 tokenizer ，需要排查具体是哪的问题
        # model = zoo.model_zoo.get_model(self.context.tts_config.mid)
        # return model.encode(text)

    def make_segment(self, text: str):
        temperature = self.context.tts_config.temperature
        top_P = self.context.tts_config.top_p
        top_K = self.context.tts_config.top_k
        spk = self.context.spk
        infer_seed = self.context.infer_config.seed
        prompt1 = self.context.tts_config.prompt1
        prompt2 = self.context.tts_config.prompt2
        prefix = self.context.tts_config.prefix
        emotion = self.context.tts_config.emotion

        eos = self.context.infer_config.eos

        if not text.strip().endswith(eos):
            text = text + eos

        return TTSSegment(
            _type="audio",
            text=text,
            temperature=temperature,
            top_p=top_P,
            top_k=top_K,
            spk=spk,
            infer_seed=infer_seed,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
            emotion=emotion,
        )

    def text_segments(self):
        spliter_threshold = self.context.infer_config.spliter_threshold
        text = self.context.text

        spliter = SentenceSplitter(threshold=spliter_threshold, tokenizer=self.tokenize)
        sentences = spliter.parse(text)

        text_segments = [self.make_segment(s) for s in sentences]
        return text_segments

    def create_ssml_ctx(self):
        ctx = SSMLContext()
        ctx.spk = self.context.spk

        ctx.style = self.context.tts_config.style
        ctx.volume = self.context.adjust_config.volume_gain_db
        ctx.rate = self.context.adjust_config.speed_rate
        ctx.pitch = self.context.adjust_config.pitch
        ctx.temp = self.context.tts_config.temperature
        ctx.top_p = self.context.tts_config.top_p
        ctx.top_k = self.context.tts_config.top_k
        ctx.seed = self.context.infer_config.seed
        ctx.prompt1 = self.context.tts_config.prompt1
        ctx.prompt2 = self.context.tts_config.prompt2
        ctx.prefix = self.context.tts_config.prefix

        return ctx

    def ssml_segments(self):
        ssml = self.context.ssml
        eos = self.context.infer_config.eos
        thr = self.context.infer_config.spliter_threshold

        parser = get_ssml_parser_for("0.1")
        parser_ctx = self.create_ssml_ctx()
        segments = parser.parse(ssml=ssml, root_ctx=parser_ctx)
        normalizer = SsmlNormalizer(context=self.context, eos=eos, spliter_thr=thr)
        segments = normalizer.normalize(segments)
        return segments

    def texts_segments(self):
        texts = self.context.texts
        return [self.make_segment(t) for t in texts]
