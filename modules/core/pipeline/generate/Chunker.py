from typing import List

from modules.core.models import zoo
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.generate.SsmlNormalizer import SsmlNormalizer
from modules.core.ssml.SSMLParser import SSMLContext, get_ssml_parser_for
from modules.core.tools.SentenceSplitter import SentenceSplitter


class TTSChunker:
    def __init__(self, context: TTSPipelineContext) -> None:
        self.context = context

    def segments(self) -> List[TTSSegment]:
        text = self.context.text
        ssml = self.context.ssml

        if text is not None and len(text.strip()) > 0:
            return self.text_segments()
        if ssml is not None and len(ssml.strip()) > 0:
            return self.ssml_segments()
        raise ValueError("No input text or ssml")

    def tokenize(self, text: str) -> list[int]:
        model = zoo.model_zoo.get_model(self.context.tts_config.mid)
        return model.encode(text)

    def text_segments(self):
        spliter_threshold = self.context.infer_config.spliter_threshold
        text = self.context.text

        temperature = self.context.tts_config.temperature
        top_P = self.context.tts_config.top_p
        top_K = self.context.tts_config.top_k
        spk = self.context.spk
        infer_seed = self.context.infer_config.seed
        # 这个只有 chattts 需要，并且没必要填 false...
        # use_decoder = self.context.tts_config.use_decoder
        use_decoder = True
        prompt1 = self.context.tts_config.prompt1
        prompt2 = self.context.tts_config.prompt2
        prefix = self.context.tts_config.prefix
        emotion = self.context.tts_config.emotion

        eos = self.context.infer_config.eos

        spliter = SentenceSplitter(threshold=spliter_threshold, tokenizer=self.tokenize)
        sentences = spliter.parse(text)

        text_segments = [
            TTSSegment(
                _type="audio",
                text=s + eos,
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
            for s in sentences
        ]
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
        # ctx.noramalize = self.context.tts_config.normalize
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
