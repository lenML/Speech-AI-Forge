import copy
import re
from typing import List, Union

from modules.api.utils import calc_spk_style, to_number
from modules.core.models import zoo
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment
from modules.core.pipeline.generate.SimpleTokenizer import RegexpTokenizer
from modules.core.ssml.SSMLParser import SSMLBreak, SSMLSegment
from modules.core.tools.SentenceSplitter import SentenceSplitter
from modules.utils import rng


class SsmlNormalizer:
    """
    SSML segment normalizer

    input: List[SSMLSegment, SSMLBreak] => output: List[TTSSegment]

    and split the text in SSMLSegment into multiple segments if the text is too long
    """

    # NOTE: 为了性能，所以用这个代替 model.tokenizer
    tokenizer = RegexpTokenizer()

    def __init__(self, context: TTSPipelineContext, eos="", spliter_thr=100):
        self.batch_default_spk_seed = rng.np_rng()
        self.batch_default_infer_seed = rng.np_rng()
        self.eos = eos
        self.spliter_thr = spliter_thr
        self.context = context

    def get_eoslike_arr(self):
        # 根据模型来确定可能的 eos
        mid = self.context.tts_config.mid
        eos_arr = ["。", "."]
        if mid == "chat-tts":
            eos_arr = ["[uv_break]", "[v_break]", "[lbreak]", "[llbreak]"]
        else:
            # TODO 其他模型可能也需要判断
            pass
        return eos_arr

    def append_eos(self, text: str):
        text = text.strip()
        eos_arr = self.get_eoslike_arr()
        has_eos = False
        for eos in eos_arr:
            if eos in text:
                has_eos = True
                break
        if not has_eos:
            text += self.eos
        return text

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

        # model = zoo.model_zoo.get_model(self.context.tts_config.mid)
        # return model.encode(text)

    def convert_ssml_seg(self, segment: Union[SSMLSegment, SSMLBreak]) -> TTSSegment:
        if isinstance(segment, SSMLBreak):
            return TTSSegment(_type="break", duration_ms=segment.attrs.duration)

        tts_config = self.context.tts_config
        infer_config = self.context.infer_config

        params = segment.params
        text = segment.text or ""
        text = text.strip()

        if text:
            text = self.append_eos(text)

        if params is not None:
            return TTSSegment(
                _type="voice",
                text=text,
                temperature=params.get("temperature", tts_config.temperature),
                top_P=params.get("top_p", tts_config.top_p),
                top_K=params.get("top_k", tts_config.top_k),
                spk=params.get("spk", None),
                infer_seed=params.get("seed", infer_config.seed),
                prompt1=params.get("prompt1", ""),
                prompt2=params.get("prompt2", ""),
                prefix=params.get("prefix", ""),
                emotion=params.get("emotion", ""),
                duration_ms=params.get("duration", None),
            )

        text = str(text).strip()

        attrs = segment.attrs
        spk = attrs.spk
        style = attrs.style
        emotion = attrs.emotion

        # FIXME: 这个逻辑有点...emmm 最好干掉
        ss_params = calc_spk_style(spk, style)

        if "spk" in ss_params:
            spk = ss_params["spk"]

        seed = to_number(attrs.seed, int, ss_params.get("seed") or -1)
        top_k = to_number(attrs.top_k, int, None)
        top_p = to_number(attrs.top_p, float, None)
        temp = to_number(attrs.temp, float, None)

        duration_ms = to_number(attrs.duration, float, None)
        speed_rate = to_number(attrs.rate, float, None)

        prompt1 = attrs.prompt1 or ss_params.get("prompt1")
        prompt2 = attrs.prompt2 or ss_params.get("prompt2")
        prefix = attrs.prefix or ss_params.get("prefix")

        seg = TTSSegment(
            _type="voice",
            text=text,
            temperature=temp or tts_config.temperature,
            top_p=top_p or tts_config.top_p,
            top_k=top_k or tts_config.top_k,
            spk=spk,
            infer_seed=seed,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
            emotion=emotion or style,
            duration_ms=duration_ms,
            speed_rate=speed_rate,
        )

        # NOTE 每个batch的默认seed保证前后一致即使是没设置spk的情况
        if seg.spk == -1:
            seg.spk = self.batch_default_spk_seed
        if seg.infer_seed == -1:
            seg.infer_seed = self.batch_default_infer_seed

        return seg

    def split_segments(self, segments: List[Union[SSMLSegment, SSMLBreak]]):
        """
        将 segments 中的 text 经过 spliter 处理成多个 segments
        """
        spliter_threshold = self.context.infer_config.spliter_threshold
        spliter = SentenceSplitter(threshold=spliter_threshold, tokenizer=self.tokenize)
        ret_segments: List[Union[SSMLSegment, SSMLBreak]] = []

        for segment in segments:
            if isinstance(segment, SSMLBreak):
                ret_segments.append(segment)
                continue

            text = segment.text
            if not text:
                continue

            sentences = spliter.parse(text)
            for sentence in sentences:
                seg = SSMLSegment(
                    text=sentence,
                    attrs=segment.attrs.copy(),
                    params=copy.copy(segment.params),
                )
                ret_segments.append(seg)
                setattr(seg, "_idx", len(ret_segments) - 1)

        def is_none_speak_segment(segment: SSMLSegment):
            text = segment.text.strip()
            regexp = r"\[[^\]]+?\]"
            text = re.sub(regexp, "", text)
            text = text.strip()
            if not text:
                return True
            return False

        # 将 none_speak 合并到前一个 speak segment
        for i in range(1, len(ret_segments)):
            segment = ret_segments[i]
            if not isinstance(segment, SSMLBreak) and is_none_speak_segment(
                segment=segment
            ):
                ret_segments[i - 1].text += ret_segments[i].text
                ret_segments[i].text = ""
        # 移除空的 segment
        ret_segments = [
            seg
            for seg in ret_segments
            if (isinstance(seg, SSMLSegment) and seg.text.strip())
            or isinstance(seg, SSMLBreak)
        ]

        return ret_segments

    def normalize(
        self, segments: list[Union[SSMLSegment, SSMLBreak]]
    ) -> list[TTSSegment]:
        segments = self.split_segments(segments)
        return [self.convert_ssml_seg(seg) for seg in segments]
