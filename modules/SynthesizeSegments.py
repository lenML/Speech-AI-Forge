import copy
from box import Box
from pydub import AudioSegment
from typing import List, Union
from scipy.io.wavfile import write
import io
from modules.SentenceSplitter import SentenceSplitter
from modules.api.utils import calc_spk_style
from modules.ssml_parser.SSMLParser import SSMLSegment, SSMLBreak, SSMLContext
from modules.utils import rng
from modules.utils.audio import time_stretch, pitch_shift
from modules import generate_audio
from modules.normalization import text_normalize
import logging
import json

from modules.speaker import Speaker, speaker_mgr

logger = logging.getLogger(__name__)


def audio_data_to_segment(audio_data, sr):
    byte_io = io.BytesIO()
    write(byte_io, rate=sr, data=audio_data)
    byte_io.seek(0)

    return AudioSegment.from_file(byte_io, format="wav")


def combine_audio_segments(audio_segments: list[AudioSegment]) -> AudioSegment:
    combined_audio = AudioSegment.empty()
    for segment in audio_segments:
        combined_audio += segment
    return combined_audio


def apply_prosody(
    audio_segment: AudioSegment, rate: float, volume: float, pitch: float
) -> AudioSegment:
    if rate != 1:
        audio_segment = time_stretch(audio_segment, rate)

    if volume != 0:
        audio_segment += volume

    if pitch != 0:
        audio_segment = pitch_shift(audio_segment, pitch)

    return audio_segment


def to_number(value, t, default=0):
    try:
        number = t(value)
        return number
    except (ValueError, TypeError) as e:
        return default


class TTSAudioSegment(Box):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._type = kwargs.get("_type", "voice")
        self.text = kwargs.get("text", "")
        self.temperature = kwargs.get("temperature", 0.3)
        self.top_P = kwargs.get("top_P", 0.5)
        self.top_K = kwargs.get("top_K", 20)
        self.spk = kwargs.get("spk", -1)
        self.infer_seed = kwargs.get("infer_seed", -1)
        self.prompt1 = kwargs.get("prompt1", "")
        self.prompt2 = kwargs.get("prompt2", "")
        self.prefix = kwargs.get("prefix", "")


class SynthesizeSegments:
    def __init__(self, batch_size: int = 8, eos="", spliter_thr=100):
        self.batch_size = batch_size
        self.batch_default_spk_seed = rng.np_rng()
        self.batch_default_infer_seed = rng.np_rng()
        self.eos = eos
        self.spliter_thr = spliter_thr

    def segment_to_generate_params(
        self, segment: Union[SSMLSegment, SSMLBreak]
    ) -> TTSAudioSegment:
        if isinstance(segment, SSMLBreak):
            return TTSAudioSegment(_type="break")

        if segment.get("params", None) is not None:
            params = segment.get("params")
            text = segment.get("text", None) or segment.text or ""
            return TTSAudioSegment(**params, text=text)

        text = segment.get("text", None) or segment.text or ""
        is_end = segment.get("is_end", False)

        text = str(text).strip()

        attrs = segment.attrs
        spk = attrs.spk
        style = attrs.style

        ss_params = calc_spk_style(spk, style)

        if "spk" in ss_params:
            spk = ss_params["spk"]

        seed = to_number(attrs.seed, int, ss_params.get("seed") or -1)
        top_k = to_number(attrs.top_k, int, None)
        top_p = to_number(attrs.top_p, float, None)
        temp = to_number(attrs.temp, float, None)

        prompt1 = attrs.prompt1 or ss_params.get("prompt1")
        prompt2 = attrs.prompt2 or ss_params.get("prompt2")
        prefix = attrs.prefix or ss_params.get("prefix")
        disable_normalize = attrs.get("normalize", "") == "False"

        seg = TTSAudioSegment(
            _type="voice",
            text=text,
            temperature=temp if temp is not None else 0.3,
            top_P=top_p if top_p is not None else 0.5,
            top_K=top_k if top_k is not None else 20,
            spk=spk if spk else -1,
            infer_seed=seed if seed else -1,
            prompt1=prompt1 if prompt1 else "",
            prompt2=prompt2 if prompt2 else "",
            prefix=prefix if prefix else "",
        )

        if not disable_normalize:
            seg.text = text_normalize(text, is_end=is_end)

        # NOTE 每个batch的默认seed保证前后一致即使是没设置spk的情况
        if seg.spk == -1:
            seg.spk = self.batch_default_spk_seed
        if seg.infer_seed == -1:
            seg.infer_seed = self.batch_default_infer_seed

        return seg

    def process_break_segments(
        self,
        src_segments: List[SSMLBreak],
        bucket_segments: List[SSMLBreak],
        audio_segments: List[AudioSegment],
    ):
        for segment in bucket_segments:
            index = src_segments.index(segment)
            audio_segments[index] = AudioSegment.silent(
                duration=int(segment.attrs.duration)
            )

    def process_voice_segments(
        self,
        src_segments: List[SSMLSegment],
        bucket: List[SSMLSegment],
        audio_segments: List[AudioSegment],
    ):
        for i in range(0, len(bucket), self.batch_size):
            batch = bucket[i : i + self.batch_size]
            param_arr = [self.segment_to_generate_params(segment) for segment in batch]
            texts = [params.text + self.eos for params in param_arr]

            params = param_arr[0]
            audio_datas = generate_audio.generate_audio_batch(
                texts=texts,
                temperature=params.temperature,
                top_P=params.top_P,
                top_K=params.top_K,
                spk=params.spk,
                infer_seed=params.infer_seed,
                prompt1=params.prompt1,
                prompt2=params.prompt2,
                prefix=params.prefix,
            )
            for idx, segment in enumerate(batch):
                sr, audio_data = audio_datas[idx]
                rate = float(segment.get("rate", "1.0"))
                volume = float(segment.get("volume", "0"))
                pitch = float(segment.get("pitch", "0"))

                audio_segment = audio_data_to_segment(audio_data, sr)
                audio_segment = apply_prosody(audio_segment, rate, volume, pitch)
                original_index = src_segments.index(segment)
                audio_segments[original_index] = audio_segment

    def bucket_segments(
        self, segments: List[Union[SSMLSegment, SSMLBreak]]
    ) -> List[List[Union[SSMLSegment, SSMLBreak]]]:
        buckets = {"<break>": []}
        for segment in segments:
            if isinstance(segment, SSMLBreak):
                buckets["<break>"].append(segment)
                continue

            params = self.segment_to_generate_params(segment)

            if isinstance(params.spk, Speaker):
                params.spk = str(params.spk.id)

            key = json.dumps(
                {k: v for k, v in params.items() if k != "text"}, sort_keys=True
            )
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(segment)

        return buckets

    def split_segments(self, segments: List[Union[SSMLSegment, SSMLBreak]]):
        """
        将 segments 中的 text 经过 spliter 处理成多个 segments
        """
        spliter = SentenceSplitter(threshold=self.spliter_thr)
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
                ret_segments.append(
                    SSMLSegment(
                        text=sentence,
                        attrs=segment.attrs.copy(),
                        params=copy.copy(segment.params),
                    )
                )

        return ret_segments

    def synthesize_segments(
        self, segments: List[Union[SSMLSegment, SSMLBreak]]
    ) -> List[AudioSegment]:
        segments = self.split_segments(segments)
        audio_segments = [None] * len(segments)
        buckets = self.bucket_segments(segments)

        break_segments = buckets.pop("<break>")
        self.process_break_segments(segments, break_segments, audio_segments)

        buckets = list(buckets.values())

        for bucket in buckets:
            self.process_voice_segments(segments, bucket, audio_segments)

        return audio_segments


# 示例使用
if __name__ == "__main__":
    ctx1 = SSMLContext()
    ctx1.spk = 1
    ctx1.seed = 42
    ctx1.temp = 0.1
    ctx2 = SSMLContext()
    ctx2.spk = 2
    ctx2.seed = 42
    ctx2.temp = 0.1
    ssml_segments = [
        SSMLSegment(text="大🍌，一条大🍌，嘿，你的感觉真的很奇妙", attrs=ctx1.copy()),
        SSMLBreak(duration_ms=1000),
        SSMLSegment(text="大🍉，一个大🍉，嘿，你的感觉真的很奇妙", attrs=ctx1.copy()),
        SSMLSegment(text="大🍊，一个大🍊，嘿，你的感觉真的很奇妙", attrs=ctx2.copy()),
    ]

    synthesizer = SynthesizeSegments(batch_size=2)
    audio_segments = synthesizer.synthesize_segments(ssml_segments)
    print(audio_segments)
    combined_audio = combine_audio_segments(audio_segments)
    combined_audio.export("output.wav", format="wav")
