import numpy as np
from pydub import AudioSegment
from typing import Any, List, Dict
from scipy.io.wavfile import write
import io
from modules.utils.audio import time_stretch, pitch_shift
from modules import generate_audio
from modules.normalization import text_normalize
import logging
import json
import random

from modules.speaker import Speaker

logger = logging.getLogger(__name__)


def audio_data_to_segment(audio_data, sr):
    byte_io = io.BytesIO()
    write(byte_io, rate=sr, data=audio_data)
    byte_io.seek(0)

    return AudioSegment.from_file(byte_io, format="wav")


def combine_audio_segments(audio_segments: list) -> AudioSegment:
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


class SynthesizeSegments:
    batch_default_spk_seed = int(np.random.randint(0, 2**32 - 1))
    batch_default_infer_seed = int(np.random.randint(0, 2**32 - 1))

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size

    def segment_to_generate_params(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        text = segment.get("text", "")
        is_end = segment.get("is_end", False)

        text = str(text).strip()

        attrs = segment.get("attrs", {})
        spk = attrs.get("spk", "")
        if isinstance(spk, str):
            spk = int(spk)
        seed = to_number(attrs.get("seed", ""), int, -1)
        top_k = to_number(attrs.get("top_k", ""), int, None)
        top_p = to_number(attrs.get("top_p", ""), float, None)
        temp = to_number(attrs.get("temp", ""), float, None)

        prompt1 = attrs.get("prompt1", "")
        prompt2 = attrs.get("prompt2", "")
        prefix = attrs.get("prefix", "")
        disable_normalize = attrs.get("normalize", "") == "False"

        params = {
            "text": text,
            "temperature": temp if temp is not None else 0.3,
            "top_P": top_p if top_p is not None else 0.5,
            "top_K": top_k if top_k is not None else 20,
            "spk": spk if spk else -1,
            "infer_seed": seed if seed else -1,
            "prompt1": prompt1 if prompt1 else "",
            "prompt2": prompt2 if prompt2 else "",
            "prefix": prefix if prefix else "",
        }

        if not disable_normalize:
            params["text"] = text_normalize(text, is_end=is_end)

        # Set default values for spk and infer_seed
        if params["spk"] == -1:
            params["spk"] = self.batch_default_spk_seed
        if params["infer_seed"] == -1:
            params["infer_seed"] = self.batch_default_infer_seed

        return params

    def bucket_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        # Create a dictionary to hold buckets
        buckets = {}
        for segment in segments:
            params = self.segment_to_generate_params(segment)

            key_params = params
            if isinstance(key_params.get("spk"), Speaker):
                key_params["spk"] = str(key_params["spk"].id)
            key = json.dumps(
                {k: v for k, v in key_params.items() if k != "text"}, sort_keys=True
            )
            if params["spk"] == -1 or params["infer_seed"] == -1:
                key = random.random()
                buckets[key] = [segment]
            else:
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(segment)

        # Convert dictionary to list of buckets
        bucket_list = list(buckets.values())
        return bucket_list

    def synthesize_segments(self, segments: List[Dict[str, Any]]) -> List[AudioSegment]:
        audio_segments = [None] * len(
            segments
        )  # Create a list with the same length as segments
        buckets = self.bucket_segments(segments)
        logger.debug(f"segments len: {len(segments)}")
        logger.debug(f"bucket pool size: {len(buckets)}")
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                param_arr = [
                    self.segment_to_generate_params(segment) for segment in batch
                ]
                texts = [params["text"] for params in param_arr]

                params = param_arr[0]  # Use the first segment to get the parameters
                audio_datas = generate_audio.generate_audio_batch(
                    texts=texts,
                    temperature=params["temperature"],
                    top_P=params["top_P"],
                    top_K=params["top_K"],
                    spk=params["spk"],
                    infer_seed=params["infer_seed"],
                    prompt1=params["prompt1"],
                    prompt2=params["prompt2"],
                    prefix=params["prefix"],
                )
                for idx, segment in enumerate(batch):
                    (sr, audio_data) = audio_datas[idx]
                    rate = float(segment.get("rate", "1.0"))
                    volume = float(segment.get("volume", "0"))
                    pitch = float(segment.get("pitch", "0"))

                    audio_segment = audio_data_to_segment(audio_data, sr)
                    audio_segment = apply_prosody(audio_segment, rate, volume, pitch)
                    original_index = segments.index(
                        segment
                    )  # Get the original index of the segment
                    audio_segments[original_index] = (
                        audio_segment  # Place the audio_segment in the correct position
                    )

        return audio_segments


def generate_audio_segment(
    text: str,
    spk: int = -1,
    seed: int = -1,
    top_p: float = 0.5,
    top_k: int = 20,
    temp: float = 0.3,
    prompt1: str = "",
    prompt2: str = "",
    prefix: str = "",
    enable_normalize=True,
    is_end: bool = False,
) -> AudioSegment:
    if enable_normalize:
        text = text_normalize(text, is_end=is_end)

    logger.debug(f"generate segment: {text}")

    sample_rate, audio_data = generate_audio.generate_audio(
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

    byte_io = io.BytesIO()
    write(byte_io, sample_rate, audio_data)
    byte_io.seek(0)

    return AudioSegment.from_file(byte_io, format="wav")


def synthesize_segment(segment: Dict[str, Any]) -> AudioSegment | None:
    if "break" in segment:
        pause_segment = AudioSegment.silent(duration=segment["break"])
        return pause_segment

    attrs = segment.get("attrs", {})
    text = segment.get("text", "")
    is_end = segment.get("is_end", False)

    text = str(text).strip()

    if text == "":
        return None

    spk = attrs.get("spk", "")
    if isinstance(spk, str):
        spk = int(spk)
    seed = to_number(attrs.get("seed", ""), int, -1)
    top_k = to_number(attrs.get("top_k", ""), int, None)
    top_p = to_number(attrs.get("top_p", ""), float, None)
    temp = to_number(attrs.get("temp", ""), float, None)

    prompt1 = attrs.get("prompt1", "")
    prompt2 = attrs.get("prompt2", "")
    prefix = attrs.get("prefix", "")
    disable_normalize = attrs.get("normalize", "") == "False"

    audio_segment = generate_audio_segment(
        text,
        enable_normalize=not disable_normalize,
        spk=spk,
        seed=seed,
        top_k=top_k,
        top_p=top_p,
        temp=temp,
        prompt1=prompt1,
        prompt2=prompt2,
        prefix=prefix,
        is_end=is_end,
    )

    rate = float(attrs.get("rate", "1.0"))
    volume = float(attrs.get("volume", "0"))
    pitch = float(attrs.get("pitch", "0"))

    audio_segment = apply_prosody(audio_segment, rate, volume, pitch)

    return audio_segment


# 示例使用
if __name__ == "__main__":
    ssml_segments = [
        {
            "text": "大🍌，一条大🍌，嘿，你的感觉真的很奇妙  [lbreak]",
            "attrs": {"spk": 2, "temp": 0.1, "seed": 42},
        },
        {
            "text": "大🍉，一个大🍉，嘿，你的感觉真的很奇妙  [lbreak]",
            "attrs": {"spk": 2, "temp": 0.1, "seed": 42},
        },
        {
            "text": "大🍌，一条大🍌，嘿，你的感觉真的很奇妙  [lbreak]",
            "attrs": {"spk": 2, "temp": 0.3, "seed": 42},
        },
    ]

    synthesizer = SynthesizeSegments(batch_size=2)
    audio_segments = synthesizer.synthesize_segments(ssml_segments)
    combined_audio = combine_audio_segments(audio_segments)
    combined_audio.export("output.wav", format="wav")
