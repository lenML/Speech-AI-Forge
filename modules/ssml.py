from lxml import etree

from pydub import AudioSegment

from typing import Any, Tuple, List, Dict
from scipy.io.wavfile import write
import io
import numpy as np

from modules.utils.audio import time_stretch, pitch_shift

from modules import generate_audio
from modules.utils.normalization import text_normalize

import logging

from modules.data import styles_mgr
from modules.speaker import speaker_mgr

import random

import json

logger = logging.getLogger(__name__)


def bytes_to_array(bytes_data, sample_width):
    """
    将bytes类型的原始音频数据转换为numpy数组。

    参数:
    - bytes_data: bytes类型，原始音频数据。
    - sample_width: int类型，音频的采样宽度（比特数/8）。

    返回:
    - numpy数组，表示音频数据。
    """
    # 根据采样宽度选择正确的数据类型
    if sample_width == 1:
        # 8-bit音频
        dtype = np.int8
    elif sample_width == 2:
        # 16-bit音频
        dtype = np.int16
    else:
        # 采样宽度为3或4（24-bit或32-bit音频）
        dtype = np.int32

    # 使用numpy.frombuffer将bytes数据转换为数组
    return np.frombuffer(bytes_data, dtype=dtype)


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


def expand_spk(attrs: dict):
    input_spk = attrs.get("spk", "")
    if isinstance(input_spk, int):
        return
    if isinstance(input_spk, str) and input_spk.isdigit():
        attrs.update({"spk": int(input_spk)})
        return
    try:
        speaker = speaker_mgr.get_speaker(input_spk)
        attrs.update({"spk": speaker})
    except Exception as e:
        logger.error(f"apply style failed, {e}")


def expand_style(attrs: dict):
    if attrs.get("style", "") != "":
        try:
            params = styles_mgr.find_params_by_name(str(attrs["style"]))
            attrs.update(params)
        except Exception as e:
            logger.error(f"apply style failed, {e}")


def merge_prompt(attrs: dict, elem):

    def attr_num(attrs: Dict[str, Any], k: str, min_value: int, max_value: int):
        val = elem.get(k, attrs.get(k, ""))
        if val == "":
            return
        if val == "max":
            val = max_value
        if val == "min":
            val = min_value
        val = np.clip(int(val), min_value, max_value)
        if "prefix" not in attrs or attrs["prefix"] == None:
            attrs["prefix"] = ""
        attrs["prefix"] += " " + f"[{k}_{val}]"

    attr_num(attrs, "oral", 0, 9)
    attr_num(attrs, "speed", 0, 9)
    attr_num(attrs, "laugh", 0, 2)
    attr_num(attrs, "break", 0, 7)


def apply_random_seed(attrs: dict):
    seed = attrs.get("seed", "")
    if seed == "random" or seed == "rand":
        seed = random.randint(0, 2**32 - 1)
        attrs["seed"] = seed
        logger.info(f"random seed: {seed}")


class NotSupportSSML(Exception):
    pass


def parse_ssml(ssml: str) -> List[Dict[str, Any]]:
    root = etree.fromstring(ssml)

    ssml_version = root.get("version", "NONE")
    if ssml_version != "0.1":
        raise NotSupportSSML("Unsupported ssml version: {ssml_version}")

    segments = []

    for voice in root.findall(".//voice"):
        voice_attrs = {
            "spk": voice.get("spk"),
            "style": voice.get("style"),
            "seed": voice.get("seed"),
            "top_p": voice.get("top_p"),
            "top_k": voice.get("top_k"),
            "temp": voice.get("temp"),
            "prompt1": voice.get("prompt1"),
            "prompt2": voice.get("prompt2"),
            "prefix": voice.get("prefix"),
            "normalize": voice.get("normalize"),
        }

        voice_attrs = {k: v for k, v in voice_attrs.items() if v is not None}

        expand_spk(voice_attrs)
        expand_style(voice_attrs)

        merge_prompt(voice_attrs, voice)
        apply_random_seed(voice_attrs)

        voice_segments = []

        if voice_attrs.get("temp", "") == "min":
            # ref: https://github.com/2noise/ChatTTS/issues/123#issue-2326908144
            voice_attrs["temp"] = 0.000000000001
        if voice_attrs.get("temp", "") == "max":
            voice_attrs["temp"] = 1

        # 处理 voice 开头的文本
        if voice.text and voice.text.strip():
            voice_segments.append(
                {"text": voice.text.strip(), "attrs": voice_attrs.copy()}
            )

        # 处理 voice 内部的文本和 prosody 元素
        for node in voice.iterchildren():
            if node.tag == "prosody":
                prosody_attrs = voice_attrs.copy()
                new_attrs = {
                    "rate": node.get("rate"),
                    "volume": node.get("volume"),
                    "pitch": node.get("pitch"),
                }
                prosody_attrs.update(
                    {k: v for k, v in new_attrs.items() if v is not None}
                )
                expand_style(prosody_attrs)
                merge_prompt(prosody_attrs, node)
                apply_random_seed(voice_attrs)

                if node.text and node.text.strip():
                    voice_segments.append(
                        {"text": node.text.strip(), "attrs": prosody_attrs}
                    )
            elif node.tag == "break":
                time_ms = int(node.get("time", "0").replace("ms", ""))
                segment = {"break": time_ms}
                voice_segments.append(segment)

            if node.tail and node.tail.strip():
                voice_segments.append(
                    {"text": node.tail.strip(), "attrs": voice_attrs.copy()}
                )

        end_segment = voice_segments[-1]
        end_segment["is_end"] = True

        segments = segments + voice_segments

    logger.info(f"collect len(segments): {len(segments)}")
    # logger.info(f"segments: {json.dumps(segments, ensure_ascii=False)}")

    return segments


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


def synthesize_segments(segments: List[Dict[str, Any]]) -> List[AudioSegment]:
    audio_segments = []
    for segment in segments:
        audio_segment = synthesize_segment(segment=segment)
        audio_segments.append(audio_segment)

    return audio_segments


def combine_audio_segments(audio_segments: list) -> AudioSegment:
    combined_audio = AudioSegment.empty()
    for segment in audio_segments:
        combined_audio += segment
    return combined_audio


if __name__ == "__main__":
    # 示例 SSML 输入
    ssml1 = """
    <speak version="0.1">
        <voice spk="20398768" seed="42" temp="min" top_p="0.9" top_k="20">
            电影中梁朝伟扮演的陈永仁的
            <prosody volume="5">
                编号27149
            </prosody>
            <prosody rate="2">
                编号27149
            </prosody>
            <prosody pitch="-12">
                编号27149
            </prosody>
            <prosody pitch="12">
                编号27149
            </prosody>
        </voice>
        <voice spk="20398768" seed="42" speed="9">
            编号27149
        </voice>
        <voice spk="20398768" seed="42">
            电影中梁朝伟扮演的陈永仁的编号27149
        </voice>
    </speak>
    """

    ssml2 = """
    <speak version="0.1">
        <voice spk="Bob">
            也可以合成多角色多情感的有声 [uv_break] 书 [uv_break] ，例如：
        </voice>
        <voice spk="Bob">
            黛玉冷笑道：
        </voice>
        <voice spk="female2">
            我说呢，亏了绊住，不然，早就飞了来了。
        </voice>
        <voice spk="Bob" speed="0">
            宝玉道：
        </voice>
        <voice spk="Alice">
            “只许和你玩，替你解闷。不过偶然到他那里，就说这些闲话。”
        </voice>
        <voice spk="female2">
            “好没意思的话！去不去，关我什么事儿？又没叫你替我解闷儿，还许你不理我呢”
        </voice>
        <voice spk="Bob">
            说着，便赌气回房去了。
        </voice>
    </speak>
    """
    ssml22 = """
<speak version="0.1">
    <voice spk="Bob" style="narration-relaxed">
        下面是一个 ChatTTS 用于合成多角色多情感的有声书示例
    </voice>
    <voice spk="Bob" style="narration-relaxed">
        黛玉冷笑道：
    </voice>
    <voice spk="female2" style="angry">
        我说呢 [uv_break] ，亏了绊住，不然，早就飞起来了。
    </voice>
    <voice spk="Bob" style="narration-relaxed">
        宝玉道：
    </voice>
    <voice spk="Alice" style="unfriendly">
        “只许和你玩 [uv_break] ，替你解闷。不过偶然到他那里，就说这些闲话。”
    </voice>
    <voice spk="female2" style="angry">
        “好没意思的话！[uv_break] 去不去，关我什么事儿？ 又没叫你替我解闷儿 [uv_break]，还许你不理我呢”
    </voice>
    <voice spk="Bob" style="narration-relaxed">
        说着，便赌气回房去了。
    </voice>
</speak>
    """

    ssml3 = """
    <speak version="0.1">
        <voice spk="Bob" style="angry">
            “你到底在想什么？这已经是第三次了！每次我都告诉你要按时完成任务，可你总是拖延。你知道这对整个团队有多大的影响吗？！”
        </voice>
        <voice spk="Bob" style="assistant">
            “你到底在想什么？这已经是第三次了！每次我都告诉你要按时完成任务，可你总是拖延。你知道这对整个团队有多大的影响吗？！”
        </voice>
        <voice spk="Bob" style="gentle">
            “你到底在想什么？这已经是第三次了！每次我都告诉你要按时完成任务，可你总是拖延。你知道这对整个团队有多大的影响吗？！”
        </voice>
    </speak>
    """

    ssml4 = """
    <speak version="0.1">
        <voice spk="Bob" style="narration-relaxed">
            使用 prosody 控制生成文本的语速语调和音量，示例如下

            <prosody>
                无任何限制将会继承父级voice配置进行生成
            </prosody>
            <prosody rate="1.5">
                设置 rate 大于1表示加速，小于1为减速
            </prosody>
            <prosody pitch="6">
                设置 pitch 调整音调，设置为6表示提高6个半音
            </prosody>
            <prosody volume="2">
                设置 volume 调整音量，设置为2表示提高2个分贝
            </prosody>

            在 voice 中无prosody包裹的文本即为默认生成状态下的语音
        </voice>
    </speak>
    """

    ssml5 = """
    <speak version="0.1">
        <voice spk="Bob" style="narration-relaxed">
            使用 break 标签将会简单的
            
            <break time="500" />

            插入一段空白到生成结果中 
        </voice>
    </speak>
    """

    ssml6 = """
    <speak version="0.1">
        <voice spk="Bob" style="excited">
            temperature for sampling (may be overridden by style or speaker)
            <break time="500" />
            温度值用于采样，这个值有可能被 style 或者 speaker 覆盖 
            <break time="500" />
            temperature for sampling ，这个值有可能被 style 或者 speaker 覆盖 
            <break time="500" />
            温度值用于采样，(may be overridden by style or speaker)
        </voice>
    </speak>
    """

    segments = parse_ssml(ssml6)

    print(segments)

    audio_segments = synthesize_segments(segments)
    combined_audio = combine_audio_segments(audio_segments)

    combined_audio.export("output.wav", format="wav")
