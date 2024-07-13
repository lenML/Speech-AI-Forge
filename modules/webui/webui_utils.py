from typing import Union

import gradio as gr
import numpy as np
import torch
import torch.profiler

from modules import refiner
from modules.api.utils import calc_spk_style
from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.chattts_model import ChatTTSConfig, InferConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.SSMLHandler import SSMLHandler
from modules.core.handler.TTSHandler import TTSHandler
from modules.core.speaker import Speaker, speaker_mgr
from modules.core.ssml.SSMLParser import SSMLBreak, SSMLSegment, create_ssml_v01_parser
from modules.core.tn import ChatTtsTN
from modules.core.tools.SentenceSplitter import SentenceSplitter
from modules.data import styles_mgr
from modules.Enhancer.ResembleEnhance import apply_audio_enhance as _apply_audio_enhance
from modules.utils import audio_utils
from modules.utils.hf import spaces
from modules.webui import webui_config


def get_speakers():
    return speaker_mgr.list_speakers()


def get_speaker_names() -> tuple[list[Speaker], list[str]]:
    speakers = get_speakers()

    def get_speaker_show_name(spk):
        if spk.gender == "*" or spk.gender == "":
            return spk.name
        return f"{spk.gender} : {spk.name}"

    speaker_names = [get_speaker_show_name(speaker) for speaker in speakers]
    speaker_names.sort(key=lambda x: x.startswith("*") and "-1" or x)

    return speakers, speaker_names


def get_styles():
    return styles_mgr.list_items()


def load_spk_info(file):
    if file is None:
        return "empty"
    try:

        spk: Speaker = Speaker.from_file(file)
        infos = spk.to_json()
        return f"""
- name: {infos.name}
- gender: {infos.gender}
- describe: {infos.describe}
    """.strip()
    except:
        return "load failed"


def segments_length_limit(
    segments: list[Union[SSMLBreak, SSMLSegment]], total_max: int
) -> list[Union[SSMLBreak, SSMLSegment]]:
    ret_segments = []
    total_len = 0
    for seg in segments:
        if isinstance(seg, SSMLBreak):
            ret_segments.append(seg)
            continue
        total_len += len(seg["text"])
        if total_len > total_max:
            break
        ret_segments.append(seg)
    return ret_segments


@torch.inference_mode()
@spaces.GPU(duration=120)
def apply_audio_enhance(audio_data, sr, enable_denoise, enable_enhance):
    return _apply_audio_enhance(audio_data, sr, enable_denoise, enable_enhance)


@torch.inference_mode()
@spaces.GPU(duration=120)
def synthesize_ssml(
    ssml: str,
    batch_size=4,
    enable_enhance=False,
    enable_denoise=False,
    eos: str = "[uv_break]",
    spliter_thr: int = 100,
    pitch: float = 0,
    speed_rate: float = 1,
    volume_gain_db: float = 0,
    normalize: bool = True,
    headroom: float = 1,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = 8

    ssml = ssml.strip()

    if ssml == "":
        raise gr.Error("SSML is empty, please input some SSML")

    parser = create_ssml_v01_parser()
    segments = parser.parse(ssml)
    max_len = webui_config.ssml_max
    segments = segments_length_limit(segments, max_len)

    if len(segments) == 0:
        raise gr.Error("No valid segments in SSML")

    infer_config = InferConfig(
        batch_size=batch_size,
        spliter_threshold=spliter_thr,
        eos=eos,
        # NOTE: SSML not support `infer_seed` contorl
        # seed=42,
        # NOTE: 开启以支持 track_tqdm
        sync_gen=True,
    )
    adjust_config = AdjustConfig(
        pitch=pitch,
        speed_rate=speed_rate,
        volume_gain_db=volume_gain_db,
        normalize=normalize,
        headroom=headroom,
    )
    enhancer_config = EnhancerConfig(
        enabled=enable_denoise or enable_enhance or False,
        lambd=0.9 if enable_denoise else 0.1,
    )

    handler = SSMLHandler(
        ssml_content=ssml,
        infer_config=infer_config,
        adjust_config=adjust_config,
        enhancer_config=enhancer_config,
    )

    sample_rate, audio_data = handler.enqueue()

    # NOTE: 这里必须要加，不然 gradio 没法解析成 mp3 格式
    audio_data = audio_utils.audio_to_int16(audio_data)

    return sample_rate, audio_data


# @torch.inference_mode()
@spaces.GPU(duration=120)
def tts_generate(
    text,
    temperature=0.3,
    top_p=0.7,
    top_k=20,
    spk=-1,
    infer_seed=-1,
    use_decoder=True,
    prompt1="",
    prompt2="",
    prefix="",
    style="",
    disable_normalize=False,
    batch_size=4,
    enable_enhance=False,
    enable_denoise=False,
    spk_file=None,
    spliter_thr: int = 100,
    eos: str = "[uv_break]",
    pitch: float = 0,
    speed_rate: float = 1,
    volume_gain_db: float = 0,
    normalize: bool = True,
    headroom: float = 1,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = 4

    max_len = webui_config.tts_max
    text = text.strip()[0:max_len]

    if text == "":
        raise gr.Error("Text is empty, please input some text")

    if style == "*auto":
        style = ""

    if isinstance(top_k, float):
        top_k = int(top_k)

    params = calc_spk_style(spk=spk, style=style)
    spk = params.get("spk", spk)

    infer_seed = infer_seed or params.get("seed", infer_seed)
    temperature = temperature or params.get("temperature", temperature)
    prefix = prefix or params.get("prefix", prefix)
    prompt1 = prompt1 or params.get("prompt1", "")
    prompt2 = prompt2 or params.get("prompt2", "")

    infer_seed = np.clip(infer_seed, -1, 2**32 - 1, out=None, dtype=np.float64)
    infer_seed = int(infer_seed)

    if isinstance(spk, int):
        spk = Speaker.from_seed(spk)

    if spk_file:
        try:
            spk: Speaker = Speaker.from_file(spk_file)
        except Exception:
            raise gr.Error("Failed to load speaker file")

        if not isinstance(spk.emb, torch.Tensor):
            raise gr.Error("Speaker file is not supported")

    tts_config = ChatTTSConfig(
        style=style,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        prefix=prefix,
        prompt1=prompt1,
        prompt2=prompt2,
    )
    infer_config = InferConfig(
        batch_size=batch_size,
        spliter_threshold=spliter_thr,
        eos=eos,
        seed=infer_seed,
        # NOTE: 开启以支持 track_tqdm
        sync_gen=True,
    )
    adjust_config = AdjustConfig(
        pitch=pitch,
        speed_rate=speed_rate,
        volume_gain_db=volume_gain_db,
        normalize=normalize,
        headroom=headroom,
    )
    enhancer_config = EnhancerConfig(
        enabled=enable_denoise or enable_enhance or False,
        lambd=0.9 if enable_denoise else 0.1,
    )

    handler = TTSHandler(
        text_content=text,
        spk=spk,
        tts_config=tts_config,
        infer_config=infer_config,
        adjust_config=adjust_config,
        enhancer_config=enhancer_config,
    )

    sample_rate, audio_data = handler.enqueue()

    # NOTE: 这里必须要加，不然 gradio 没法解析成 mp3 格式
    audio_data = audio_utils.audio_to_int16(audio_data)
    return sample_rate, audio_data


def text_normalize(text: str) -> str:
    return ChatTtsTN.ChatTtsTN.normalize(text)


@torch.inference_mode()
@spaces.GPU(duration=120)
def refine_text(
    text: str,
    oral: int = -1,
    speed: int = -1,
    rf_break: int = -1,
    laugh: int = -1,
    # TODO 这个还没ui
    spliter_threshold: int = 300,
    progress=gr.Progress(track_tqdm=True),
):
    text = text_normalize(text)
    prompt = []
    if oral != -1:
        prompt.append(f"[oral_{oral}]")
    if speed != -1:
        prompt.append(f"[speed_{speed}]")
    if rf_break != -1:
        prompt.append(f"[break_{rf_break}]")
    if laugh != -1:
        prompt.append(f"[laugh_{laugh}]")
    return refiner.refine_text(
        text, prompt="".join(prompt), spliter_threshold=spliter_threshold
    )


@torch.inference_mode()
@spaces.GPU(duration=120)
def split_long_text(long_text_input, spliter_threshold=100, eos=""):
    spliter = SentenceSplitter(threshold=spliter_threshold)
    sentences = spliter.parse(long_text_input)
    sentences = [text_normalize(s) + eos for s in sentences]
    data = []
    for i, text in enumerate(sentences):
        token_length = spliter.len(text)
        data.append([i, text, token_length])
    return data
