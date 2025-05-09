import logging
import re
from typing import Optional, Union

import gradio as gr
import numpy as np
import torch
import torch.profiler

from modules import refiner
from modules.api.utils import calc_spk_style
from modules.core.handler.datacls.audio_model import (
    AdjustConfig,
    AudioFormat,
    EncoderConfig,
)
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tn_model import TNConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.handler.TTSHandler import TTSHandler
from modules.core.models.tts import ChatTtsModel
from modules.core.spk import TTSSpeaker, spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.core.ssml.SSMLParser import SSMLBreak, SSMLSegment, create_ssml_v01_parser
from modules.core.tn import ChatTtsTN
from modules.core.tools.SentenceSplitter import SentenceSplitter
from modules.data import styles_mgr
from modules.utils import audio_utils
from modules.utils.hf import spaces
from modules.webui import webui_config
from modules.webui.speaker.wav_misc import encode_to_wav

logger = logging.getLogger(__name__)

SPK_FILE_EXTS = [
    # ".spkv1.json",
    # ".spkv1.png",
    ".json",
    ".png",
]


def get_speakers(filter: Optional[callable] = None) -> list[TTSSpeaker]:
    spks = spk_mgr.list_speakers()
    if filter is not None:
        spks = [spk for spk in spks if filter(spk)]

    return spks


def get_speaker_names(
    filter: Optional[callable] = None,
) -> tuple[list[TTSSpeaker], list[str]]:
    speakers = get_speakers(filter)

    def get_speaker_show_name(spk: TTSSpeaker):
        if spk.gender == "*" or spk.gender == "":
            return spk.name
        return f"{spk.gender} : {spk.name}"

    speaker_names = [get_speaker_show_name(speaker) for speaker in speakers]
    speaker_names.sort(key=lambda x: x.startswith("*") and "-1" or x)

    return speakers, speaker_names


def get_spk_emotions(file):
    if file is None:
        return ["default"]
    try:
        spk: TTSSpeaker = TTSSpeaker.from_file(file)
        return spk.get_emotions()
    except Exception as e:
        logger.error(f"load spk emotions failed: {e}")
        return ["default"]


def get_spk_emotions_from_name(spk_name: str) -> list[str]:
    spk = spk_mgr.get_speaker(spk_name)
    if spk is not None:
        return spk.get_emotions()
    return ["default"]


def get_styles():
    return styles_mgr.list_items()


def load_spk_info(file):
    if file is None:
        return "empty"
    try:

        spk: TTSSpeaker = TTSSpeaker.from_file(file)
        return f"""
- name: {spk.name}
- gender: {spk.gender}
- describe: {spk.desc}
    """.strip()
    except Exception as e:
        logger.error(f"load spk info failed: {e}")
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


@spaces.GPU(duration=120)
async def synthesize_ssml(
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
    model_id: str = "chat-tts",
    remove_silence: bool = False,
    remove_silence_threshold=-42,
    progress=gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
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
        remove_silence=remove_silence,
        remove_silence_threshold=remove_silence_threshold,
    )
    enhancer_config = EnhancerConfig(
        enabled=enable_denoise or enable_enhance or False,
        lambd=0.9 if enable_denoise else 0.1,
    )
    encoder_config = EncoderConfig(
        format=AudioFormat.mp3,
        bitrate="64k",
    )
    tts_config = TTSConfig(mid=model_id)

    handler = TTSHandler(
        ssml_content=ssml,
        tts_config=tts_config,
        infer_config=infer_config,
        adjust_config=adjust_config,
        enhancer_config=enhancer_config,
        encoder_config=encoder_config,
    )

    sample_rate, audio_data = await handler.enqueue()

    # NOTE: 这里必须要加，不然 gradio 没法解析成 mp3 格式
    audio_data = audio_utils.audio_to_int16(audio_data)

    return sample_rate, audio_data


async def run_tts_pipe(
    text: str,
    spk: Optional[TTSSpeaker],
    tts_config: TTSConfig,
    infer_config: InferConfig,
    adjust_config: AdjustConfig,
    enhancer_config: EnhancerConfig,
    encoder_config: EncoderConfig,
):

    handler = TTSHandler(
        text_content=text,
        spk=spk,
        tts_config=tts_config,
        infer_config=infer_config,
        adjust_config=adjust_config,
        enhancer_config=enhancer_config,
        encoder_config=encoder_config,
        vc_config=VCConfig(enabled=False),
    )

    sample_rate, audio_data = await handler.enqueue()

    # NOTE: 这里必须要加，不然 gradio 没法解析成 mp3 格式
    audio_data = audio_utils.audio_to_int16(audio_data)
    return sample_rate, audio_data


def is_number_str(s: str):
    if not isinstance(s, str):
        return False
    # 使用正则表达式匹配数字字符串，包括负数和浮点数
    return bool(re.match(r"^[-+]?\d*\.?\d+$", s))


# @torch.inference_mode()
# NOTE: async funciton 没法用 inference_mode
@spaces.GPU(duration=120)
async def tts_generate(
    text: str,
    temperature=0.3,
    top_p=0.7,
    top_k=20,
    spk=-1,
    infer_seed=-1,
    prompt1="",
    prompt2="",
    prefix="",
    style="",
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
    remove_silence=False,
    remove_silence_threshold=-42,
    ref_audio: Optional[tuple[int, np.ndarray]] = None,
    ref_audio_text: Optional[str] = None,
    # 这个是非上传音色的 emotion
    spk_emotion1="default",
    # 这个是上传音色的 emotion
    spk_emotion2="default",
    model_id: str = "chat-tts",
    progress=gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
):
    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = 4

    spk_emotion = spk_emotion1

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
    prefix = prefix or params.get("prefix", "")
    prompt = params.get("prompt", "")
    prompt1 = prompt1 or params.get("prompt1", "")
    prompt2 = prompt2 or params.get("prompt2", "")

    infer_seed = np.clip(infer_seed, -1, 2**32 - 1, out=None, dtype=np.float64)
    infer_seed = int(infer_seed)

    # ref: https://github.com/2noise/ChatTTS/issues/123#issue-2326908144
    min_n = 0.000000001
    if temperature == 0.1:
        temperature = min_n

    if isinstance(spk, str) and is_number_str(spk):
        spk = int(spk)

    if isinstance(spk, int):
        if model_id != "chat-tts":
            # raise gr.Error("Only ChatTTS model support create speaker from seed")

            # NOTE: 创建一个空的说话人，表示随机，也有的模型可能不支持空说话人，那就会在推理的时候报错
            spk = TTSSpeaker.empty()
        else:
            # NOTE: 只有 ChatTTS 模型支持从 seed 创建 speaker
            spk = ChatTtsModel.ChatTTSModel.create_speaker_from_seed(spk)

    if spk_file:
        try:
            spk: TTSSpeaker = TTSSpeaker.from_file(spk_file)
            # 如果读取文件，那么就使用 emotion2 这个是 UI 顺序决定的
            spk_emotion = spk_emotion2
        except Exception:
            raise gr.Error("Failed to load speaker file")

    if ref_audio is not None:
        if ref_audio_text is None or ref_audio_text.strip() == "":
            raise gr.Error("ref_audio_text is empty")
        ref_audio_bytes = encode_to_wav(audio_tuple=ref_audio)
        spk = TTSSpeaker.from_ref_wav_bytes(
            ref_wav=ref_audio_bytes,
            text=ref_audio_text,
        )

    tts_config = TTSConfig(
        mid=model_id,
        style=style,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        prompt=prompt,
        prefix=prefix,
        prompt1=prompt1,
        prompt2=prompt2,
        emotion=spk_emotion,
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
        remove_silence=remove_silence,
        remove_silence_threshold=remove_silence_threshold,
    )
    enhancer_config = EnhancerConfig(
        enabled=enable_denoise or enable_enhance or False,
        lambd=0.9 if enable_denoise else 0.1,
    )
    # NOTE: 这里只是占位，其实用不到，因为webui音频编码是走的gradio逻辑，我们只生成ndarray，不会调用 encoder 逻辑
    encoder_config = EncoderConfig()

    return await run_tts_pipe(
        text=text,
        spk=spk,
        tts_config=tts_config,
        infer_config=infer_config,
        adjust_config=adjust_config,
        enhancer_config=enhancer_config,
        encoder_config=encoder_config,
    )


# @torch.inference_mode()
@spaces.GPU(duration=120)
def dit_tts_generate(
    text:str,
    spk:Optional[TTSSpeaker] = None,
    infer_seed=-1,
    disable_normalize=False,
    batch_size=4,
    # dit configs
    nfe_step = 32,
    cfg_strength = 2.0,
    sway_sampling_coef = -1.0,
    speed_scale = 1.0,
    # enhancer
    enable_enhance=False,
    enable_denoise=False,
    # spliter
    spliter_thr: int = 100,
    eos: str = "[uv_break]",
    # adjuster
    pitch: float = 0,
    speed_rate: float = 1,
    volume_gain_db: float = 0,
    normalize: bool = True,
    headroom: float = 1,
    # refrerence audio: 这个会覆盖 spk
    ref_audio: Optional[tuple[int, np.ndarray]] = None,
    ref_audio_text: Optional[str] = None,
    # dit 模型
    model_id: str = "f5-tts",
    progress=gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
):
    """
    dit 模型的调用函数

    TODO: 传递dit模型参数
    """
    pass


@torch.inference_mode()
def text_normalize(text: str) -> str:
    return ChatTtsTN.ChatTtsTN.normalize(
        text, config=TNConfig(disabled=["replace_unk_tokens"])
    )


@spaces.GPU(duration=120)
def refine_text(
    text: str,
    oral: int = -1,
    speed: int = -1,
    rf_break: int = -1,
    laugh: int = -1,
    # TODO 这个还没ui
    spliter_threshold: int = 300,
    progress=gr.Progress(track_tqdm=not webui_config.off_track_tqdm),
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


@spaces.GPU(duration=120)
def split_long_text(long_text_input, spliter_threshold=100, eos=""):
    # TODO 传入 tokenizer
    spliter = SentenceSplitter(threshold=spliter_threshold)
    sentences = spliter.parse(long_text_input)
    sentences = [text_normalize(s) + eos for s in sentences]
    data = []
    for i, text in enumerate(sentences):
        token_length = spliter.len(text)
        data.append([i, text, token_length])
    return data
