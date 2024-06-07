from pydantic import BaseModel
from typing import Any, Union

import torch

from modules.speaker import Speaker, speaker_mgr


from modules.data import styles_mgr

from pydub import AudioSegment

from modules.ssml import merge_prompt


from enum import Enum


class ParamsTypeError(Exception):
    pass


class AudioFormat(str, Enum):
    mp3 = "mp3"
    wav = "wav"


class BaseResponse(BaseModel):
    message: str
    data: Any


def success_response(data: Any, message: str = "ok") -> BaseResponse:
    return BaseResponse(message=message, data=data)


def wav_to_mp3(wav_data, bitrate="48k"):
    audio = AudioSegment.from_wav(
        wav_data,
    )
    return audio.export(format="mp3", bitrate=bitrate)


def to_number(value, t, default=0):
    try:
        number = t(value)
        return number
    except (ValueError, TypeError) as e:
        return default


def calc_spk_style(spk: Union[str, int], style: Union[str, int]):
    voice_attrs = {
        "spk": None,
        "prompt1": None,
        "prompt2": None,
        "prefix": None,
        "temperature": None,
    }
    params = {}

    if type(spk) == int:
        voice_attrs["spk"] = spk
    elif type(spk) == str:
        if spk.isdigit():
            voice_attrs["spk"] = int(spk)
        else:
            spker = speaker_mgr.get_speaker(spk)
            if spker:
                voice_attrs["spk"] = spker

    if type(style) == int or type(style) == float:
        raise ParamsTypeError("The style parameter cannot be a number.")
    elif type(style) == str and style != "":
        if style.isdigit():
            raise ParamsTypeError("The style parameter cannot be a number.")
        else:
            style_params = styles_mgr.find_params_by_name(style)
            for k, v in style_params.items():
                params[k] = v

    voice_attrs = {k: v for k, v in voice_attrs.items() if v is not None}

    merge_prompt(voice_attrs, params)

    voice_attrs["spk"] = params.get("spk", voice_attrs.get("spk", None))
    voice_attrs["temperature"] = params.get(
        "temp", voice_attrs.get("temperature", None)
    )
    voice_attrs["prefix"] = params.get("prefix", voice_attrs.get("prefix", None))
    voice_attrs["prompt1"] = params.get("prompt1", voice_attrs.get("prompt1", None))
    voice_attrs["prompt2"] = params.get("prompt2", voice_attrs.get("prompt2", None))

    if voice_attrs.get("temperature", "") == "min":
        # ref: https://github.com/2noise/ChatTTS/issues/123#issue-2326908144
        voice_attrs["temperature"] = 0.000000000001
    if voice_attrs.get("temperature", "") == "max":
        voice_attrs["temperature"] = 1

    voice_attrs = {k: v for k, v in voice_attrs.items() if v is not None}
    # print(voice_attrs)

    return voice_attrs
