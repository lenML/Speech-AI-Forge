import io
from typing import Optional

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
from pydub import AudioSegment

from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.core.handler.datacls.stt_model import STTConfig, STTOutputFormat
from modules.core.handler.STTHandler import STTHandler


import base64
import aiohttp
from pydantic import (
    BaseModel,
    Field,
    conint,
    confloat,
    constr,
    HttpUrl,
    model_validator,
)
from typing import Optional, Dict
from modules.core.handler.datacls.stt_model import STTOutputFormat
from fastapi import Body

from modules.core.models.stt.STTModel import TranscribeResult


class TranscriptionsResponse(BaseModel):
    message: str
    data: TranscribeResult


class FromUrl(BaseModel):
    url: HttpUrl = Field(..., description="音频文件 URL，必须是合法的 http(s) 地址")
    headers: Optional[Dict[str, str]] = Field(
        None, description="请求 URL 时附带的自定义 header"
    )


class InputAudio(BaseModel):
    from_url: Optional[FromUrl] = Field(None, description="从 URL 加载音频")
    from_base64: Optional[str] = Field(None, description="base64 编码的音频数据")

    @model_validator(mode="after")
    def check_exclusive(self):
        if not self.from_url and not self.from_base64:
            raise ValueError("from_url 或 from_base64 必须提供其一")
        if self.from_url and self.from_base64:
            raise ValueError("from_url 与 from_base64 不可同时提供")
        return self


class TranscriptionRequest(BaseModel):
    input_audio: InputAudio

    model: str = Field(
        "whisper.large", description="ASR 模型名称，如 whisper.base、whisper.large"
    )

    refrence_transcript: Optional[str] = Field(None, description="参考文案")
    prompt: Optional[str] = Field(None, description="提示词，用于引导模型")
    prefix: Optional[str] = Field(None, description="对话历史或固定开头")

    language: Optional[str] = Field(None, description="语言代码，如 'en', 'zh' 等")

    temperature: Optional[confloat(ge=0.0, le=1.0)] = Field(
        None, description="采样温度，控制多样性，范围 0~1"
    )
    sample_len: Optional[conint(ge=1)] = Field(None, description="采样长度，必须 >= 1")
    best_of: Optional[conint(ge=1)] = Field(
        None, description="在 temperature > 0 时采样 n 次取最佳结果"
    )
    beam_size: Optional[conint(ge=1)] = Field(
        None, description="beam search 的宽度，推荐 5~10"
    )
    patience: Optional[confloat(ge=0.0)] = Field(
        None, description="beam search 的 patience 参数，越大越宽松"
    )
    length_penalty: Optional[confloat()] = Field(
        None, description="对生成文本长度的惩罚，负值鼓励短输出"
    )

    format: STTOutputFormat = Field(
        STTOutputFormat.txt, description="输出格式，如 txt, json, srt, vtt"
    )
    highlight_words: Optional[bool] = Field(
        False, description="是否高亮每个识别的单词（如 JSON 格式）"
    )
    max_line_count: Optional[conint(gt=0)] = Field(
        None, description="最大行数限制（用于格式化输出）"
    )
    max_line_width: Optional[conint(gt=0)] = Field(
        None, description="最大每行字符数（用于 txt 格式）"
    )
    max_words_per_line: Optional[conint(gt=0)] = Field(
        None, description="每行最多几个词（用于格式化输出）"
    )


async def load_audio_bytes(input_audio: InputAudio) -> bytes:
    if input_audio.from_base64:
        try:
            return base64.b64decode(input_audio.from_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 audio.")
    elif input_audio.from_url:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    input_audio.from_url.url, headers=input_audio.from_url.headers
                ) as resp:
                    if resp.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to fetch audio: {resp.status}",
                        )
                    return await resp.read()
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to fetch audio from URL: {str(e)}"
            )
    else:
        raise HTTPException(status_code=400, detail="No valid input_audio provided.")


def setup(app: APIManager):

    def pydub_to_numpy(audio_segment: AudioSegment) -> np.ndarray:
        raw_data = audio_segment.raw_data
        sample_width = audio_segment.sample_width
        channels = audio_segment.channels
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        if channels > 1:
            audio_data = audio_data.reshape((-1, channels))
            audio_data = audio_data.mean(axis=1).astype(np.int16)
        return audio_data

    @app.post(
        "/v2/stt",
        response_model=TranscriptionsResponse,
        tags=["Forge V2"],
        description="Transcribes audio using base64 or URL input. Stateless interface.",
    )
    async def transcribe_v2(payload: TranscriptionRequest = Body(...)):
        # TODO: 缺少输入参数校验，比如应该校验模型是否可用参数配置是否正确等等
        try:
            audio_bytes = await load_audio_bytes(payload.input_audio)
            audio_segment: AudioSegment = AudioSegment.from_file(
                io.BytesIO(audio_bytes)
            )
            sample_rate: int = audio_segment.frame_rate
            samples = pydub_to_numpy(audio_segment=audio_segment)
            input_audio = (sample_rate, samples)

            sst_config = STTConfig(
                mid=payload.model,
                refrence_transcript=payload.refrence_transcript,
                prompt=payload.prompt,
                language=payload.language,
                temperature=payload.temperature,
                sample_len=payload.sample_len,
                best_of=payload.best_of,
                beam_size=payload.beam_size,
                patience=payload.patience,
                length_penalty=payload.length_penalty,
                format=payload.format,
                highlight_words=payload.highlight_words,
                max_line_count=payload.max_line_count,
                max_line_width=payload.max_line_width,
                max_words_per_line=payload.max_words_per_line,
            )

            handler = STTHandler(input_audio=input_audio, stt_config=sst_config)
            result = await handler.enqueue()
            return api_utils.success_response(result.__dict__)
        except Exception as e:
            import logging

            logging.exception(e)
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=str(e))
