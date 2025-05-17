import io
from typing import List, Optional

import numpy as np
from fastapi import Body, File, Form, HTTPException, Request, UploadFile
from numpy import clip
from pydantic import BaseModel, Field
from pydub import AudioSegment

from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.api.constants import support_bitrates
from modules.core.handler.datacls.audio_model import (
    AdjustConfig,
    AudioFormat,
    EncoderConfig,
)
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.stt_model import STTConfig, STTOutputFormat
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.handler.STTHandler import STTHandler
from modules.core.handler.TTSHandler import TTSHandler
from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.data import styles_mgr


class AudioSpeechParams(BaseModel):
    input: str  # 需要合成的文本
    model: str = "chat-tts"
    voice: str = "female2"
    response_format: AudioFormat = "raw"
    speed: float = Field(1, ge=0.1, le=10, description="Speed of the audio")
    seed: int = 42

    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.7

    style: str = ""
    batch_size: int = Field(1, ge=1, le=20, description="Batch size")
    spliter_threshold: float = Field(
        100, ge=10, le=1024, description="Threshold for sentence spliter"
    )
    # end of sentence
    eos: str = "[uv_break]"

    enhance: bool = False
    denoise: bool = False

    stream: bool = False

    bitrate: str = "64k"


async def openai_speech_api(
    request: Request,
    params: AudioSpeechParams = Body(
        ..., description="JSON body with model, input text, and voice"
    ),
):
    model = params.model
    input_text = params.input
    voice = params.voice
    style = params.style
    eos = params.eos
    seed = params.seed
    stream = params.stream
    audio_bitrate = params.bitrate

    response_format = params.response_format
    if not isinstance(response_format, AudioFormat) and isinstance(
        response_format, str
    ):
        response_format = AudioFormat(response_format)

    batch_size = params.batch_size
    spliter_threshold = params.spliter_threshold
    speed = params.speed
    speed = clip(speed, 0.1, 10)

    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is required.")
    if spk_mgr.get_speaker(voice) is None:
        raise HTTPException(status_code=400, detail="Invalid voice.")
    try:
        if style:
            styles_mgr.find_item_by_name(style)
    except:
        raise HTTPException(status_code=400, detail="Invalid style.")

    if audio_bitrate not in support_bitrates:
        raise HTTPException(
            status_code=422,
            detail=f"The specified bitrate is not supported. support bitrates: {str(support_bitrates)}",
        )

    ctx_params = api_utils.calc_spk_style(spk=voice, style=style)

    speaker = ctx_params.get("spk")
    if not isinstance(speaker, TTSSpeaker):
        raise HTTPException(status_code=400, detail="Invalid voice.")

    tts_config = TTSConfig(
        style=style,
        temperature=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        mid=model,
    )
    infer_config = InferConfig(
        batch_size=batch_size,
        spliter_threshold=spliter_threshold,
        eos=eos,
        seed=seed,
        stream=stream,
    )
    adjust_config = AdjustConfig(speaking_rate=speed)
    enhancer_config = EnhancerConfig(
        enabled=params.enhance or params.denoise or False,
        lambd=0.9 if params.denoise else 0.1,
    )
    encoder_config = EncoderConfig(
        format=response_format,
        bitrate=audio_bitrate,
    )
    try:
        handler = TTSHandler(
            text_content=input_text,
            spk=speaker,
            tts_config=tts_config,
            infer_config=infer_config,
            adjust_config=adjust_config,
            enhancer_config=enhancer_config,
            encoder_config=encoder_config,
            vc_config=VCConfig(enabled=False),
        )

        handler.set_current_request(request=request)
        return await handler.enqueue_to_response()

    except Exception as e:
        import logging

        logging.exception(e)
        handler.interrupt()

        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=str(e))


class TranscribeSegment(BaseModel):
    id: int
    seek: float
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class TranscriptionsVerboseResponse(BaseModel):
    task: str
    language: str
    duration: float
    text: str
    segments: list[TranscribeSegment]


class TranscriptionsResponse(BaseModel):
    text: str


def setup(app: APIManager):
    app.post(
        "/v1/audio/speech",
        description="""
openai api document: 
[https://platform.openai.com/docs/guides/text-to-speech](https://platform.openai.com/docs/guides/text-to-speech)

以下属性为本系统自定义属性，不在openai文档中：
- batch_size: 是否开启batch合成，小于等于1表示不使用batch （不推荐）
- spliter_threshold: 开启batch合成时，句子分割的阈值
- style: 风格

> model 可填任意值
        """,
        tags=["OpenAI API"],
    )(openai_speech_api)

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
        "/v1/audio/transcriptions",
        # NOTE: 其实最好是不设置这个model...因为这个接口可以返回很多情况...
        # response_model=TranscriptionsResponse,
        description="Transcribes audio into the input language.",
        tags=["OpenAI API"],
    )
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form("whisper.large"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        # TODO 不支持 verbose_json
        response_format: str = Form("txt"),
        temperature: float = Form(0),
        # TODO 这个没实现，需要重写 whisper 的 transcribe 函数
        timestamp_granularities: List[str] = Form(["segment"]),
    ):
        try:
            response_format = STTOutputFormat(response_format)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid response format.")

        audio_bytes = await file.read()
        audio_segment: AudioSegment = AudioSegment.from_file(io.BytesIO(audio_bytes))

        sample_rate = audio_segment.frame_rate
        samples = pydub_to_numpy(audio_segment=audio_segment)

        input_audio = (sample_rate, samples)

        sst_config = STTConfig(
            mid=model,
            prompt=prompt,
            language=language,
            temperature=temperature if temperature > 0 else None,
            format=response_format,
        )

        try:
            handler = STTHandler(input_audio=input_audio, stt_config=sst_config)

            result = await handler.enqueue()
            return {"text": result.text}
        except Exception as e:
            import logging

            logging.exception(e)
            # TODO: STT 也应该支持 interupt
            # handler.interrupt()

            if isinstance(e, HTTPException):
                raise e
            else:
                raise HTTPException(status_code=500, detail=str(e))
