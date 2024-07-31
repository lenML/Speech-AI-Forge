from typing import List, Optional

from fastapi import Body, File, Form, HTTPException, UploadFile
from numpy import clip
from pydantic import BaseModel, Field

from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.core.handler.datacls.audio_model import (
    AdjustConfig,
    AudioFormat,
    EncoderConfig,
)
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.handler.TTSHandler import TTSHandler
from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.data import styles_mgr


class AudioSpeechRequest(BaseModel):
    input: str  # 需要合成的文本
    model: str = "chat-tts"
    voice: str = "female2"
    response_format: AudioFormat = "mp3"
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


async def openai_speech_api(
    request: AudioSpeechRequest = Body(
        ..., description="JSON body with model, input text, and voice"
    )
):
    model = request.model
    input_text = request.input
    voice = request.voice
    style = request.style
    eos = request.eos
    seed = request.seed
    stream = request.stream

    response_format = request.response_format
    if not isinstance(response_format, AudioFormat) and isinstance(
        response_format, str
    ):
        response_format = AudioFormat(response_format)

    batch_size = request.batch_size
    spliter_threshold = request.spliter_threshold
    speed = request.speed
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

    ctx_params = api_utils.calc_spk_style(spk=voice, style=style)

    speaker = ctx_params.get("spk")
    if not isinstance(speaker, TTSSpeaker):
        raise HTTPException(status_code=400, detail="Invalid voice.")

    tts_config = TTSConfig(
        style=style,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
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
        enabled=request.enhance or request.denoise or False,
        lambd=0.9 if request.denoise else 0.1,
    )
    encoder_config = EncoderConfig(
        format=response_format,
        bitrate="64k",
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
        )

        return handler.enqueue_to_response(request=request)

    except Exception as e:
        import logging

        logging.exception(e)

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
    )(openai_speech_api)

    @app.post(
        "/v1/audio/transcriptions",
        response_model=TranscriptionsVerboseResponse,
        description="Transcribes audio into the input language.",
    )
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0),
        timestamp_granularities: List[str] = Form(["segment"]),
    ):
        # TODO: Implement transcribe
        return api_utils.success_response("not implemented yet")
