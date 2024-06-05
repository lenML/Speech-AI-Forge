from fastapi import HTTPException, Body
from fastapi.responses import StreamingResponse

import io
from numpy import clip
import soundfile as sf
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse


from modules.synthesize_audio import synthesize_audio
from modules.normalization import text_normalize

from modules import generate_audio as generate


from typing import Literal
import pyrubberband as pyrb

from modules.api import utils as api_utils
from modules.api.Api import APIManager

import numpy as np


class AudioSpeechRequest(BaseModel):
    input: str  # 需要合成的文本
    model: str = "chattts-4w"
    voice: str = "female2"
    response_format: Literal["mp3", "wav"] = "mp3"
    speed: int = Field(1, ge=1, le=10, description="Speed of the audio")
    style: str = ""
    # 是否开启batch合成，小于等于1表示不适用batch
    # 开启batch合成会自动分割句子
    batch_size: int = Field(1, ge=1, le=10, description="Batch size")
    spliter_threshold: float = Field(
        100, ge=10, le=1024, description="Threshold for sentence spliter"
    )


async def openai_speech_api(
    request: AudioSpeechRequest = Body(
        ..., description="JSON body with model, input text, and voice"
    )
):
    try:
        model = request.model
        input_text = request.input
        voice = request.voice
        style = request.style
        response_format = request.response_format
        batch_size = request.batch_size
        spliter_threshold = request.spliter_threshold
        speed = request.speed
        speed = clip(speed, 0.1, 10)

        if not input_text:
            raise HTTPException(status_code=400, detail="Input text is required.")

        # Normalize the text
        text = text_normalize(input_text, is_end=True)

        # Calculate speaker and style based on input voice
        params = api_utils.calc_spk_style(spk=voice, style=style)

        spk = params.get("spk", -1)
        seed = params.get("seed", 42)
        temperature = params.get("temperature", 0.3)
        prompt1 = params.get("prompt1", "")
        prompt2 = params.get("prompt2", "")
        prefix = params.get("prefix", "")

        # Generate audio
        sample_rate, audio_data = synthesize_audio(
            text,
            temperature=temperature,
            top_P=0.7,
            top_K=20,
            spk=spk,
            infer_seed=seed,
            batch_size=batch_size,
            spliter_threshold=spliter_threshold,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
        )

        if speed != 1:
            audio_data = pyrb.time_stretch(audio_data, sample_rate, speed)

        # Convert audio data to wav format
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="wav")
        buffer.seek(0)

        if response_format == "mp3":
            # Convert wav to mp3
            buffer = api_utils.wav_to_mp3(buffer)

        return StreamingResponse(buffer, media_type="audio/mp3")

    except Exception as e:
        import logging

        logging.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


def setup(api_manager: APIManager):
    api_manager.post("/v1/openai/audio/speech", response_class=FileResponse)(
        openai_speech_api
    )
