import io
from typing import Optional

import numpy as np
from fastapi import Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment

from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.core.handler.datacls.audio_model import AudioFormat, EncoderConfig
from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.handler.VCHandler import VCHandler
from modules.core.spk.TTSSpeaker import TTSSpeaker


class VoiceCloneForm(BaseModel):
    src_audio: UploadFile

    ref_audio: Optional[UploadFile]

    ref_spk: Optional[str]
    spk_emotion: Optional[str]

    model: str = "open-voice"

    tau: float = 0.3

    format: str = "mp3"

    @classmethod
    def as_form(
        cls,
        src_audio: UploadFile,
        ref_audio: Optional[UploadFile] = None,
        ref_spk: Optional[str] = None,
        spk_emotion: Optional[str] = None,
        model: str = "open-voice",
        tau: float = 0.3,
        format: str = "mp3",
    ):
        return cls(
            src_audio=src_audio,
            ref_audio=ref_audio,
            ref_spk=ref_spk,
            spk_emotion=spk_emotion,
            model=model,
            tau=tau,
            format=format,
        )


def pydub_to_numpy(audio_segment: AudioSegment) -> np.ndarray:
    raw_data = audio_segment.raw_data
    sample_width = audio_segment.sample_width
    channels = audio_segment.channels
    audio_data: np.ndarray = np.frombuffer(raw_data, dtype=np.int16)
    if channels > 1:
        audio_data = audio_data.reshape((-1, channels))
        audio_data = audio_data.mean(axis=1).astype(np.int16)
    return audio_data


async def read_upload_file(file: UploadFile):
    audio_bytes = await file.read()
    audio_segment: AudioSegment = AudioSegment.from_file(io.BytesIO(audio_bytes))

    sample_rate = audio_segment.frame_rate
    samples = pydub_to_numpy(audio_segment=audio_segment)

    input_audio = (sample_rate, samples)
    return input_audio


def setup(app: APIManager):

    @app.post(
        "/v1/vc",
        # 此接口依赖模型 openvoice ，并且，不再维护，且准备废弃
        description="""
Voice cloning API

**Deprecated**
This API is deprecated and will be removed in the future.
Please use the `TTS API` instead.
""",
        response_class=StreamingResponse,
        tags=["Voice Clone"],
    )
    async def voice_clone(
        request: Request, form: VoiceCloneForm = Depends(VoiceCloneForm.as_form)
    ):
        model = form.model
        src_audio = form.src_audio
        ref_audio = form.ref_audio
        ref_spk = form.ref_spk
        spk_emotion = form.spk_emotion
        tau = form.tau
        format = form.format

        if ref_audio is None and ref_spk is None:
            raise HTTPException(
                status_code=422, detail="Either ref_audio or ref_spk should be provided"
            )
        if src_audio is None:
            raise HTTPException(status_code=422, detail="src_audio should be provided")
        if format not in AudioFormat.__members__:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid format. Supported formats are {AudioFormat.__members__}",
            )

        src_audio_wav = await read_upload_file(src_audio)

        if ref_spk is not None:
            calc_params = api_utils.calc_spk_style(spk=ref_spk)

            ref_spk = calc_params.get("spk")
            if not isinstance(ref_spk, TTSSpeaker):
                raise HTTPException(status_code=422, detail="Invalid speaker")

        if ref_audio is not None:
            ref_audio_wav = (
                await read_upload_file(ref_audio) if ref_audio is not None else None
            )
            ref_spk = TTSSpeaker.from_ref_wav(ref_wav=ref_audio_wav)

        vc_config = VCConfig(enabled=True, mid=model, emotion=spk_emotion, tau=tau)
        encoder_config = EncoderConfig(
            format=AudioFormat(format),
            bitrate="64k",
        )

        if ref_spk is None:
            raise HTTPException(
                status_code=422, detail="Either ref_audio or ref_spk should be provided"
            )

        try:
            handler = VCHandler(
                ref_spk=ref_spk,
                input_audio=src_audio_wav,
                vc_config=vc_config,
                encoder_config=encoder_config,
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
