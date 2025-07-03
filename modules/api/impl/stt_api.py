import io
from typing import Optional

import numpy as np
from fastapi import Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from pydub import AudioSegment

from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.core.handler.datacls.stt_model import STTConfig, STTOutputFormat
from modules.core.handler.STTHandler import STTHandler
from modules.core.models.stt.STTModel import TranscribeResult


class TranscriptionsForm(BaseModel):
    file: UploadFile

    model: str = "whisper"

    prompt: Optional[str] = None
    prefix: Optional[str] = None

    language: Optional[str] = None
    temperature: Optional[float] = None
    sample_len: Optional[int] = None
    best_of: Optional[int] = None
    beam_size: Optional[int] = None
    patience: Optional[int] = None
    length_penalty: Optional[float] = None

    format: Optional[STTOutputFormat] = STTOutputFormat.txt

    highlight_words: Optional[bool] = False
    max_line_count: Optional[int] = None
    max_line_width: Optional[int] = None
    max_words_per_line: Optional[int] = None

    @classmethod
    def as_form(
        cls,
        file: UploadFile = File(...),
        model: str = Form("whisper.large"),
        prompt: Optional[str] = Form(None),
        prefix: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        temperature: Optional[float] = Form(None),
        sample_len: Optional[int] = Form(None),
        best_of: Optional[int] = Form(None),
        beam_size: Optional[int] = Form(None),
        patience: Optional[int] = Form(None),
        length_penalty: Optional[float] = Form(None),
        format: Optional[STTOutputFormat] = Form(STTOutputFormat.txt),
        highlight_words: Optional[bool] = Form(False),
        max_line_count: Optional[int] = Form(None),
        max_line_width: Optional[int] = Form(None),
        max_words_per_line: Optional[int] = Form(None),
    ):
        return cls(
            file=file,
            model=model,
            prompt=prompt,
            prefix=prefix,
            language=language,
            temperature=temperature,
            sample_len=sample_len,
            best_of=best_of,
            beam_size=beam_size,
            patience=patience,
            length_penalty=length_penalty,
            format=format,
            highlight_words=highlight_words,
            max_line_count=max_line_count,
            max_line_width=max_line_width,
            max_words_per_line=max_words_per_line,
        )


class TranscriptionsResponse(BaseModel):
    message: str
    data: TranscribeResult


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
        "/v1/stt/transcribe",
        response_model=TranscriptionsResponse,
        tags=["STT"],
        description="""
Transcribes audio into the input language.
""",
    )
    async def transcribe(
        form_data: TranscriptionsForm = Depends(TranscriptionsForm.as_form),
    ):
        file = form_data.file
        model = form_data.model
        prompt = form_data.prompt
        language = form_data.language
        temperature = form_data.temperature
        sample_len = form_data.sample_len
        best_of = form_data.best_of
        beam_size = form_data.beam_size
        patience = form_data.patience
        length_penalty = form_data.length_penalty
        response_format = form_data.format
        highlight_words = form_data.highlight_words
        max_line_count = form_data.max_line_count
        max_line_width = form_data.max_line_width
        max_words_per_line = form_data.max_words_per_line

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
            temperature=temperature,
            sample_len=sample_len,
            best_of=best_of,
            beam_size=beam_size,
            patience=patience,
            length_penalty=length_penalty,
            format=response_format,
            highlight_words=highlight_words,
            max_line_count=max_line_count,
            max_line_width=max_line_width,
            max_words_per_line=max_words_per_line,
        )

        try:
            handler = STTHandler(input_audio=input_audio, stt_config=sst_config)

            result = await handler.enqueue()
            return api_utils.success_response(result.__dict__)
        except Exception as e:
            import logging

            logging.exception(e)

            if isinstance(e, HTTPException):
                raise e
            else:
                raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/stt/stream",
        tags=["STT"],
        description="""
Transcribes audio into the input language in real-time.

* Not implemented yet (WIP)
""",
    )
    async def transcribe_stream(
        form_data: TranscriptionsForm = Depends(TranscriptionsForm.as_form),
    ):
        raise HTTPException(status_code=501, detail="Not implemented")
