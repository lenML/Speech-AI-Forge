from fastapi import HTTPException, Body
from fastapi.responses import StreamingResponse

import io
from pydantic import BaseModel
from fastapi.responses import FileResponse


from modules.ssml import parse_ssml
from modules.SynthesizeSegments import (
    SynthesizeSegments,
    synthesize_segment,
    combine_audio_segments,
)


from modules.api import utils as api_utils

from modules.api.Api import APIManager


class SSMLRequest(BaseModel):
    ssml: str
    format: str = "mp3"
    batch: bool = False


async def synthesize_ssml(
    request: SSMLRequest = Body(
        ..., description="JSON body with SSML string and format"
    )
):
    try:
        ssml = request.ssml
        format = request.format
        batch = request.batch

        if not ssml:
            raise HTTPException(status_code=400, detail="SSML content is required.")

        segments = parse_ssml(ssml)

        if batch:
            synthesize = SynthesizeSegments(16)
            audio_segments = synthesize.synthesize_segments(segments)
            combined_audio = combine_audio_segments(audio_segments)
            buffer = io.BytesIO()
            combined_audio.export(buffer, format="wav")
            buffer.seek(0)
            if format == "mp3":
                buffer = api_utils.wav_to_mp3(buffer)
            return StreamingResponse(buffer, media_type=f"audio/{format}")
        else:

            def audio_streamer():
                for segment in segments:
                    audio_segment = synthesize_segment(segment=segment)
                    buffer = io.BytesIO()
                    audio_segment.export(buffer, format="wav")
                    buffer.seek(0)
                    if format == "mp3":
                        buffer = api_utils.wav_to_mp3(buffer)
                    yield buffer.read()

            return StreamingResponse(audio_streamer(), media_type=f"audio/{format}")

    except Exception as e:
        import logging

        logging.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


def setup(api_manager: APIManager):
    api_manager.post("/v1/ssml", response_class=FileResponse)(synthesize_ssml)
