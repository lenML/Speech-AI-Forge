from fastapi import HTTPException, Body
from fastapi.responses import StreamingResponse

import io
from pydantic import BaseModel
from fastapi.responses import FileResponse


from modules.ssml import parse_ssml, synthesize_segment


from modules.api import utils as api_utils

from modules.api.Api import APIManager


class SSMLRequest(BaseModel):
    ssml: str
    format: str = "mp3"


async def synthesize_ssml(
    request: SSMLRequest = Body(
        ..., description="JSON body with SSML string and format"
    )
):
    try:
        ssml = request.ssml
        format = request.format

        if not ssml:
            raise HTTPException(status_code=400, detail="SSML content is required.")

        segments = parse_ssml(ssml)

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
