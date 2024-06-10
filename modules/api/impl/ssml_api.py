from fastapi import HTTPException, Body
from fastapi.responses import StreamingResponse

import io
from pydantic import BaseModel
from fastapi.responses import FileResponse


from modules.normalization import text_normalize
from modules.ssml_parser.SSMLParser import create_ssml_parser
from modules.SynthesizeSegments import (
    SynthesizeSegments,
    combine_audio_segments,
)


from modules.api import utils as api_utils

from modules.api.Api import APIManager


class SSMLRequest(BaseModel):
    ssml: str
    format: str = "mp3"

    # NOTE: 🤔 也许这个值应该配置成系统变量？ 传进来有点奇怪
    batch_size: int = 4


async def synthesize_ssml(
    request: SSMLRequest = Body(
        ..., description="JSON body with SSML string and format"
    )
):
    try:
        ssml = request.ssml
        format = request.format.lower()
        batch_size = request.batch_size

        if batch_size < 1:
            raise HTTPException(
                status_code=400, detail="Batch size must be greater than 0."
            )

        if not ssml or ssml == "":
            raise HTTPException(status_code=400, detail="SSML content is required.")

        if format not in ["mp3", "wav"]:
            raise HTTPException(
                status_code=400, detail="Format must be 'mp3' or 'wav'."
            )

        parser = create_ssml_parser()
        segments = parser.parse(ssml)
        for seg in segments:
            seg["text"] = text_normalize(seg["text"], is_end=True)

        synthesize = SynthesizeSegments(batch_size)
        audio_segments = synthesize.synthesize_segments(segments)
        combined_audio = combine_audio_segments(audio_segments)
        buffer = io.BytesIO()
        combined_audio.export(buffer, format="wav")
        buffer.seek(0)
        if format == "mp3":
            buffer = api_utils.wav_to_mp3(buffer)
        return StreamingResponse(buffer, media_type=f"audio/{format}")

    except Exception as e:
        import logging

        logging.exception(e)

        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=str(e))


def setup(api_manager: APIManager):
    api_manager.post("/v1/ssml", response_class=FileResponse)(synthesize_ssml)
