from fastapi import Body, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from modules.api.Api import APIManager
from modules.core.handler.datacls.audio_model import (
    AdjustConfig,
    AudioFormat,
    EncoderConfig,
)
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.handler.TTSHandler import TTSHandler


class SSMLRequest(BaseModel):
    ssml: str
    format: AudioFormat = "mp3"

    # NOTE: 🤔 也许这个值应该配置成系统变量？ 传进来有点奇怪
    batch_size: int = 4

    # end of sentence
    eos: str = "[uv_break]"

    model: str = "chat-tts"

    spliter_thr: int = 100

    enhancer: EnhancerConfig = EnhancerConfig()
    adjuster: AdjustConfig = AdjustConfig()

    stream: bool = False


async def synthesize_ssml_api(
    request: SSMLRequest = Body(
        ..., description="JSON body with SSML string and format"
    )
):
    try:
        ssml = request.ssml
        format = request.format.lower()
        batch_size = request.batch_size
        eos = request.eos
        stream = request.stream
        spliter_thr = request.spliter_thr
        enhancer = request.enhancer
        adjuster = request.adjuster
        model = request.model

        if batch_size < 1:
            raise HTTPException(
                status_code=400, detail="Batch size must be greater than 0."
            )

        if spliter_thr < 50:
            raise HTTPException(
                status_code=400, detail="Spliter threshold must be greater than 50."
            )

        if not ssml or ssml == "":
            raise HTTPException(status_code=400, detail="SSML content is required.")

        if format not in ["mp3", "wav"]:
            raise HTTPException(
                status_code=400, detail="Format must be 'mp3' or 'wav'."
            )

        infer_config = InferConfig(
            batch_size=batch_size, spliter_threshold=spliter_thr, eos=eos, stream=stream
        )
        adjust_config = adjuster
        enhancer_config = enhancer
        encoder_config = EncoderConfig(
            format=AudioFormat(format),
            bitrate="64k",
        )
        tts_config = TTSConfig(mid=model)

        handler = TTSHandler(
            ssml_content=ssml,
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


def setup(api_manager: APIManager):
    api_manager.post("/v1/ssml", response_class=FileResponse, tags=["SSML"])(
        synthesize_ssml_api
    )
