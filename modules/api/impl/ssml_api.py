from fastapi import Body, HTTPException, Request
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


class SSMLParams(BaseModel):
    ssml: str
    format: AudioFormat = "raw"

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
    request: Request,
    params: SSMLParams = Body(..., description="JSON body with SSML string and format"),
):
    try:
        ssml = params.ssml
        format = params.format.lower()
        batch_size = params.batch_size
        eos = params.eos
        stream = params.stream
        spliter_thr = params.spliter_thr
        enhancer = params.enhancer
        adjuster = params.adjuster
        model = params.model

        if batch_size < 1:
            raise HTTPException(
                status_code=422, detail="Batch size must be greater than 0."
            )

        if spliter_thr < 50:
            raise HTTPException(
                status_code=422, detail="Spliter threshold must be greater than 50."
            )

        if not ssml or ssml == "":
            raise HTTPException(status_code=422, detail="SSML content is required.")

        if format not in AudioFormat.__members__:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid format. Supported formats are {AudioFormat.__members__}",
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


def setup(api_manager: APIManager):
    api_manager.post("/v1/ssml", response_class=FileResponse, tags=["SSML"])(
        synthesize_ssml_api
    )
