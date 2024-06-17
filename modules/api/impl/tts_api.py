from fastapi import Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.api.impl.handler.TTSHandler import TTSHandler
from modules.api.impl.model.audio_model import AdjustConfig, AudioFormat
from modules.api.impl.model.chattts_model import ChatTTSConfig, InferConfig
from modules.api.impl.model.enhancer_model import EnhancerConfig
from modules.speaker import Speaker


class TTSParams(BaseModel):
    text: str = Query(..., description="Text to synthesize")
    spk: str = Query(
        "female2", description="Specific speaker by speaker name or speaker seed"
    )
    style: str = Query("chat", description="Specific style by style name")
    temperature: float = Query(
        0.3, description="Temperature for sampling (may be overridden by style or spk)"
    )
    top_p: float = Query(
        0.5, description="Top P for sampling (may be overridden by style or spk)"
    )
    top_k: int = Query(
        20, description="Top K for sampling (may be overridden by style or spk)"
    )
    seed: int = Query(
        42, description="Seed for generate (may be overridden by style or spk)"
    )
    format: str = Query("mp3", description="Response audio format: [mp3,wav]")
    prompt1: str = Query("", description="Text prompt for inference")
    prompt2: str = Query("", description="Text prompt for inference")
    prefix: str = Query("", description="Text prefix for inference")
    bs: str = Query("8", description="Batch size for inference")
    thr: str = Query("100", description="Threshold for sentence spliter")
    eos: str = Query("[uv_break]", description="End of sentence str")

    enhance: bool = Query(False, description="Enable enhancer")
    denoise: bool = Query(False, description="Enable denoiser")

    speed: float = Query(1.0, description="Speed of the audio")
    pitch: float = Query(0, description="Pitch of the audio")
    volume_gain: float = Query(0, description="Volume gain of the audio")


async def synthesize_tts(params: TTSParams = Depends()):
    try:
        # Validate text
        if not params.text.strip():
            raise HTTPException(
                status_code=422, detail="Text parameter cannot be empty"
            )

        # Validate temperature
        if not (0 <= params.temperature <= 1):
            raise HTTPException(
                status_code=422, detail="Temperature must be between 0 and 1"
            )

        # Validate top_p
        if not (0 <= params.top_p <= 1):
            raise HTTPException(status_code=422, detail="top_p must be between 0 and 1")

        # Validate top_k
        if params.top_k <= 0:
            raise HTTPException(
                status_code=422, detail="top_k must be a positive integer"
            )
        if params.top_k > 100:
            raise HTTPException(
                status_code=422, detail="top_k must be less than or equal to 100"
            )

        # Validate format
        if params.format not in ["mp3", "wav"]:
            raise HTTPException(
                status_code=422,
                detail="Invalid format. Supported formats are mp3 and wav",
            )

        calc_params = api_utils.calc_spk_style(spk=params.spk, style=params.style)

        spk = calc_params.get("spk", params.spk)
        if not isinstance(spk, Speaker):
            raise HTTPException(status_code=422, detail="Invalid speaker")

        style = calc_params.get("style", params.style)
        seed = params.seed or calc_params.get("seed", params.seed)
        temperature = params.temperature or calc_params.get(
            "temperature", params.temperature
        )
        prefix = params.prefix or calc_params.get("prefix", params.prefix)
        prompt1 = params.prompt1 or calc_params.get("prompt1", params.prompt1)
        prompt2 = params.prompt2 or calc_params.get("prompt2", params.prompt2)
        eos = params.eos or ""

        batch_size = int(params.bs)
        threshold = int(params.thr)

        tts_config = ChatTTSConfig(
            style=style,
            temperature=temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            prefix=prefix,
            prompt1=prompt1,
            prompt2=prompt2,
        )
        infer_config = InferConfig(
            batch_size=batch_size,
            spliter_threshold=threshold,
            eos=eos,
            seed=seed,
        )
        adjust_config = AdjustConfig(
            pitch=params.pitch,
            speed_rate=params.speed,
            volume_gain_db=params.volume_gain,
        )
        enhancer_config = EnhancerConfig(
            enabled=params.enhance or params.denoise or False,
            lambd=0.9 if params.denoise else 0.1,
        )

        handler = TTSHandler(
            text_content=params.text,
            spk=spk,
            tts_config=tts_config,
            infer_config=infer_config,
            adjust_config=adjust_config,
            enhancer_config=enhancer_config,
        )

        buffer = handler.enqueue_to_buffer(format=AudioFormat(params.format))

        media_type = f"audio/{params.format}"
        if params.format == "mp3":
            media_type = "audio/mpeg"
        return StreamingResponse(buffer, media_type=media_type)

    except Exception as e:
        import logging

        logging.exception(e)

        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=str(e))


def setup(api_manager: APIManager):
    api_manager.get("/v1/tts", response_class=FileResponse)(synthesize_tts)
