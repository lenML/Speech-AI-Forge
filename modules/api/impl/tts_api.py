from fastapi import Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

import io
from pydantic import BaseModel
import soundfile as sf
from fastapi.responses import FileResponse


from modules.normalization import text_normalize

from modules import generate_audio as generate

from modules.api import utils as api_utils
from modules.api.Api import APIManager


class TTSParams(BaseModel):
    text: str = Query(..., description="Text to synthesize")
    spk: str = Query(
        "female2", description="Specific speaker by speaker name or speaker seed"
    )
    style: str = Query("chat", description="Specific style by style name")
    temperature: float = Query(
        0.3, description="Temperature for sampling (may be overridden by style or spk)"
    )
    top_P: float = Query(
        0.5, description="Top P for sampling (may be overridden by style or spk)"
    )
    top_K: int = Query(
        20, description="Top K for sampling (may be overridden by style or spk)"
    )
    seed: int = Query(
        -1, description="Seed for generate (may be overridden by style or spk)"
    )
    format: str = Query("mp3", description="Response audio format: [mp3,wav]")
    prompt1: str = Query("", description="Text prompt for inference")
    prompt2: str = Query("", description="Text prompt for inference")
    prefix: str = Query("", description="Text prefix for inference")


async def synthesize_tts(params: TTSParams = Depends()):
    try:
        text = text_normalize(params.text, is_end=False)

        calc_params = api_utils.calc_spk_style(spk=params.spk, style=params.style)

        spk = calc_params.get("spk", params.spk)
        seed = params.seed or calc_params.get("seed", params.seed)
        temperature = params.temperature or calc_params.get(
            "temperature", params.temperature
        )
        prefix = params.prefix or calc_params.get("prefix", params.prefix)

        sample_rate, audio_data = generate.generate_audio(
            text,
            temperature=temperature,
            top_P=params.top_P,
            top_K=params.top_K,
            spk=spk,
            infer_seed=seed,
            prompt1=params.prompt1 or calc_params.get("prompt1", ""),
            prompt2=params.prompt2 or calc_params.get("prompt2", ""),
            prefix=prefix,
        )

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="wav")
        buffer.seek(0)

        if format == "mp3":
            buffer = api_utils.wav_to_mp3(buffer)

        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        import logging

        logging.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


def setup(api_manager: APIManager):
    api_manager.get("/v1/tts", response_class=FileResponse)(synthesize_tts)
