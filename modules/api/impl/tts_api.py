from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse

import io
import soundfile as sf
from fastapi.responses import FileResponse


from modules.utils.normalization import text_normalize

from modules import generate_audio as generate
from enum import Enum

from modules.api import utils as api_utils
from modules.api.Api import APIManager


class ParamsTypeError(Exception):
    pass


async def synthesize_tts(
    text: str = Query(..., description="Text to synthesize"),
    spk: str = Query(
        "female2", description="Specific speaker by speaker name or speaker seed"
    ),
    style: str = Query("chat", description="Specific style by style name"),
    temperature: float = Query(
        0.3, description="Temperature for sampling (may be overridden by style or spk)"
    ),
    top_P: float = Query(
        0.5, description="Top P for sampling (may be overridden by style or spk)"
    ),
    top_K: int = Query(
        20, description="Top K for sampling (may be overridden by style or spk)"
    ),
    seed: int = Query(
        34060637, description="Seed for generate (may be overridden by style or spk)"
    ),
    format: api_utils.AudioFormat = Query(
        "mp3", description="Response audio format: [mp3,wav]"
    ),
    prompt1: str = Query("", description="Text prompt for inference"),
    prompt2: str = Query("", description="Text prompt for inference"),
    prefix: str = Query("", description="Text prefix for inference"),
):
    try:
        text = text_normalize(text, is_end=True)

        params = api_utils.calc_spk_style(spk=spk, style=style)

        spk = params.get("spk", spk)
        seed = seed if seed else params.get("seed", seed)
        temperature = (
            temperature if temperature else params.get("temperature", temperature)
        )
        prefix = prefix if prefix else params.get("prefix", prefix)

        sample_rate, audio_data = generate.generate_audio(
            text,
            temperature=temperature,
            top_P=top_P,
            top_K=top_K,
            spk=spk,
            infer_seed=seed,
            prompt1=prompt1 if prompt1 else params.get("prompt1", ""),
            prompt2=prompt2 if prompt2 else params.get("prompt2", ""),
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
