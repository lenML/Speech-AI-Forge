from fastapi import Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

import io
from pydantic import BaseModel
import soundfile as sf
from fastapi.responses import FileResponse


from modules.normalization import text_normalize

from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.synthesize_audio import synthesize_audio


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
        42, description="Seed for generate (may be overridden by style or spk)"
    )
    format: str = Query("mp3", description="Response audio format: [mp3,wav]")
    prompt1: str = Query("", description="Text prompt for inference")
    prompt2: str = Query("", description="Text prompt for inference")
    prefix: str = Query("", description="Text prefix for inference")
    bs: str = Query("8", description="Batch size for inference")
    thr: str = Query("100", description="Threshold for sentence spliter")
    eos: str = Query("", description="End of sentence str")


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

        # Validate top_P
        if not (0 <= params.top_P <= 1):
            raise HTTPException(status_code=422, detail="top_P must be between 0 and 1")

        # Validate top_K
        if params.top_K <= 0:
            raise HTTPException(
                status_code=422, detail="top_K must be a positive integer"
            )
        if params.top_K > 100:
            raise HTTPException(
                status_code=422, detail="top_K must be less than or equal to 100"
            )

        # Validate format
        if params.format not in ["mp3", "wav"]:
            raise HTTPException(
                status_code=422,
                detail="Invalid format. Supported formats are mp3 and wav",
            )

        text = text_normalize(params.text, is_end=False)

        calc_params = api_utils.calc_spk_style(spk=params.spk, style=params.style)

        spk = calc_params.get("spk", params.spk)
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

        sample_rate, audio_data = synthesize_audio(
            text,
            temperature=temperature,
            top_P=params.top_P,
            top_K=params.top_K,
            spk=spk,
            infer_seed=seed,
            prompt1=prompt1,
            prompt2=prompt2,
            prefix=prefix,
            batch_size=batch_size,
            spliter_threshold=threshold,
            end_of_sentence=eos,
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

        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=str(e))


def setup(api_manager: APIManager):
    api_manager.get("/v1/tts", response_class=FileResponse)(synthesize_tts)
