import io
import logging
from typing import Literal, Optional, Union

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.api.constants import support_bitrates
from modules.core.handler.datacls.audio_model import (
    AdjustConfig,
    AudioFormat,
    EncoderConfig,
)
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.handler.TTSHandler import TTSHandler
from modules.core.models.zoo.ModelZoo import model_zoo
from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker

logger = logging.getLogger(__name__)


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

    format: str = Query("raw", description="Response audio format: [mp3,wav,raw]")
    bitrate: str = Query("64k", description="Response audio bitrate")

    prompt: str = Query("", description="Text prompt for inference")
    prompt1: str = Query("", description="Text prompt_1 for inference")
    prompt2: str = Query("", description="Text prompt_2 for inference")
    prefix: str = Query("", description="Text prefix for inference")
    bs: str = Query("8", description="Batch size for inference")
    thr: str = Query("100", description="Threshold for sentence spliter")
    eos: str = Query("[uv_break]", description="End of sentence str")

    enhance: bool = Query(False, description="Enable enhancer")
    denoise: bool = Query(False, description="Enable denoiser")

    speed: float = Query(1.0, description="Speed of the audio")
    pitch: float = Query(0, description="Pitch of the audio")
    volume_gain: float = Query(0, description="Volume gain of the audio")

    stream: bool = Query(False, description="Enable streaming generation")
    chunk_size: int = Query(64, description="Chunk size for streaming generation")

    no_cache: Union[bool, Literal["on", "off"]] = Query(
        False, description="Disable cache"
    )

    model: str = Query(
        "chat-tts",
        description="Model ID",
        examples=["chat-tts", "cosy-voice", "f5-tts"],
    )


async def synthesize_tts(request: Request, params: TTSParams = Depends()):
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
        if params.bitrate not in support_bitrates:
            raise HTTPException(
                status_code=422,
                detail=f"The specified bitrate is not supported. support bitrates: {str(support_bitrates)}",
            )

        format = params.format
        # Validate format
        if format not in AudioFormat.__members__:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid format. Supported formats are {AudioFormat.__members__}",
            )

        model_ids = model_zoo.get_tts_model_ids()
        if params.model not in model_ids:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid model_id. Supported model_ids are {model_ids}",
            )
        if params.chunk_size <= 0:
            raise HTTPException(
                status_code=422, detail="chunk_size must be a positive integer"
            )

        calc_params = api_utils.calc_spk_style(spk=params.spk, style=params.style)

        spk = calc_params.get("spk", params.spk)
        if not isinstance(spk, TTSSpeaker):
            raise HTTPException(status_code=422, detail="Invalid speaker")

        style = calc_params.get("style", params.style)
        seed = params.seed or calc_params.get("seed", params.seed)
        temperature = params.temperature or calc_params.get(
            "temperature", params.temperature
        )
        prefix = params.prefix or calc_params.get("prefix", params.prefix)
        prompt = params.prompt or calc_params.get("prompt", params.prompt)
        prompt1 = params.prompt1 or calc_params.get("prompt1", params.prompt1)
        prompt2 = params.prompt2 or calc_params.get("prompt2", params.prompt2)
        eos = params.eos or ""
        stream = params.stream
        chunk_size = params.chunk_size
        no_cache = (
            params.no_cache
            if isinstance(params.no_cache, bool)
            else params.no_cache == "on"
        )

        if eos == "[uv_break]" and params.model != "chat-tts":
            eos = " "

        batch_size = int(params.bs)
        threshold = int(params.thr)

        tts_config = TTSConfig(
            style=style,
            temperature=temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            prefix=prefix,
            prompt=prompt,
            prompt1=prompt1,
            prompt2=prompt2,
            mid=params.model,
        )
        infer_config = InferConfig(
            batch_size=batch_size,
            spliter_threshold=threshold,
            eos=eos,
            seed=seed,
            stream=stream,
            no_cache=no_cache,
            stream_chunk_size=chunk_size,
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
        encoder_config = EncoderConfig(
            format=AudioFormat(format),
            bitrate=params.bitrate,
        )

        # NOTE: 这个接口不在支持 voice clone
        vc_config = VCConfig(enabled=False)
        handler = TTSHandler(
            text_content=params.text,
            spk=spk,
            tts_config=tts_config,
            infer_config=infer_config,
            adjust_config=adjust_config,
            enhancer_config=enhancer_config,
            encoder_config=encoder_config,
            vc_config=vc_config,
        )
        handler.set_current_request(request=request)
        return await handler.enqueue_to_response()
    except Exception as e:
        import logging

        logging.exception(e)

        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=str(e))


def setup(api_manager: APIManager):
    api_manager.get(
        "/v1/tts",
        description="""
**Text-to-Speech Synthesis API (v1 - GET)**

This endpoint converts text into speech using GET request parameters.
It offers various options for customizing the voice, style, audio output, and processing.

**Mandatory Parameter**:
*   `text`: The text string to be synthesized.

**Speaker and Style Customization**:
*   `spk`: Specify the speaker by name or seed (e.g., "female2").
*   `style`: Define the speaking style (e.g., "chat").

**Generation Control**:
*   `temperature`: Sampling temperature (0.0-1.0, e.g., 0.3). Controls randomness.
*   `top_p`: Nucleus sampling probability (0.0-1.0, e.g., 0.5).
*   `top_k`: Limits sampling to the K most likely next tokens (e.g., 20).
*   `seed`: Seed for reproducible generation (e.g., 42).
*   `prompt`, `prompt1`, `prompt2`: Optional text prompts to guide inference.
*   `prefix`: Optional text prefix for inference.
*   `bs` (batch_size): Batch size for processing (e.g., "8").
*   `thr` (threshold): Sentence splitter threshold (e.g., "100").
*   `eos`: End-of-sentence marker (e.g., "[uv_break]").

**Output Audio Configuration**:
*   `format`: Desired audio output format. Supported: "mp3", "wav", "raw" (default).
*   `bitrate`: Audio bitrate for compressed formats (e.g., "64k").

**Audio Adjustments**:
*   `speed`: Playback speed multiplier (e.g., 1.0 for normal).
*   `pitch`: Pitch adjustment (e.g., 0 for no change).
*   `volume_gain`: Volume gain in dB (e.g., 0 for no change).

**Enhancements**:
*   `enhance`: Boolean to enable audio enhancement (default: false).
*   `denoise`: Boolean to enable audio denoising (default: false).

**Streaming Output**:
*   `stream`: Boolean to enable streaming audio generation (default: false).
*   `chunk_size`: Size of chunks for streaming (e.g., 64, if stream is true).

**Caching**:
*   `no_cache`: Boolean or "on"/"off" to disable caching (default: false).

**Model Selection**:
*   `model`: Specify the TTS model ID to use (e.g., "chat-tts", "cosy-voice").

**Response**:
*   **Success**: An audio file stream (`FileResponse`).
*   **Failure**: A JSON object detailing the error (e.g., validation errors, internal server error).

**Note**: This v1 endpoint does *not* support voice cloning via reference audio. For voice cloning, please refer to the v2 API. Parameters like temperature, top_p, top_k, seed, prompts, and prefix might be overridden by specific speaker (`spk`) or style (`style`) configurations.
""",
        response_class=FileResponse,
        tags=["TTS"],
    )(synthesize_tts)
