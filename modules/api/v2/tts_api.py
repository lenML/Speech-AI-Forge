"""
tts post api
接受 json 格式数据
支持直接上传使用参考音频
"""

import base64
import logging
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from modules.api.Api import APIManager
from modules.api.v2.dcls import SpeakerConfig
from modules.core.handler.datacls.audio_model import AdjustConfig, EncoderConfig
from modules.core.handler.datacls.enhancer_model import EnhancerConfig
from modules.core.handler.datacls.tn_model import TNConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.handler.datacls.vc_model import VCConfig
from modules.core.handler.TTSHandler import TTSHandler
from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker
from modules.utils.bytes_to_wav import convert_bytes_to_wav_bytes

logger = logging.getLogger(__name__)

class V2TtsParams(BaseModel):

    # audio
    adjust: Optional[AdjustConfig] = None
    encoder: Optional[EncoderConfig] = None
    enhance: Optional[EnhancerConfig] = None
    infer: Optional[InferConfig] = None
    vc: Optional[VCConfig] = None
    tn: Optional[TNConfig] = None
    tts: TTSConfig = Field(default_factory=TTSConfig)
    # spk
    spk: Optional[SpeakerConfig] = None
    # input
    text: Optional[str] = None
    texts: Optional[list[str]] = None
    ssml: Optional[str] = None


async def forge_text_synthesize(params: V2TtsParams, request: Request):
    # spk
    spk = None
    if params.spk is not None:
        if params.spk.from_spk_id is not None:
            spk = spk_mgr.get_speaker_by_id(params.spk.from_spk_id)
        elif params.spk.from_spk_name is not None:
            spk = spk_mgr.get_speaker(params.spk.from_spk_name)
        elif params.spk.from_ref is not None:
            audio_data = base64.b64decode(params.spk.from_ref.wav_b64)
            wav_data, wav_sr = convert_bytes_to_wav_bytes(audio_bytes=audio_data)
            ref_text = params.spk.from_ref.text
            spk = TTSSpeaker.from_ref_wav_bytes(
                ref_wav=(wav_sr, wav_data),
                text=ref_text,
            )
    if spk is None:
        # TODO：部分模型不支持没有 spk 需要报错
        pass

    # input
    text = params.text
    texts = params.texts
    ssml = params.ssml

    if text is None and ssml is None and texts is None:
        raise HTTPException(
            status_code=400,
            detail="text or ssml must be set",
        )
    if text is not None and (ssml is not None or texts is not None):
        raise HTTPException(
            status_code=400,
            detail="text and ssml cannot be set at the same time",
        )

    # configs
    tts_config = params.tts or TTSConfig()
    infer_config = params.infer or InferConfig()
    vc_config = params.vc or VCConfig()
    tn_config = params.tn or TNConfig()
    enhancer_config = params.enhance or EnhancerConfig()
    encoder_config = params.encoder or EncoderConfig()
    adjust_config = params.adjust or AdjustConfig()

    handler = TTSHandler(
        ssml_content=ssml,
        batch_content=texts,
        text_content=text,
        spk=spk,
        tts_config=tts_config,
        infer_config=infer_config,
        adjust_config=adjust_config,
        enhancer_config=enhancer_config,
        encoder_config=encoder_config,
        vc_config=vc_config,
        tn_config=tn_config,
    )

    try:
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
    api_manager.post(
        "/v2/tts",
        description="""
**Text-to-Speech Synthesis API (v2)**

This endpoint converts text into speech, offering a wide range of customization options.
It accepts JSON formatted data in the request body.

**Core Functionality**:
*   Synthesizes speech from various text inputs.
*   Supports voice cloning by directly uploading a reference audio.

**Input Flexibility**:
*   Provide text via one of these mutually exclusive fields:
    *   `text`: A single string of text.
    *   `texts`: A list of strings for batch processing.
    *   `ssml`: Text formatted using Speech Synthesis Markup Language (SSML).

**Speaker Customization (`spk`)**:
*   Use pre-defined speakers via `spk.from_spk_id` or `spk.from_spk_name`.
*   **Voice Cloning**: Provide reference audio for voice cloning:
    *   `spk.from_ref.wav_b64`: Base64 encoded WAV audio data.
    *   `spk.from_ref.text`: The transcript corresponding to the reference audio.

**Audio Processing & Control**:
*   **Adjustments (`adjust`)**:
    *   `pitch`: Modify audio pitch.
    *   `speed_rate`: Adjust speaking rate.
    *   `volume_gain_db`: Change audio volume.
    *   `normalize`: Apply volume normalization.
    *   `remove_silence`: Trim silence from audio ends.
*   **Encoding (`encoder`)**:
    *   `format`: Output audio format (e.g., "mp3", "wav").
    *   `bitrate`: Audio bitrate (e.g., "64k").
    *   `acodec`: Audio codec (e.g., "libmp3lame").
*   **Enhancement (`enhance`)**:
    *   `enabled`: Toggle audio enhancement.
    *   `model`: Select enhancement model.
    *   Additional parameters like `nfe`, `solver`, `lambd`, `tau` for fine-tuning.
*   **Inference (`infer`)**:
    *   `batch_size`, `spliter_threshold`, `eos` for text processing.
    *   `seed`: For reproducible outputs.
    *   `stream`, `stream_chunk_size`: Enable and configure streaming output.
    *   `no_cache`, `sync_gen`: Control caching and generation mode.
*   **Text Normalization (`tn`)**:
    *   `enabled`/`disabled`: Specify text normalization rules.
*   **TTS Model Parameters (`tts`)**:
    *   `mid`: TTS model ID (e.g., "cosy-voice").
    *   `style`, `emotion`: Control speaking style and emotion.
    *   `temperature`, `top_p`, `top_k`: Adjust sampling parameters for generation.
    *   Deprecated: `prompt`, `prompt1`, `prompt2`, `prefix`.
*   **Voice Cloning (`vc`)** (Note: Potentially deprecated, mainly for OpenVoice):
    *   `enabled`, `mid`, `emotion`, `tau`.

**Response**:
*   **Success**: An audio file stream (`FileResponse`).
*   **Failure**: A JSON object detailing the error.

**Key Considerations**:
1.  Ensure only one of `text`, `texts`, or `ssml` is provided.
2.  For voice cloning, use high-quality reference audio (3-5 seconds recommended).
3.  Enabling audio enhancement may increase generation latency.
""",
        response_class=FileResponse,
        tags=["Forge V2"],
    )(forge_text_synthesize)
