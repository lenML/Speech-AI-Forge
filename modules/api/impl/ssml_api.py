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
        "/v1/ssml",
        response_class=FileResponse,
        tags=["SSML"],
        description="""
Synthesize speech from SSML-formatted input using a specified TTS model.

This endpoint supports multi-speaker, multi-style speech synthesis based on structured SSML input.
It can return audio in various formats (e.g., `raw`, `wav`, `mp3`), with optional enhancements and prosody adjustments.

### Supported Features
- Multiple speakers via `<voice spk="...">`
- Arbitrary text segmentation with sentence break markers (e.g., `eos="[uv_break]"`)
- Streaming or full-response audio
- Audio enhancement & pitch/speed adjustment via `EnhancerConfig` / `AdjustConfig`
- Custom batch size & segment length control via `batch_size` / `spliter_thr`

### Parameters
- `ssml` (str): SSML XML string containing structured speech content (required)
- `format` (str): Output audio format. One of: `raw`, `wav`, `mp3` (default: `raw`)
- `batch_size` (int): Batch size for internal TTS inference, must be > 0
- `eos` (str): End-of-sentence token for segmentation (default: `[uv_break]`)
- `model` (str): TTS model identifier to be used (default: `chat-tts`)
- `spliter_thr` (int): Threshold to split long texts (default: 100, minimum: 50)
- `enhancer` (EnhancerConfig): Optional audio enhancer settings
- `adjuster` (AdjustConfig): Optional pitch/speed/volume control
- `stream` (bool): If true, returns a streaming response; otherwise, file response

### Example SSML Input

```xml
<speak version="0.1">
    <voice spk="mona">ChatTTS 用于合成多角色多情感的有声书示例</voice>
    <voice spk="mona">黛玉冷笑道：</voice>
    <voice spk="doubao" emotion="happy">我说呢，亏了绊住，不然，早就飞起来了。</voice>
    <voice spk="mona">宝玉道：</voice>
    <voice spk="mona">“只许和你玩，替你解闷。不过偶然到他那里，就说这些闲话。”</voice>
    <voice spk="doubao" emotion="angry">“好没意思的话！ 去不去，关我什么事儿？ 又没叫你替我解闷儿 ，还许你不理我呢”</voice>
    <voice spk="mona">说着，便赌气回房去了。</voice>
</speak>
````

The endpoint returns a synthesized audio file or stream based on the provided SSML and configuration.
""",
    )(synthesize_ssml_api)
