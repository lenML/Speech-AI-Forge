from typing import Union

from fastapi import HTTPException, Request
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
from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker


class SynthesisInput(BaseModel):
    text: Union[str, None] = None
    ssml: Union[str, None] = None


class VoiceSelectionParams(BaseModel):
    languageCode: str = "ZH-CN"

    name: str = "female2"
    style: str = ""
    temperature: float = 0.3
    topP: float = 0.7
    topK: int = 20
    seed: int = 42

    # end_of_sentence
    eos: str = "[uv_break]"

    model: str = "chat-tts"


class AudioConfig(BaseModel):
    # 编码器参数
    audioEncoding: AudioFormat = AudioFormat.raw
    audioBitrate: str = "64k"

    speakingRate: float = 1
    pitch: float = 0
    volumeGainDb: float = 0
    sampleRateHertz: int = 24000
    batchSize: int = 4
    spliterThreshold: int = 100


class GoogleTextSynthesizeParams(BaseModel):
    input: SynthesisInput
    voice: VoiceSelectionParams
    audioConfig: AudioConfig
    enhancerConfig: EnhancerConfig = None


class GoogleTextSynthesizeResponse(BaseModel):
    audioContent: str


async def google_text_synthesize(params: GoogleTextSynthesizeParams, request: Request):
    input = params.input
    voice = params.voice
    audioConfig = params.audioConfig
    enhancerConfig = params.enhancerConfig

    # 提取参数

    # TODO 这个也许应该传给 normalizer
    language_code = voice.languageCode
    voice_name = voice.name
    voice_model = voice.model
    infer_seed = voice.seed or 42
    eos = voice.eos or "[uv_break]"
    audio_format = audioConfig.audioEncoding
    audio_bitrate = audioConfig.audioBitrate

    if not isinstance(audio_format, AudioFormat) and isinstance(audio_format, str):
        audio_format = AudioFormat(audio_format)
    if audio_bitrate not in support_bitrates:
        raise HTTPException(
            status_code=422,
            detail=f"The specified bitrate is not supported. support bitrates: {str(support_bitrates)}",
        )

    speaking_rate = audioConfig.speakingRate or 1
    pitch = audioConfig.pitch or 0
    volume_gain_db = audioConfig.volumeGainDb or 0

    batch_size = audioConfig.batchSize or 1

    spliter_threshold = audioConfig.spliterThreshold or 100

    # TODO
    sample_rate = audioConfig.sampleRateHertz or 24000

    params = api_utils.calc_spk_style(spk=voice.name, style=voice.style)

    # 虽然 calc_spk_style 可以解析 seed 形式，但是这个接口只准备支持 speakers list 中存在的 speaker
    if spk_mgr.get_speaker(voice_name) is None:
        raise HTTPException(
            status_code=422, detail="The specified voice name is not supported."
        )

    if not isinstance(params.get("spk"), TTSSpeaker):
        raise HTTPException(
            status_code=422, detail="The specified voice name is not supported."
        )

    speaker = params.get("spk")
    tts_config = TTSConfig(
        style=params.get("style", ""),
        temperature=voice.temperature,
        top_k=voice.topK,
        top_p=voice.topP,
        mid=voice_model,
    )
    infer_config = InferConfig(
        batch_size=batch_size,
        spliter_threshold=spliter_threshold,
        eos=eos,
        seed=infer_seed,
    )
    adjust_config = AdjustConfig(
        speaking_rate=speaking_rate,
        pitch=pitch,
        volume_gain_db=volume_gain_db,
    )
    enhancer_config = enhancerConfig
    encoder_config = EncoderConfig(
        format=audio_format,
        bitrate=audio_bitrate,
    )

    text_content = input.text
    ssml_content = input.ssml
    handler = TTSHandler(
        text_content=text_content,
        ssml_content=ssml_content,
        spk=speaker,
        tts_config=tts_config,
        infer_config=infer_config,
        adjust_config=adjust_config,
        enhancer_config=enhancer_config,
        encoder_config=encoder_config,
        vc_config=VCConfig(enabled=False),
    )
    try:
        media_type = handler.get_media_type()
        handler.set_current_request(request=request)
        base64_string = await handler.enqueue_to_base64()
        return {"audioContent": f"data:{media_type};base64,{base64_string}"}
    except Exception as e:
        import logging

        logging.exception(e)
        handler.interrupt()

        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=str(e))


def setup(app: APIManager):
    app.post(
        "/v1/text:synthesize",
        response_model=GoogleTextSynthesizeResponse,
        description="""
google api document: <br/>
[https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize](https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize)

- 多个属性在本系统中无用仅仅是为了兼容google api
- voice 中的 topP, topK, temperature 为本系统中的参数
- voice.name 即 speaker name （或者speaker seed）
- voice.seed 为 infer seed （可在webui中测试具体作用）

- 编码格式影响的是 audioContent 的二进制格式，所以所有format都是返回带有base64数据的json
        """,
        tags=["Google API"],
    )(google_text_synthesize)

    @app.post(
        "/v1/speech:recognize",
        # response_model=None,
        description="Performs synchronous speech recognition: receive results after all audio has been sent and processed.",
        tags=["Google API"],
    )
    async def speech_recognize():
        raise HTTPException(status_code=501, detail="Not implemented")

    @app.post(
        "/v1/speech:longrunningrecognize",
        # response_model=None,
        description="Performs asynchronous speech recognition: receive results via the google.longrunning.Operations interface.",
        tags=["Google API"],
    )
    async def long_running_recognize():
        raise HTTPException(status_code=501, detail="Not implemented")
