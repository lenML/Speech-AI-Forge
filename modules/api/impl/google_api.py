from typing import Union
from fastapi import HTTPException

from pydantic import BaseModel


from modules.api.Api import APIManager
from modules.api.impl.handler.SSMLHandler import SSMLHandler
from modules.api.impl.handler.TTSHandler import TTSHandler
from modules.api.impl.model.audio_model import AdjustConfig, AudioFormat
from modules.api.impl.model.chattts_model import ChatTTSConfig, InferConfig
from modules.api.impl.model.enhancer_model import EnhancerConfig

from modules.speaker import Speaker, speaker_mgr


from modules.api import utils as api_utils


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


class AudioConfig(BaseModel):
    audioEncoding: AudioFormat = AudioFormat.mp3
    speakingRate: float = 1
    pitch: float = 0
    volumeGainDb: float = 0
    sampleRateHertz: int = 24000
    batchSize: int = 4
    spliterThreshold: int = 100


class GoogleTextSynthesizeRequest(BaseModel):
    input: SynthesisInput
    voice: VoiceSelectionParams
    audioConfig: AudioConfig
    enhancerConfig: EnhancerConfig = None


class GoogleTextSynthesizeResponse(BaseModel):
    audioContent: str


async def google_text_synthesize(request: GoogleTextSynthesizeRequest):
    input = request.input
    voice = request.voice
    audioConfig = request.audioConfig
    enhancerConfig = request.enhancerConfig

    # 提取参数

    # TODO 这个也许应该传给 normalizer
    language_code = voice.languageCode
    voice_name = voice.name
    infer_seed = voice.seed or 42
    eos = voice.eos or "[uv_break]"
    audio_format = audioConfig.audioEncoding

    if not isinstance(audio_format, AudioFormat) and isinstance(audio_format, str):
        audio_format = AudioFormat(audio_format)

    speaking_rate = audioConfig.speakingRate or 1
    pitch = audioConfig.pitch or 0
    volume_gain_db = audioConfig.volumeGainDb or 0

    batch_size = audioConfig.batchSize or 1

    spliter_threshold = audioConfig.spliterThreshold or 100

    # TODO
    sample_rate = audioConfig.sampleRateHertz or 24000

    params = api_utils.calc_spk_style(spk=voice.name, style=voice.style)

    # 虽然 calc_spk_style 可以解析 seed 形式，但是这个接口只准备支持 speakers list 中存在的 speaker
    if speaker_mgr.get_speaker(voice_name) is None:
        raise HTTPException(
            status_code=422, detail="The specified voice name is not supported."
        )

    if not isinstance(params.get("spk"), Speaker):
        raise HTTPException(
            status_code=422, detail="The specified voice name is not supported."
        )

    speaker = params.get("spk")
    tts_config = ChatTTSConfig(
        style=params.get("style", ""),
        temperature=voice.temperature,
        top_k=voice.topK,
        top_p=voice.topP,
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

    mime_type = f"audio/{audio_format.value}"
    if audio_format == AudioFormat.mp3:
        mime_type = "audio/mpeg"
    try:
        if input.text:
            text_content = input.text

            handler = TTSHandler(
                text_content=text_content,
                spk=speaker,
                tts_config=tts_config,
                infer_config=infer_config,
                adjust_config=adjust_config,
                enhancer_config=enhancer_config,
            )

            base64_string = handler.enqueue_to_base64(format=audio_format)
            return {"audioContent": f"data:{mime_type};base64,{base64_string}"}

        elif input.ssml:
            ssml_content = input.ssml

            handler = SSMLHandler(
                ssml_content=ssml_content,
                infer_config=infer_config,
                adjust_config=adjust_config,
                enhancer_config=enhancer_config,
            )

            base64_string = handler.enqueue_to_base64(format=audio_format)

            return {"audioContent": f"data:{mime_type};base64,{base64_string}"}

        else:
            raise HTTPException(
                status_code=422, detail="Invalid input text or ssml specified."
            )

    except Exception as e:
        import logging

        logging.exception(e)

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
    )(google_text_synthesize)
