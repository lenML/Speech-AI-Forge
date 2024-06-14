import base64
from typing import Literal
from fastapi import HTTPException

import io
import soundfile as sf
from pydantic import BaseModel


from modules.Enhancer.ResembleEnhance import (
    apply_audio_enhance,
    apply_audio_enhance_full,
)
from modules.api.Api import APIManager
from modules.synthesize_audio import synthesize_audio
from modules.utils import audio
from modules.utils.audio import apply_prosody_to_audio_data
from modules.normalization import text_normalize

from modules import generate_audio as generate
from modules.speaker import speaker_mgr


from modules.ssml_parser.SSMLParser import create_ssml_parser
from modules.SynthesizeSegments import (
    SynthesizeSegments,
    combine_audio_segments,
)

from modules.api import utils as api_utils


class SynthesisInput(BaseModel):
    text: str = ""
    ssml: str = ""


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
    audioEncoding: api_utils.AudioFormat = "mp3"
    speakingRate: float = 1
    pitch: float = 0
    volumeGainDb: float = 0
    sampleRateHertz: int = 24000
    batchSize: int = 1
    spliterThreshold: int = 100


class EnhancerConfig(BaseModel):
    enabled: bool = False
    model: str = "resemble-enhance"
    nfe: int = 32
    solver: Literal["midpoint", "rk4", "euler"] = "midpoint"
    lambd: float = 0.5
    tau: float = 0.5


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
    audio_format = audioConfig.audioEncoding or "mp3"
    speaking_rate = audioConfig.speakingRate or 1
    pitch = audioConfig.pitch or 0
    volume_gain_db = audioConfig.volumeGainDb or 0

    batch_size = audioConfig.batchSize or 1

    spliter_threshold = audioConfig.spliterThreshold or 100

    sample_rate = audioConfig.sampleRateHertz or 24000

    params = api_utils.calc_spk_style(spk=voice.name, style=voice.style)

    # 虽然 calc_spk_style 可以解析 seed 形式，但是这个接口只准备支持 speakers list 中存在的 speaker
    if speaker_mgr.get_speaker(voice_name) is None:
        raise HTTPException(
            status_code=422, detail="The specified voice name is not supported."
        )

    if audio_format != "mp3" and audio_format != "wav":
        raise HTTPException(
            status_code=422, detail="Invalid audio encoding format specified."
        )

    if enhancerConfig.enabled:
        # TODO enhancer params checker
        pass

    try:
        if input.text:
            # 处理文本合成逻辑
            text = text_normalize(input.text, is_end=True)
            sample_rate, audio_data = synthesize_audio(
                text,
                temperature=(
                    voice.temperature
                    if voice.temperature
                    else params.get("temperature", 0.3)
                ),
                top_P=voice.topP if voice.topP else params.get("top_p", 0.7),
                top_K=voice.topK if voice.topK else params.get("top_k", 20),
                spk=params.get("spk", -1),
                infer_seed=infer_seed,
                prompt1=params.get("prompt1", ""),
                prompt2=params.get("prompt2", ""),
                prefix=params.get("prefix", ""),
                batch_size=batch_size,
                spliter_threshold=spliter_threshold,
                end_of_sentence=eos,
            )

        elif input.ssml:
            parser = create_ssml_parser()
            segments = parser.parse(input.ssml)
            for seg in segments:
                seg["text"] = text_normalize(seg["text"], is_end=True)

            if len(segments) == 0:
                raise HTTPException(
                    status_code=422, detail="The SSML text is empty or parsing failed."
                )

            synthesize = SynthesizeSegments(
                batch_size=batch_size, eos=eos, spliter_thr=spliter_threshold
            )
            audio_segments = synthesize.synthesize_segments(segments)
            combined_audio = combine_audio_segments(audio_segments)

            sample_rate, audio_data = audio.pydub_to_np(combined_audio)
        else:
            raise HTTPException(
                status_code=422, detail="Either text or SSML input must be provided."
            )

        if enhancerConfig.enabled:
            audio_data, sample_rate = apply_audio_enhance_full(
                audio_data=audio_data,
                sr=sample_rate,
                nfe=enhancerConfig.nfe,
                solver=enhancerConfig.solver,
                lambd=enhancerConfig.lambd,
                tau=enhancerConfig.tau,
            )

        audio_data = apply_prosody_to_audio_data(
            audio_data,
            rate=speaking_rate,
            pitch=pitch,
            volume=volume_gain_db,
            sr=sample_rate,
        )

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="wav")
        buffer.seek(0)

        if audio_format == "mp3":
            buffer = api_utils.wav_to_mp3(buffer)

        base64_encoded = base64.b64encode(buffer.read())
        base64_string = base64_encoded.decode("utf-8")

        return {
            "audioContent": f"data:audio/{audio_format.lower()};base64,{base64_string}"
        }

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
