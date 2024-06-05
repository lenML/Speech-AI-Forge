import base64
from fastapi import HTTPException

import io
import soundfile as sf
from pydantic import BaseModel


from modules.api.Api import APIManager
from modules.utils.audio import apply_prosody_to_audio_data
from modules.normalization import text_normalize

from modules import generate_audio as generate


from modules.ssml import parse_ssml
from modules.SynthesizeSegments import (
    SynthesizeSegments,
    combine_audio_segments,
    synthesize_segment,
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


class AudioConfig(BaseModel):
    audioEncoding: api_utils.AudioFormat = "mp3"
    speakingRate: float = 1
    pitch: float = 0
    volumeGainDb: float = 0
    sampleRateHertz: int
    batchSize: int = 1
    spliterThreshold: int = 100


class GoogleTextSynthesizeRequest(BaseModel):
    input: SynthesisInput
    voice: VoiceSelectionParams
    audioConfig: dict


class GoogleTextSynthesizeResponse(BaseModel):
    audioContent: str


async def google_text_synthesize(request: GoogleTextSynthesizeRequest):
    input = request.input
    voice = request.voice
    audioConfig = request.audioConfig

    # 提取参数
    language_code = voice.languageCode
    voice_name = voice.name
    infer_seed = voice.seed or 42
    audio_format = audioConfig.get("audioEncoding", "mp3")
    speaking_rate = audioConfig.get("speakingRate", 1)
    pitch = audioConfig.get("pitch", 0)
    volume_gain_db = audioConfig.get("volumeGainDb", 0)

    batch_size = audioConfig.get("batchSize", 1)
    spliter_threshold = audioConfig.get("spliterThreshold", 100)

    # TODO sample_rate
    sample_rate_hertz = audioConfig.get("sampleRateHertz", 24000)

    params = api_utils.calc_spk_style(spk=voice.name, style=voice.style)

    # TODO maybe need to change the sample rate
    sample_rate = 24000

    try:
        if input.text:
            # 处理文本合成逻辑
            text = text_normalize(input.text, is_end=True)
            sample_rate, audio_data = generate.generate_audio(
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
            )

        elif input.ssml:
            # 处理SSML合成逻辑
            segments = parse_ssml(input.ssml)
            for seg in segments:
                seg["text"] = text_normalize(seg["text"], is_end=True)

            if len(segments) == 0:
                raise HTTPException(
                    status_code=400, detail="The SSML text is empty or parsing failed."
                )

            synthesize = SynthesizeSegments(batch_size=batch_size)
            audio_segments = synthesize.synthesize_segments(segments)
            combined_audio = combine_audio_segments(audio_segments)

            buffer = io.BytesIO()
            combined_audio.export(buffer, format="wav")

            buffer.seek(0)

            audio_data = buffer.read()

        else:
            raise HTTPException(
                status_code=400, detail="Either text or SSML input must be provided."
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
        raise HTTPException(status_code=500, detail=str(e))


def setup(app: APIManager):
    app.post("/v1/google/text:synthesize", response_model=GoogleTextSynthesizeResponse)(
        google_text_synthesize
    )
