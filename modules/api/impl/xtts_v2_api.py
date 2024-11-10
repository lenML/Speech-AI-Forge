import logging

from fastapi import HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modules.api.Api import APIManager
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

logger = logging.getLogger(__name__)


class XTTS_V2_Settings:
    def __init__(self):
        self.stream_chunk_size = 100
        self.temperature = 0.3
        self.speed = 1

        # TODO: 这两个参数现在用不着...但是其实gpt是可以用的可以考虑增加
        self.length_penalty = 0.5
        self.repetition_penalty = 1.0

        self.top_p = 0.7
        self.top_k = 20
        self.enable_text_splitting = True

        # 下面是额外配置 xtts_v2 中不包含的，但是本系统需要的
        self.batch_size = 4
        self.eos = "[uv_break]"
        self.infer_seed = 42
        self.use_decoder = True
        self.prompt1 = ""
        self.prompt2 = ""
        self.prefix = ""
        self.spliter_threshold = 100
        self.style = ""


class TTSSettingsRequest(BaseModel):
    # 这个 stream_chunk 现在当作 spliter_threshold 用
    stream_chunk_size: int
    temperature: float
    speed: float
    length_penalty: float
    repetition_penalty: float
    top_p: float
    top_k: int
    enable_text_splitting: bool

    batch_size: int = None
    eos: str = None
    infer_seed: int = None
    use_decoder: bool = None
    prompt1: str = None
    prompt2: str = None
    prefix: str = None
    spliter_threshold: int = None
    style: str = None


class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: str

    # xtts 接口这个是必填的，但是我们这里不需要所以可以为空
    language: str = "cn"


def setup(app: APIManager):
    XTTSV2 = XTTS_V2_Settings()

    @app.get("/v1/xtts_v2/speakers", tags=["XTTS"])
    async def speakers():
        spks = spk_mgr.list_speakers()
        return [
            {
                "name": spk.name,
                "voice_id": spk.id,
                # TODO: 也许可以放一个 "/v1/tts" 接口地址在这里
                "preview_url": "",
            }
            for spk in spks
        ]

    @app.post(
        "/v1/xtts_v2/tts_to_audio", response_class=StreamingResponse, tags=["XTTS"]
    )
    async def tts_to_audio(request: SynthesisRequest):
        text = request.text
        # speaker_wav 就是 speaker id 。。。
        voice_id = request.speaker_wav
        language = request.language

        spk = spk_mgr.get_speaker_by_id(voice_id) or spk_mgr.get_speaker(voice_id)
        if spk is None:
            raise HTTPException(status_code=400, detail="Invalid speaker id")

        tts_config = TTSConfig(
            style=XTTSV2.style,
            temperature=XTTSV2.temperature,
            top_k=XTTSV2.top_k,
            top_p=XTTSV2.top_p,
            prefix=XTTSV2.prefix,
            prompt1=XTTSV2.prompt1,
            prompt2=XTTSV2.prompt2,
        )
        infer_config = InferConfig(
            batch_size=XTTSV2.batch_size,
            spliter_threshold=XTTSV2.spliter_threshold,
            eos=XTTSV2.eos,
            seed=XTTSV2.infer_seed,
            stream=False,
        )
        adjust_config = AdjustConfig(
            speed_rate=XTTSV2.speed,
        )
        # TODO: support enhancer
        enhancer_config = EnhancerConfig(
            # enabled=params.enhance or params.denoise or False,
            # lambd=0.9 if params.denoise else 0.1,
        )
        encoder_config = EncoderConfig(
            format=AudioFormat.raw,
        )

        handler = TTSHandler(
            text_content=text,
            spk=spk,
            tts_config=tts_config,
            infer_config=infer_config,
            adjust_config=adjust_config,
            enhancer_config=enhancer_config,
            encoder_config=encoder_config,
            # NOTE: 这里不支持是因为 xtts 接口不太好加参数
            vc_config=VCConfig(enabled=False),
        )

        return handler.enqueue_to_response(request=request)

    @app.get("/v1/xtts_v2/tts_stream", tags=["XTTS"])
    async def tts_stream(
        request: Request,
        text: str = Query(),
        speaker_wav: str = Query(),
        language: str = Query("cn"),
        # just for test
        no_cache: bool = Query(False),
    ):
        # speaker_wav 就是 speaker id 。。。
        voice_id = speaker_wav

        spk = spk_mgr.get_speaker_by_id(voice_id) or spk_mgr.get_speaker(voice_id)
        if spk is None:
            raise HTTPException(status_code=400, detail="Invalid speaker id")

        tts_config = TTSConfig(
            style=XTTSV2.style,
            temperature=XTTSV2.temperature,
            top_k=XTTSV2.top_k,
            top_p=XTTSV2.top_p,
            prefix=XTTSV2.prefix,
            prompt1=XTTSV2.prompt1,
            prompt2=XTTSV2.prompt2,
        )
        infer_config = InferConfig(
            batch_size=XTTSV2.batch_size,
            spliter_threshold=XTTSV2.spliter_threshold,
            eos=XTTSV2.eos,
            seed=XTTSV2.infer_seed,
            stream=True,
            # stream_chunk_size=64,
            no_cache=no_cache,
        )
        adjust_config = AdjustConfig(
            speed_rate=XTTSV2.speed,
        )
        # TODO: support enhancer
        enhancer_config = EnhancerConfig(
            # enabled=params.enhance or params.denoise or False,
            # lambd=0.9 if params.denoise else 0.1,
        )
        encoder_config = EncoderConfig(
            format=AudioFormat.raw,
        )

        handler = TTSHandler(
            text_content=text,
            spk=spk,
            tts_config=tts_config,
            infer_config=infer_config,
            adjust_config=adjust_config,
            enhancer_config=enhancer_config,
            encoder_config=encoder_config,
            # NOTE: 这里不支持是因为 xtts 接口不太好加参数
            vc_config=VCConfig(enabled=False),
        )

        return handler.enqueue_to_response(request=request)

    @app.post("/v1/xtts_v2/set_tts_settings", tags=["XTTS"])
    async def set_tts_settings(request: TTSSettingsRequest):
        try:
            if request.stream_chunk_size < 50:
                raise HTTPException(
                    status_code=400, detail="stream_chunk_size must be greater than 0"
                )
            if request.temperature < 0:
                raise HTTPException(
                    status_code=400, detail="temperature must be greater than 0"
                )
            if request.speed < 0:
                raise HTTPException(
                    status_code=400, detail="speed must be greater than 0"
                )
            if request.length_penalty < 0:
                raise HTTPException(
                    status_code=400, detail="length_penalty must be greater than 0"
                )
            if request.repetition_penalty < 0:
                raise HTTPException(
                    status_code=400, detail="repetition_penalty must be greater than 0"
                )
            if request.top_p < 0:
                raise HTTPException(
                    status_code=400, detail="top_p must be greater than 0"
                )
            if request.top_k < 0:
                raise HTTPException(
                    status_code=400, detail="top_k must be greater than 0"
                )

            XTTSV2.stream_chunk_size = request.stream_chunk_size
            XTTSV2.spliter_threshold = request.stream_chunk_size

            XTTSV2.temperature = request.temperature
            XTTSV2.speed = request.speed
            XTTSV2.length_penalty = request.length_penalty
            XTTSV2.repetition_penalty = request.repetition_penalty
            XTTSV2.top_p = request.top_p
            XTTSV2.top_k = request.top_k
            XTTSV2.enable_text_splitting = request.enable_text_splitting

            # TODO: checker
            if request.batch_size:
                XTTSV2.batch_size = request.batch_size
            if request.eos:
                XTTSV2.eos = request.eos
            if request.infer_seed:
                XTTSV2.infer_seed = request.infer_seed
            if request.use_decoder:
                XTTSV2.use_decoder = request.use_decoder
            if request.prompt1:
                XTTSV2.prompt1 = request.prompt1
            if request.prompt2:
                XTTSV2.prompt2 = request.prompt2
            if request.prefix:
                XTTSV2.prefix = request.prefix
            if request.spliter_threshold:
                XTTSV2.spliter_threshold = request.spliter_threshold
            if request.style:
                XTTSV2.style = request.style

            return {"message": "Settings successfully applied"}
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
