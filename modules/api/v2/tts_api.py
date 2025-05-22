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
通用 TTS（Text-to-Speech）推理接口，支持多种输入形式（纯文本、SSML、文本批量），并可配置完整的语音合成处理流程。

支持功能：
- 纯文本（text）、SSML（ssml）、多段文本（texts）输入（只能三选一）
- 可指定说话人（spk）信息，包括：从已有说话人ID、说话人名获取，或上传参考音频生成
- 支持完整的音频处理链条配置（可选项）：语音增强、声码器配置、TTS 推理参数、VC（语音转换）、TN（文本归一化）、调整器、编码器
- 返回音频文件，格式取决于后端设置（一般为 WAV）

参数说明：
- `text`: 单段文本输入
- `texts`: 多段文本列表，用于批量合成
- `ssml`: 使用 SSML 格式进行输入（包含富文本语音控制）
- `spk`: 指定说话人信息（支持引用已有说话人或上传参考音频）
- `tts`, `infer`, `vc`, `tn`, `enhance`, `encoder`, `adjust`: 各类语音生成和处理配置（均为可选）

注意：
- `text`、`texts` 和 `ssml` 三者只能选择一个输入
- 若模型要求必须提供说话人信息，未设置 `spk` 时将抛出错误

返回值：
- 成功时返回合成语音的音频文件
""",
        response_class=FileResponse,
        tags=["Forge V2"],
    )(forge_text_synthesize)
