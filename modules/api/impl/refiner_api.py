from typing import Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel

from modules import refiner
from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.core.handler.datacls.tn_model import TNConfig
from modules.core.tn.ChatTtsTN import ChatTtsTN
from modules.core.tn.CosyVoiceTN import CosyVoiceTN
from modules.core.tn.FishSpeechTN import FishSpeechTN


class RefineTextRequest(BaseModel):
    text: str
    prompt: str = "[oral_2][laugh_0][break_6]"
    seed: int = -1
    top_P: float = 0.7
    top_K: int = 20
    temperature: float = 0.7
    repetition_penalty: float = 1.0
    max_new_token: int = 384
    spliter_threshold: int = 300
    normalize: bool = True


async def refiner_prompt_post(request: RefineTextRequest):
    """
    This endpoint receives a prompt and returns the refined result
    """

    try:
        text = request.text
        if request.normalize:
            text = ChatTtsTN.normalize(request.text)
        # TODO 需要迁移使用 refiner model
        refined_text = refiner.refine_text(
            text=text,
            prompt=request.prompt,
            seed=request.seed,
            top_P=request.top_P,
            top_K=request.top_K,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            max_new_token=request.max_new_token,
            spliter_threshold=request.spliter_threshold,
        )
        return api_utils.success_response(data=refined_text)

    except Exception as e:
        import logging

        logging.exception(e)

        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=str(e))


class TextNormalizeRequest(BaseModel):
    text: str

    pipe_id: Literal["chat-tts", "cosy-voice", "fish-speech"] = "chat-tts"

    config: Optional[TNConfig] = None


pipelines = {
    "chat-tts": ChatTtsTN,
    "cosy-voice": CosyVoiceTN,
    "fish-speech": FishSpeechTN,
}


async def text_normalize_post(request: TextNormalizeRequest):
    """
    This endpoint receives a text and returns the normalized text
    """
    try:
        text = request.text
        pipe_id = request.pipe_id
        config = request.config

        if text is None:
            raise HTTPException(status_code=400, detail="text is required")
        if pipe_id not in pipelines:
            raise HTTPException(
                status_code=400, detail=f"pipe_id {pipe_id} is not supported"
            )

        pipe = pipelines[pipe_id]

        normalized_text = pipe.normalize(text=text, config=config)
        return api_utils.success_response(data=normalized_text)
    except Exception as e:
        import logging

        logging.exception(e)
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=str(e))


def setup(app: APIManager):
    app.post(
        "/v1/prompt/refine",
        response_model=api_utils.BaseResponse,
        tags=["Text Refiner"],
    )(refiner_prompt_post)

    app.post(
        "/v1/text/normalize",
        response_model=api_utils.BaseResponse,
        tags=["Text Refiner"],
    )(text_normalize_post)
