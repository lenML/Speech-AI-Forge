from pydantic import BaseModel


from modules.speaker import speaker_mgr


from modules.data import styles_mgr


from modules.api import utils as api_utils
from modules.api.Api import APIManager


async def list_styles():
    return {"message": "ok", "data": styles_mgr.list_items()}


async def list_speakers():
    return {
        "message": "ok",
        "data": [spk.to_json() for spk in speaker_mgr.list_speakers()],
    }


class CreateSpeaker(BaseModel):
    seed: int
    name: str = ""


async def create_speaker(request: CreateSpeaker):
    speaker = speaker_mgr.create_speaker(request.seed, request.name)
    return {"message": "ok", "data": speaker.to_json()}


async def refresh_speakers():
    speaker_mgr.refresh_speakers()
    return {"message": "ok"}


def setup(app: APIManager):
    app.get("/v1/styles/list", response_model=api_utils.BaseResponse)(list_styles)
    app.get("/v1/speakers/list", response_model=api_utils.BaseResponse)(list_speakers)
    app.post("/v1/speaker/create", response_model=api_utils.BaseResponse)(
        create_speaker
    )
    app.post("/v1/speaker/refresh", response_model=api_utils.BaseResponse)(
        refresh_speakers
    )
