import torch
from fastapi import Depends, HTTPException, Query, Request
from pydantic import BaseModel

from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.core.models.tts import ChatTtsModel
from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker


class CreateSpeaker(BaseModel):
    name: str
    gender: str
    describe: str
    tensor: list = None
    seed: int = None


class UpdateSpeaker(BaseModel):
    id: str
    name: str
    gender: str
    describe: str
    tensor: list


class SpeakerDetail(BaseModel):
    id: str
    with_emb: bool = False


class SpeakersUpdate(BaseModel):
    speakers: list


def setup(app: APIManager):
    class SpkListParams(BaseModel):
        full_data: bool = Query(False, description="Return all data")

    @app.get(
        "/v1/speakers/list", response_model=api_utils.BaseResponse, tags=["Speaker"]
    )
    async def list_speakers(
        request: Request,
        parmas: SpkListParams = Depends(),
    ):
        data = [
            spk.to_json(just_info=not parmas.full_data)
            for spk in spk_mgr.list_speakers()
        ]

        return api_utils.success_response(data)

    @app.post(
        "/v1/speakers/refresh", response_model=api_utils.BaseResponse, tags=["Speaker"]
    )
    async def refresh_speakers():
        spk_mgr.refresh()
        return api_utils.success_response(None)

    @app.post(
        "/v1/speakers/update", response_model=api_utils.BaseResponse, tags=["Speaker"]
    )
    async def update_speakers(request: SpeakersUpdate):
        for config in request.speakers:
            config: dict = config
            cfg_spk = TTSSpeaker.from_json(config)
            spk = spk_mgr.get_speaker_by_id(cfg_spk.id)
            if spk is None:
                raise HTTPException(
                    status_code=404, detail=f"Speaker not found: {config['id']}"
                )
            spk.set_name(cfg_spk.name)
            spk.set_gender(cfg_spk.gender)
            spk.set_desc(cfg_spk.desc)
            spk.set_version(cfg_spk.version)
            spk.set_author(cfg_spk.author)

            # TODO: 支持更新其他属性
            # if (
            #     config.get("tensor")
            #     and isinstance(config["tensor"], list)
            #     and len(config["tensor"]) > 0
            # ):
            #     # number array => Tensor
            #     token = torch.tensor(config["tensor"])
            #     spk.set_token(tokens=[token], model_id="chat-tts")
        spk_mgr.save_all()

        return api_utils.success_response(None)

    # TODO 需要适配新版本 speaker
    @app.post(
        "/v1/speaker/create", response_model=api_utils.BaseResponse, tags=["Speaker"]
    )
    async def create_speaker(request: CreateSpeaker):
        if (
            request.tensor
            and isinstance(request.tensor, list)
            and len(request.tensor) > 0
        ):
            # from tensor
            token = torch.tensor(request.tensor)
            spk = TTSSpeaker.empty()
            spk.set_token(tokens=[token], model_id="chat-tts")
        elif request.seed:
            # from seed
            spk = ChatTtsModel.ChatTTSModel.create_speaker_from_seed(request.seed)
            spk.set_name(request.name)
            spk.set_gender(request.gender)
            spk.set_desc(request.describe)
        else:
            raise HTTPException(
                status_code=400, detail="Missing tensor or seed in request"
            )
        filepath = spk_mgr.filepath(request.name + ".spkv1.json")
        spk_mgr.save_item(spk, file_path=filepath)
        spk_mgr.refresh()
        return api_utils.success_response(spk.to_json())

    @app.post(
        "/v1/speaker/update", response_model=api_utils.BaseResponse, tags=["Speaker"]
    )
    async def update_speaker(request: UpdateSpeaker):
        speaker = spk_mgr.get_speaker_by_id(request.id)
        if speaker is None:
            raise HTTPException(
                status_code=404, detail=f"Speaker not found: {request.id}"
            )
        speaker.set_name(request.name)
        speaker.set_gender(request.gender)
        speaker.set_desc(request.describe)
        if (
            request.tensor
            and isinstance(request.tensor, list)
            and len(request.tensor) > 0
        ):
            # number array => Tensor
            token = torch.tensor(request.tensor)
            speaker.set_token(tokens=[token], model_id="chat-tts")
        spk_mgr.update_item(speaker, lambda x: x.id == speaker.id)
        return api_utils.success_response(None)

    @app.post(
        "/v1/speaker/detail", response_model=api_utils.BaseResponse, tags=["Speaker"]
    )
    async def speaker_detail(request: SpeakerDetail):
        speaker = spk_mgr.get_speaker_by_id(request.id)
        if speaker is None:
            raise HTTPException(status_code=404, detail="Speaker not found")
        return api_utils.success_response(
            speaker.to_json(just_info=not request.with_emb)
        )
