from fastapi import HTTPException
from pydantic import BaseModel
import torch
from modules.speaker import speaker_mgr
from modules.api import utils as api_utils
from modules.api.Api import APIManager


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

    @app.get("/v1/speakers/list", response_model=api_utils.BaseResponse)
    async def list_speakers():
        return api_utils.success_response(
            [spk.to_json() for spk in speaker_mgr.list_speakers()]
        )

    @app.post("/v1/speakers/refresh", response_model=api_utils.BaseResponse)
    async def refresh_speakers():
        speaker_mgr.refresh_speakers()
        return api_utils.success_response(None)

    @app.post("/v1/speakers/update", response_model=api_utils.BaseResponse)
    async def update_speakers(request: SpeakersUpdate):
        for spk in request.speakers:
            speaker = speaker_mgr.get_speaker_by_id(spk["id"])
            if speaker is None:
                raise HTTPException(
                    status_code=404, detail=f"Speaker not found: {spk['id']}"
                )
            speaker.name = spk.get("name", speaker.name)
            speaker.gender = spk.get("gender", speaker.gender)
            speaker.describe = spk.get("describe", speaker.describe)
            if (
                spk.get("tensor")
                and isinstance(spk["tensor"], list)
                and len(spk["tensor"]) > 0
            ):
                # number array => Tensor
                speaker.emb = torch.tensor(spk["tensor"])
        speaker_mgr.save_all()

        return api_utils.success_response(None)

    @app.post("/v1/speaker/create", response_model=api_utils.BaseResponse)
    async def create_speaker(request: CreateSpeaker):
        if (
            request.tensor
            and isinstance(request.tensor, list)
            and len(request.tensor) > 0
        ):
            # from tensor
            tensor = torch.tensor(request.tensor)
            speaker = speaker_mgr.create_speaker_from_tensor(
                tensor=tensor,
                name=request.name,
                gender=request.gender,
                describe=request.describe,
            )
        elif request.seed:
            # from seed
            speaker = speaker_mgr.create_speaker_from_seed(
                seed=request.seed,
                name=request.name,
                gender=request.gender,
                describe=request.describe,
            )
        else:
            raise HTTPException(
                status_code=400, detail="Missing tensor or seed in request"
            )
        return api_utils.success_response(speaker.to_json())

    @app.post("/v1/speaker/update", response_model=api_utils.BaseResponse)
    async def update_speaker(request: UpdateSpeaker):
        speaker = speaker_mgr.get_speaker_by_id(request.id)
        if speaker is None:
            raise HTTPException(
                status_code=404, detail=f"Speaker not found: {request.id}"
            )
        speaker.name = request.name
        speaker.gender = request.gender
        speaker.describe = request.describe
        if (
            request.tensor
            and isinstance(request.tensor, list)
            and len(request.tensor) > 0
        ):
            # number array => Tensor
            speaker.emb = torch.tensor(request.tensor)
        speaker_mgr.update_speaker(speaker)
        return api_utils.success_response(None)

    @app.post("/v1/speaker/detail", response_model=api_utils.BaseResponse)
    async def speaker_detail(request: SpeakerDetail):
        speaker = speaker_mgr.get_speaker_by_id(request.id)
        if speaker is None:
            raise HTTPException(status_code=404, detail="Speaker not found")
        return api_utils.success_response(speaker.to_json(with_emb=request.with_emb))
