from typing import Optional
import torch
from fastapi import Depends, HTTPException, Query, Request
from pydantic import BaseModel

from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.core.models.tts import ChatTtsModel
from modules.core.spk.SpkMgr import spk_mgr
from modules.core.spk.TTSSpeaker import TTSSpeaker


class AudioReference(BaseModel):
    wav_b64: str
    text: str
    emotion: Optional[str] = "default"


class CreateSpeaker(BaseModel):
    name: str
    gender: str = ""
    author: str = ""
    desc: str = ""
    version: str = ""

    wavs: Optional[list[AudioReference]] = None

    # 是否保存到本地，默认不保存
    save_file: bool = False


class UpdateSpeaker(BaseModel):
    json: Optional[dict] = None


class SpeakerDetail(BaseModel):
    id: str
    with_emb: bool = False


class SpeakersUpdate(BaseModel):
    speakers: list[dict]


class SpkListParams(BaseModel):
    detailed: bool = Query(False, description="Return all detailed big data")

    # page params
    offset: int = Query(0, description="Offset for pagination")
    limit: int = Query(5, description="Limit for pagination")


def setup(app: APIManager):

    @app.get(
        "/v1/speakers/list",
        response_model=api_utils.BaseResponse,
        tags=["Speaker"],
        description="""
List all available speakers with optional pagination and detail control.

- `detailed`: If true, returns complete metadata including references and embeddings.
- `offset` / `limit`: Support for paginated speaker listing.
""",
    )
    async def list_speakers(
        request: Request,
        parmas: SpkListParams = Depends(),
    ):
        detailed = parmas.detailed
        offset = parmas.offset
        limit = parmas.limit

        # NOTE: 因为没有数据库，所以直接拿全部数据然后 slice 即可
        spks = spk_mgr.list_speakers()
        data = [
            spk.to_json(just_info=not detailed) for spk in spks[offset : offset + limit]
        ]

        return api_utils.success_response(
            {
                "items": data,
                "offset": offset,
                "limit": limit,
                "total": len(spks),
            }
        )

    @app.post(
        "/v1/speakers/refresh",
        response_model=api_utils.BaseResponse,
        tags=["Speaker"],
        description="""
Force reload of all speaker metadata from disk.  
Use this when files are modified externally or newly added.
""",
    )
    async def refresh_speakers():
        spk_mgr.refresh()
        return api_utils.success_response(None)

    @app.post(
        "/v1/speakers/update",
        response_model=api_utils.BaseResponse,
        tags=["Speaker"],
        description="""
Batch update multiple speakers by providing a list of speaker JSON configs.

Each speaker must already exist (matched by ID).  
Will overwrite corresponding fields and persist changes to disk.
""",
    )
    async def update_speakers(request: SpeakersUpdate):
        try:
            update_spks = [TTSSpeaker.from_json(cfg) for cfg in request.speakers]

            # check exist
            for spk in update_spks:
                if spk_mgr.get_speaker_by_id(spk.id) is None:
                    raise HTTPException(
                        status_code=404, detail=f"Speaker not found: {spk.id}"
                    )

            for cfg_spk in update_spks:
                spk = spk_mgr.get_speaker_by_id(cfg_spk.id)
                spk.update(cfg_spk)
            spk_mgr.save_all()

            return api_utils.success_response(None)
        except Exception as e:
            import logging

            logging.exception(e)
            if isinstance(e, HTTPException):
                raise e
            else:
                raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/speaker/create",
        response_model=api_utils.BaseResponse,
        tags=["Speaker"],
        description="""
Create a new speaker profile with optional reference audios.

- `name` is required and used as unique identifier.
- `wavs` is a list of audio samples (base64-encoded) and reference texts for embedding.
- `save_file`: If true, the speaker will be saved to disk and available after refresh.
""",
    )
    async def create_speaker(request: CreateSpeaker):
        try:
            spk = TTSSpeaker.empty()
            spk.set_name(request.name)
            spk.set_gender(request.gender)
            spk.set_author(request.author)
            spk.set_desc(request.desc)
            spk.set_version(request.version)

            for wav in request.wavs or []:
                spk_ref = TTSSpeaker.create_spk_ref_from_wav_b64(wav.wav_b64, wav.text)
                spk_ref.emotion = wav.emotion or "default"
                spk.add_ref(ref=spk_ref)

            filepath = spk_mgr.filepath(request.name + ".spkv1.json")

            if request.save_file:
                spk_mgr.save_item(spk, file_path=filepath)
                spk_mgr.refresh()
            return api_utils.success_response(spk.to_json())
        except Exception as e:
            import logging

            logging.exception(e)
            if isinstance(e, HTTPException):
                raise e
            else:
                raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/speaker/update",
        response_model=api_utils.BaseResponse,
        tags=["Speaker"],
        description="""
Update a single speaker's configuration by full JSON override.

The speaker must already exist (matched by ID).  
Fields like name, gender, refs, etc., will be updated accordingly.
""",
    )
    async def update_speaker(request: UpdateSpeaker):
        try:
            cfg_spk = TTSSpeaker.from_json(request.json)
            speaker = spk_mgr.get_speaker_by_id(cfg_spk.id)
            if speaker is None:
                raise HTTPException(
                    status_code=404, detail=f"Speaker not found: {request.id}"
                )
            speaker.update(cfg_spk)
            spk_mgr.update_item(speaker, lambda x: x.id == speaker.id)
            return api_utils.success_response(None)
        except Exception as e:
            import logging

            logging.exception(e)
            if isinstance(e, HTTPException):
                raise e
            else:
                raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/speaker/detail",
        response_model=api_utils.BaseResponse,
        tags=["Speaker"],
        description="""
Fetch metadata of a specific speaker by ID.

- `with_emb`: If true, includes embedding vectors and all reference data.
""",
    )
    async def speaker_detail(request: SpeakerDetail):
        speaker = spk_mgr.get_speaker_by_id(request.id)
        if speaker is None:
            raise HTTPException(
                status_code=404, detail=f"Speaker not found: {request.id}"
            )
        return api_utils.success_response(
            speaker.to_json(just_info=not request.with_emb)
        )
