from modules import config
from modules.api import utils as api_utils
from modules.api.Api import APIManager
from modules.core.handler.datacls.audio_model import AudioFormat


def setup(app: APIManager):

    @app.get(
        "/v1/ping",
        response_model=api_utils.BaseResponse,
        tags=["System"],
        description="Health check",
    )
    async def ping():
        return api_utils.success_response("pong")

    @app.get(
        "/v1/versions",
        response_model=api_utils.BaseResponse,
        tags=["System"],
        description="Get versions",
    )
    async def get_versions():
        return api_utils.success_response(config.versions.to_dict())

    @app.get(
        "/v1/audio_formats",
        response_model=api_utils.BaseResponse,
        tags=["System"],
        description="Get audio encoder formats",
    )
    async def get_audio_formats():
        return api_utils.success_response([e.value for e in AudioFormat])
