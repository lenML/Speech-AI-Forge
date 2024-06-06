from modules.api import utils as api_utils
from modules.api.Api import APIManager

from modules import config


def setup(app: APIManager):
    @app.get("/v1/ping", response_model=api_utils.BaseResponse)
    async def ping():
        return api_utils.success_response("pong")

    @app.get("/v1/versions", response_model=api_utils.BaseResponse)
    async def get_versions():
        return api_utils.success_response(config.versions.to_dict())
