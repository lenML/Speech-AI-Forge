from modules.api import utils as api_utils
from modules.api.Api import APIManager


def setup(app: APIManager):
    @app.get("/v1/ping", response_model=api_utils.BaseResponse)
    async def ping():
        return {"message": "ok", "data": "pong"}
